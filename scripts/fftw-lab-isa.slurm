#!/usr/bin/env bash
#SBATCH --job-name=fftw-lab-isa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=tmp/logs/isa-%j.out
#SBATCH --error=tmp/logs/isa-%j.err
# Orchestrator only: launch only after the previous benchmark job finishes.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
PY="$HOME/venv/bin/python"
O=tmp/fftw-lab/isa-comparison
E=versions/V2-fftw/experiments
mkdir -p "$O"
echo "job=$SLURM_JOB_ID node=$(hostname)"
git rev-parse HEAD
git status --short
grep Cpus_allowed_list /proc/self/status
g++ --version
# Both controlled libraries have identical scalar flags. Only enabled SIMD
# codelets differ. The historical native library is a third, separate control.
for isa in avx2 avx512; do
  VARIANT="$isa" PRECISION=double JOBS=8 bash "$E/isa/build_fftw_variant.sh" \
    > "tmp/logs/isa-build-$isa-$SLURM_JOB_ID.txt" 2>&1
  prefix="$PWD/tmp/fftw-lab/isa/work/install-$isa-double"
  g++ -std=c++17 -O3 -march=native -pthread -DMELSPEC_FFTW_R2C=1 -Iinclude -I"$prefix/include" \
    src/main.cpp src/melspec.cpp "$prefix/lib/libfftw3.a" -lm -o "$O/a2-$isa"
  g++ -std=c++17 -O3 -march=native -pthread -I"$prefix/include" \
    "$E/isa/isa_plan_probe.cpp" "$prefix/lib/libfftw3.a" -lm -o "$O/probe-$isa"
  sha256sum "$prefix/lib/libfftw3.a" "$O/a2-$isa"
  for n in 8192 16384; do
    for planner in estimate measure; do
      "$O/probe-$isa" --n "$n" --planner "$planner" \
        > "tmp/logs/isa-plan-$isa-$n-$planner-$SLURM_JOB_ID.txt"
    done
  done
done
prefix="$PWD/tmp/opt/fftw-3.3.11"
g++ -std=c++17 -O3 -march=native -pthread -DMELSPEC_FFTW_BATCH=4 -Iinclude -I"$prefix/include" \
  src/main.cpp "$E/batch/replacement.cpp" "$prefix/lib/libfftw3.a" -lm -o "$O/batch4-native"
scratch=$(mktemp -d "/tmp/fftw-isa-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" scripts/fftw_check.py --reference tmp/fftw-lab/orchestrator/a2 "$O/a2-avx2" "$O/a2-avx512" "$O/batch4-native"
"$PY" -u scripts/fftw_bench.py --runs 3 --programs \
  a2=tmp/fftw-lab/orchestrator/a2 "avx2=$O/a2-avx2" "avx512=$O/a2-avx512" \
  "batch4=$O/batch4-native" float-fp=tmp/fftw-lab/orchestrator/float-0 \
  --out "tmp/logs/isa-compare-$SLURM_JOB_ID.json"
"$PY" scripts/fftw_lab_memory.py --programs a2=tmp/fftw-lab/orchestrator/a2 \
  "avx2=$O/a2-avx2" "avx512=$O/a2-avx512" "batch4=$O/batch4-native" \
  --out "tmp/logs/isa-memory-$SLURM_JOB_ID.json"
