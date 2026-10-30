#!/usr/bin/env bash
#SBATCH --job-name=melspec-v3-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/amd-%j.out
#SBATCH --error=tmp/logs/amd-%j.err

# hepnode0 has no GPU GRES; normal exclusive allocation is the access mechanism.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin tmp/rocfft-cache
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
PY="$HOME/venv/bin/python"
MODE="${MODE:-bench}"
P="${P:-32}"
RUNS="${RUNS:-3}"

echo "host=$(hostname) job=$SLURM_JOB_ID mode=$MODE"
grep Cpus_allowed_list /proc/self/status
"$ROCM_PREFIX/usr/bin/rocminfo" | grep -E "Name:|Marketing Name:|gfx1201" | head -10 || true
sha256sum src/main.cpp src/melspec.cpp src/melspec_hip.cu include/melspec.h include/melspec_hip.h Makefile
make -B -j4 HIP=1 HIP_PRECISION="$P" EXEC="tmp/bin/melspec-hip"
sha256sum "tmp/bin/melspec-hip"

scratch=$(mktemp -d "/tmp/melspec-amd-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"

if [ "$MODE" = smoke ]; then
  "$PY" -u scripts/smoke.py "tmp/bin/melspec-hip" | tee "tmp/logs/smoke-${SLURM_JOB_ID}.txt"
  exit 0
fi

if [ "$MODE" = profile ]; then
  "$PY" -u scripts/cuda_profile.py "tmp/bin/melspec-hip" | tee "tmp/logs/profile-${SLURM_JOB_ID}.txt"
  exit 0
fi

# Populate rocFFT's persistent JIT cache and tail plans before scored runs.
for case in default small_mels large_fft narrow_band; do
  "$PY" -u scripts/bench.py "tmp/bin/melspec-hip" --case "$case" --runs 1 \
    --no-librosa --out "tmp/logs/prewarm-${case}-${SLURM_JOB_ID}.md" >/dev/null
done
echo "== official validate.py --runs $RUNS =="
"$PY" -u validate.py "tmp/bin/melspec-hip" --runs "$RUNS" | tee "tmp/logs/official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/official-${SLURM_JOB_ID}.txt"
