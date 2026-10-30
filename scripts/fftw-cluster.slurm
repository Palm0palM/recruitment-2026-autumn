#!/usr/bin/env bash
#SBATCH --job-name=melspec-fftw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=tmp/logs/fftw-%j.out
#SBATCH --error=tmp/logs/fftw-%j.err

# One job at a time. MODE=build only bootstraps FFTW; MODE=bench compares all
# candidates, then calls the COMPLETE protected official validator on A2.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
export FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11"
PY="${PY:-$HOME/venv/bin/python}"
BASE=15f90558b4ab83ad0e6500e84aa807a942564229
echo "job=$SLURM_JOB_ID node=$(hostname) mode=${MODE:-bench}"
git rev-parse HEAD
git status --short
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E 'Model name|^CPU\(s\)'
g++ --version
mkdir -p tmp/logs tmp/bin tmp/build
JOBS=8 bash scripts/build_fftw.sh > "tmp/logs/fftw-build-${SLURM_JOB_ID}.txt" 2>&1
sha256sum "$FFTW_PREFIX/lib/libfftw3.a" "$FFTW_PREFIX/include/fftw3.h"
grep -E '^(#define PACKAGE_VERSION|#define HAVE_AVX)' tmp/deps/build-fftw/config.h || true
if [ "${MODE:-bench}" = build ]; then exit 0; fi
scratch=$(mktemp -d "/tmp/fftw-agent-a-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
if [ "${MODE:-bench}" = plans ]; then
  g++ -std=c++17 -O3 -march=native -I"$FFTW_PREFIX/include" scripts/fftw_plan_probe.cpp \
    "$FFTW_PREFIX/lib/libfftw3.a" -lm -o tmp/bin/plan-probe
  tmp/bin/plan-probe
  exit 0
fi
if [ "${MODE:-bench}" = profile ]; then
  sha256sum src/melspec.cpp include/fftw_support.h scripts/fftw_profile.py
  "$PY" -u scripts/fftw_profile.py --mode r2c --planner estimate
  exit 0
fi
git show "$BASE:src/melspec.cpp" > tmp/build/v1.cpp
g++ -std=c++17 -O3 -march=native -pthread -Iinclude src/main.cpp tmp/build/v1.cpp -o tmp/bin/v1
for mode in c2c r2c; do
  for planner in estimate measure; do
    make -B -j4 FFT_MODE="$mode" FFTW_PLAN="$planner" EXEC="tmp/bin/$mode-$planner"
  done
done
sha256sum src/melspec.cpp include/fftw_support.h Makefile tmp/build/v1.cpp tmp/bin/*
"$PY" -u scripts/fftw_bench.py --runs 3 --out "tmp/logs/compare-${SLURM_JOB_ID}.json"
echo '== complete official validate.py, A2 ESTIMATE, 3 runs =='
"$PY" -u validate.py ./tmp/bin/r2c-estimate --runs 3 | tee "tmp/logs/official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/official-${SLURM_JOB_ID}.txt"
