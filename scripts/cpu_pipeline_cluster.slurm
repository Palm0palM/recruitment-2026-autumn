#!/usr/bin/env bash
#SBATCH --job-name=melspec-v3-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=tmp/logs/cpu-%j.out
#SBATCH --error=tmp/logs/cpu-%j.err

# One V3 CPU job at a time. MODE=build|bench|profile. No --mem.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
export FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11"
PY="${PY:-$HOME/venv/bin/python}"
BASE=b4552f3962751b5a02d2aa56798907c5a1c29afc
MODE="${MODE:-bench}"
echo "job=$SLURM_JOB_ID node=$(hostname) mode=$MODE"
git rev-parse HEAD
git status --short
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E 'Model name|^CPU\(s\)'
g++ --version
mkdir -p tmp/logs tmp/bin tmp/build
JOBS=8 bash scripts/build_fftw.sh > "tmp/logs/fftw-build-${SLURM_JOB_ID}.txt" 2>&1
sha256sum "$FFTW_PREFIX/lib/libfftw3.a" "$FFTW_PREFIX/include/fftw3.h"
grep -E '^(#define PACKAGE_VERSION|#define HAVE_AVX)' tmp/deps/build-fftw/config.h || true
if [ "$MODE" = build ]; then exit 0; fi
scratch=$(mktemp -d "/tmp/melspec-v3-cpu-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
if [ "$MODE" = profile ]; then
  sha256sum src/melspec.cpp include/fftw_support.h Makefile scripts/fftw_profile.py
  "$PY" -u scripts/fftw_profile.py --mode r2c --planner estimate --frame-block "${PROFILE_BLOCK:-1}"
  exit 0
fi
# Rebuild the fixed, verified A2 baseline on this exact node; never run V0.
git show "$BASE:src/melspec.cpp" > tmp/build/a2.cpp
g++ -std=c++17 -O3 -march=native -pthread -Iinclude -I"$FFTW_PREFIX/include" \
  src/main.cpp tmp/build/a2.cpp "$FFTW_PREFIX/lib/libfftw3.a" -lm -o tmp/bin/a2-base
programs=(a2-base=tmp/bin/a2-base)
for block in ${CANDIDATES:-1 2 4 8 16}; do
  make -B -j4 FFT_MODE=r2c FFTW_PLAN=estimate FRAME_BLOCK="$block" EXEC="tmp/bin/b$block"
  programs+=("b$block=tmp/bin/b$block")
done
sha256sum src/melspec.cpp include/fftw_support.h Makefile tmp/build/a2.cpp tmp/bin/*
"$PY" -u scripts/fftw_bench.py --programs "${programs[@]}" --runs 3 --out "tmp/logs/compare-${SLURM_JOB_ID}.json"
official="${OFFICIAL:-b1}"
echo "== complete official validate.py, $official, 3 runs =="
"$PY" -u validate.py "./tmp/bin/$official" --runs 3 | tee "tmp/logs/official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/official-${SLURM_JOB_ID}.txt"
