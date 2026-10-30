#!/usr/bin/env bash
#SBATCH --job-name=melspec-v3-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/cuda-%j.out
#SBATCH --error=tmp/logs/cuda-%j.err

# Run only after the candidate passes local/small correctness tests.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
export CUDA_HOME="${CUDA_HOME:-$PWD/toolkit/cuda-13.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
P="${P:-32}"
ARCH="${ARCH:-89}"
RUNS="${RUNS:-3}"
MODE="${MODE:-bench}"
PY="$HOME/venv/bin/python"

echo "node=$(hostname) job=$SLURM_JOB_ID precision=$P arch=$ARCH mode=$MODE"
git rev-parse HEAD
git status --short
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader
nvcc --version
make -B -j4 CUDA=1 CUDA_HOME="$CUDA_HOME" CUDA_ARCH="$ARCH" CUDA_PRECISION="$P" EXEC="tmp/bin/cuda-fp$P"
sha256sum src/main.cpp src/melspec.cpp src/melspec_cuda.cu include/melspec.h include/melspec_cuda.h Makefile "tmp/bin/cuda-fp$P"
if [ "$MODE" = build ]; then exit 0; fi
if [ "$MODE" = smoke ]; then
  "$PY" -u scripts/smoke.py "./tmp/bin/cuda-fp$P" | tee "tmp/logs/smoke-${SLURM_JOB_ID}.txt"
  exit 0
fi
scratch=$(mktemp -d "/tmp/melspec-cuda-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u validate.py "./tmp/bin/cuda-fp$P" --runs "$RUNS" | tee "tmp/logs/official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/official-${SLURM_JOB_ID}.txt"
