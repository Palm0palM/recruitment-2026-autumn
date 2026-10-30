#!/usr/bin/env bash
#SBATCH --job-name=melspec-cuda-output
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/output-sweep-%j.out
#SBATCH --error=tmp/logs/output-sweep-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin tmp/results
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
export CUDA_HOME="${CUDA_HOME:-$PWD/toolkit/cuda-13.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
A2="${A2:-/home/zsx/melspec-v3-cuda-agent/baseline-a2/tmp/bin/a2-estimate}"
PY="$HOME/venv/bin/python"
git rev-parse HEAD
grep Cpus_allowed_list /proc/self/status
programs=()
for spec in stable:0:0:16 defer:1:0:16 async:0:1:16 async_defer:1:1:16 async_r8:1:1:8 async_r32:1:1:32; do
  IFS=: read -r name defer async rows <<< "$spec"
  make -B -j4 CUDA=1 CUDA_HOME="$CUDA_HOME" CUDA_ARCH=89 CUDA_PRECISION=32 \
    CUDA_BATCH=512 CUDA_FUSED=1 CUDA_OVERLAP_INIT=1 CUDA_PINNED=0 CUDA_PROFILE=0 \
    CUDA_DEFER_RESIZE="$defer" CUDA_ASYNC_OUTPUT="$async" CUDA_STAGING_ROWS="$rows" \
    EXEC="tmp/bin/cuda-$name"
  programs+=("$name=tmp/bin/cuda-$name")
done
sha256sum src/main.cpp src/melspec.cpp src/melspec_cuda.cu include/melspec.h include/melspec_cuda.h Makefile tmp/bin/*
scratch=$(mktemp -d "/tmp/melspec-cuda-output-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/cuda_bench.py --programs "a2=$A2" "${programs[@]}" --runs 5 \
  --out "tmp/results/output-sweep-${SLURM_JOB_ID}.json"
