#!/usr/bin/env bash
#SBATCH --job-name=melspec-cuda-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/sweep-%j.out
#SBATCH --error=tmp/logs/sweep-%j.err
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
for spec in 128:1:32:0 256:1:32:0 512:1:32:0 256:0:32:0 256:1:64:0 256:1:32:1 512:1:32:1; do
  IFS=: read -r batch fused precision pinned <<< "$spec"
  name="fp${precision}-b${batch}-f${fused}-h${pinned}"
  make -B -j4 CUDA=1 CUDA_HOME="$CUDA_HOME" CUDA_ARCH=89 CUDA_PRECISION="$precision" \
    CUDA_BATCH="$batch" CUDA_FUSED="$fused" CUDA_PINNED="$pinned" EXEC="tmp/bin/$name"
  programs+=("$name=tmp/bin/$name")
done
sha256sum src/main.cpp src/melspec.cpp src/melspec_cuda.cu include/melspec.h include/melspec_cuda.h Makefile tmp/bin/* "$A2"
scratch=$(mktemp -d "/tmp/melspec-cuda-sweep-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/cuda_bench.py --programs "a2=$A2" "${programs[@]}" --runs 3 \
  --out "tmp/results/sweep-${SLURM_JOB_ID}.json"
