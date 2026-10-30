#!/usr/bin/env bash
#SBATCH --job-name=melspec-cuda-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00
#SBATCH --output=smoke-%j.out
#SBATCH --error=smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CUDA_HOME="$PWD/toolkit/cuda-13.4"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
nvcc -O2 -std=c++17 -arch=sm_89 \
  -I"$CUDA_HOME/targets/x86_64-linux/include" \
  cuda-smoke.cu -o cuda-smoke \
  -L"$CUDA_HOME/lib64" -Xlinker -rpath,"$CUDA_HOME/lib64" -lcufft -lcudart
ldd ./cuda-smoke
./cuda-smoke
