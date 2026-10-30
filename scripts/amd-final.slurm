#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:30:00
#SBATCH --output=tmp/logs/amd-final-%j.out
#SBATCH --error=tmp/logs/amd-final-%j.err

# G4: same-node interleaved HIP vs SIMD FP32 B16, official validation, profile.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin tmp/results tmp/rocfft-cache
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
PY="$HOME/venv/bin/python"
echo "host=$(hostname) job=$SLURM_JOB_ID"
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E "Model name|NUMA node\(s\)|Socket"
cat /sys/kernel/mm/transparent_hugepage/enabled

echo "== hashes =="
sha256sum src/main.cpp src/melspec.cpp src/melspec_hip.cu include/melspec.h Makefile \
  tmp/baseline-b16/src/main.cpp tmp/baseline-b16/src/melspec.cpp

echo "== build HIP candidate =="
make -B -j4 HIP=1 EXEC="tmp/bin/melspec-hip"
sha256sum tmp/bin/melspec-hip

echo "== build SIMD FP32 B16 baseline (frozen sources) =="
g++ -std=c++17 -O3 -march=native -pthread -Itmp/baseline-b16/include \
  -DMELSPEC_FP32_BATCH=16 tmp/baseline-b16/src/main.cpp tmp/baseline-b16/src/melspec.cpp \
  -o tmp/bin/melspec-b16-frozen
sha256sum tmp/bin/melspec-b16-frozen

scratch=$(mktemp -d "/tmp/melspec-final-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"

echo "== interleaved benchmark (HIP vs SIMD B16) =="
"$PY" -u scripts/cuda_bench.py \
  --programs hip=tmp/bin/melspec-hip b16=tmp/bin/melspec-b16-frozen \
  --runs "${RUNS:-6}" --out "tmp/results/amd-final-${SLURM_JOB_ID}.json" \
  | tee "tmp/logs/final-bench-${SLURM_JOB_ID}.txt"

echo "== smoke suite =="
"$PY" -u scripts/smoke.py tmp/bin/melspec-hip | tee "tmp/logs/final-smoke-${SLURM_JOB_ID}.txt"

echo "== official validate.py --runs 3 =="
"$PY" -u validate.py tmp/bin/melspec-hip --runs 3 \
  | tee "tmp/logs/final-official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/final-official-${SLURM_JOB_ID}.txt"

echo "== profile =="
make -B -j4 HIP=1 HIP_PROFILE=1 EXEC="tmp/bin/melspec-hip-profile"
"$PY" -u scripts/cuda_profile.py tmp/bin/melspec-hip-profile \
  | tee "tmp/logs/final-profile-${SLURM_JOB_ID}.txt"
