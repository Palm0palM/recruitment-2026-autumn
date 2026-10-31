#!/bin/bash
#SBATCH -J final-hip-guard
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH -w hepnode0
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/final-hip-guard-%j.out
#SBATCH --error=tmp/logs/final-hip-guard-%j.err
set -euo pipefail
if [ "${V5_BOUND:-0}" != 1 ]; then
    exec srun --cpu-bind=cores --distribution=block:block --export=ALL,V5_BOUND=1 bash "$0"
fi
cd "${SLURM_SUBMIT_DIR:?}"
source /home/zsx/melspec-v3-amd/tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0 ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/warm.db"
export AMD_COMGR_CACHE_DIR="$PWD/tmp/comgr-cache" XDG_CACHE_HOME="$PWD/tmp/xdg-cache"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
scratch=$(mktemp -d "/tmp/melspec-v5-final-guard-$SLURM_JOB_ID-XXXXXX")
export TMPDIR="$scratch"
trap 'rmdir "$scratch"' EXIT
PY="$HOME/venv/bin/python"
echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(cat tmp/source-commit.txt)"
grep Cpus_allowed_list /proc/self/status
sha256sum src/* include/* Makefile
make CUDA=0 HIP=0 -B -j8 EXEC=tmp/bin/amd-cpu > tmp/logs/guard-cpu-build-$SLURM_JOB_ID.txt 2>&1
make CUDA=0 HIP=1 -B -j8 EXEC=tmp/bin/final-hip > tmp/logs/guard-hip-build-$SLURM_JOB_ID.txt 2>&1
make CUDA=0 HIP=1 HIP_ONLY=1 -B -j8 EXEC=tmp/bin/dedicated-hip > tmp/logs/guard-dedicated-build-$SLURM_JOB_ID.txt 2>&1
"$PY" -u scripts/final_startup_check.py tmp/bin/final-hip --cpu-only tmp/bin/amd-cpu \
    --gpu hip --out tmp/logs/guard-startup-$SLURM_JOB_ID.json > tmp/logs/guard-startup-$SLURM_JOB_ID.txt
"$PY" -u scripts/final_bench.py cpu=tmp/bin/amd-force-cpu hip=tmp/bin/amd-force-hip \
    auto=tmp/bin/final-hip dedicated=tmp/bin/dedicated-hip --suite official --runs 2 \
    --out tmp/logs/guard-official-$SLURM_JOB_ID.json
for dedicated in 0 1; do
    make CUDA=0 HIP=1 HIP_ONLY="$dedicated" HIP_PROFILE=1 -B -j8 \
        EXEC=tmp/bin/hip-profile-$dedicated > tmp/logs/guard-profile-build-$dedicated-$SLURM_JOB_ID.txt 2>&1
    "$PY" -u scripts/final_gpu_profile.py tmp/bin/hip-profile-$dedicated --backend hip --cases default \
        > tmp/logs/guard-profile-$dedicated-$SLURM_JOB_ID.txt
    grep -q 'async_h2d=1 ' tmp/logs/guard-profile-$dedicated-$SLURM_JOB_ID.txt
    grep -q 'staging_rows=16 ' tmp/logs/guard-profile-$dedicated-$SLURM_JOB_ID.txt
done
sha256sum tmp/bin/amd-cpu tmp/bin/final-hip tmp/bin/dedicated-hip
