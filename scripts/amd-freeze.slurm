#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/amd-freeze-%j.out
#SBATCH --error=tmp/logs/amd-freeze-%j.err

# Confirm the frozen source reproduces the validated binary hash and still passes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin tmp/rocfft-cache
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
PY="$HOME/venv/bin/python"
echo "host=$(hostname) job=$SLURM_JOB_ID"
sha256sum src/main.cpp src/melspec.cpp src/melspec_hip.cu include/melspec.h Makefile
make -B -j4 HIP=1 EXEC="tmp/bin/melspec-hip-freeze"
sha256sum tmp/bin/melspec-hip-freeze
scratch=$(mktemp -d "/tmp/melspec-freeze-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u validate.py tmp/bin/melspec-hip-freeze --runs 3 | tee "tmp/logs/freeze-official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/freeze-official-${SLURM_JOB_ID}.txt"
"$PY" -u scripts/smoke.py tmp/bin/melspec-hip-freeze | tee "tmp/logs/freeze-smoke-${SLURM_JOB_ID}.txt"
