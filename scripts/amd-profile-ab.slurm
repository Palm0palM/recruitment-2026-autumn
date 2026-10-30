#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:15:00
#SBATCH --output=tmp/logs/amd-ab-%j.out
#SBATCH --error=tmp/logs/amd-ab-%j.err

# Profile-only A/B: isolate output.resize contention and pinned-input effect.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin tmp/rocfft-cache
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
PY="$HOME/venv/bin/python"
echo "host=$(hostname) job=$SLURM_JOB_ID"
grep Cpus_allowed_list /proc/self/status
sha256sum src/melspec_hip.cu Makefile

# name:pinned:resize_thread:defer_resize
for spec in base:0:1:0 noresize:0:0:0 defer:0:0:1 pin:1:0:1 pin_rt:1:1:0; do
  IFS=: read -r name pin rt defer <<< "$spec"
  make -B -j4 HIP=1 HIP_PROFILE=1 HIP_PINNED_INPUT="$pin" \
    HIP_RESIZE_THREAD="$rt" HIP_DEFER_RESIZE="$defer" \
    EXEC="tmp/bin/prof-$name" > "tmp/logs/build-$name.txt" 2>&1
  echo "== variant $name pin=$pin resize_thread=$rt defer=$defer ==" | tee -a "tmp/logs/ab-profile-${SLURM_JOB_ID}.txt"
  "$PY" -u scripts/cuda_profile.py "tmp/bin/prof-$name" \
    | tee -a "tmp/logs/ab-profile-${SLURM_JOB_ID}.txt"
done
