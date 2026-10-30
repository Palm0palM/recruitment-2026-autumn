#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-fft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/amd-fft-%j.out
#SBATCH --error=tmp/logs/amd-fft-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
mkdir -p tmp/rocfft-cache
echo "host=$(hostname) job=$SLURM_JOB_ID"
grep Cpus_allowed_list /proc/self/status
"$HIPCC" --offload-arch="$HIP_ARCH" -O3 \
  --rocm-path="$ROCM_PATH" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
  -I"$ROCM_PATH/include" -I"$ROCM_PATH/include/hipfft" \
  scripts/hipfft_probe.cpp -o tmp/bin/hipfft_probe \
  -L"$ROCM_PATH/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"$ROCM_PATH/lib/x86_64-linux-gnu" -lamdhip64 -lhipfft
tmp/bin/hipfft_probe | tee "tmp/logs/hipfft-probe-${SLURM_JOB_ID}.txt"
