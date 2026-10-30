#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:05:00
#SBATCH --output=tmp/logs/amd-mem-%j.out
#SBATCH --error=tmp/logs/amd-mem-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
echo "host=$(hostname) job=$SLURM_JOB_ID"
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E "Model name|NUMA node|Socket" | head -8
lspci -vv 2>/dev/null | grep -A3 "AMD/ATI" | grep -E "LnkCap|LnkSta" | head -6 || true
"$HIPCC" --offload-arch="$HIP_ARCH" -O3 \
  --rocm-path="$ROCM_PATH" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
  -I"$ROCM_PATH/include" -I"$ROCM_PATH/include/hipfft" \
  scripts/mem_probe.cpp -o tmp/bin/mem_probe \
  -L"$ROCM_PATH/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"$ROCM_PATH/lib/x86_64-linux-gnu" -lamdhip64 -lhipfft
tmp/bin/mem_probe | tee "tmp/logs/mem-probe-${SLURM_JOB_ID}.txt"
