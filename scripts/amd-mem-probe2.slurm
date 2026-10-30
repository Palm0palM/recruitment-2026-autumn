#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-mem2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:05:00
#SBATCH --output=tmp/logs/amd-mem2-%j.out
#SBATCH --error=tmp/logs/amd-mem2-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L 2>/dev/null || true
"$HIPCC" --offload-arch="$HIP_ARCH" -O3 \
  --rocm-path="$ROCM_PATH" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
  -I"$ROCM_PATH/include" \
  scripts/mem_probe2.cpp -o tmp/bin/mem_probe2 \
  -L"$ROCM_PATH/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"$ROCM_PATH/lib/x86_64-linux-gnu" -lamdhip64
tmp/bin/mem_probe2 | tee "tmp/logs/mem-probe2-${SLURM_JOB_ID}.txt"
