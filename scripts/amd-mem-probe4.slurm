#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-mem4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:05:00
#SBATCH --output=tmp/logs/amd-mem4-%j.out
#SBATCH --error=tmp/logs/amd-mem4-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
echo "host=$(hostname) job=$SLURM_JOB_ID"
"$HIPCC" --offload-arch="$HIP_ARCH" -O3 \
  --rocm-path="$ROCM_PATH" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
  -I"$ROCM_PATH/include" \
  scripts/mem_probe4.cpp -o tmp/bin/mem_probe4 \
  -L"$ROCM_PATH/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"$ROCM_PATH/lib/x86_64-linux-gnu" -lamdhip64
for c in 1 2 3 4 5 6 7; do
  tmp/bin/mem_probe4 "$c" | tee -a "tmp/logs/mem-probe4-${SLURM_JOB_ID}.txt"
done
