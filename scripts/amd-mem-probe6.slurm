#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-mem6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:08:00
#SBATCH --output=tmp/logs/amd-mem6-%j.out
#SBATCH --error=tmp/logs/amd-mem6-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
echo "host=$(hostname) job=$SLURM_JOB_ID"
"$HIPCC" --offload-arch="$HIP_ARCH" -O3 \
  --rocm-path="$ROCM_PATH" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
  -I"$ROCM_PATH/include" \
  scripts/mem_probe6.cpp -o tmp/bin/mem_probe6 \
  -L"$ROCM_PATH/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"$ROCM_PATH/lib/x86_64-linux-gnu" -lamdhip64
for round in 1 2; do
  for mode in 0 1 2; do
    tmp/bin/mem_probe6 "$mode" | tee -a "tmp/logs/mem-probe6-${SLURM_JOB_ID}.txt"
  done
done
g++ -O3 -march=native -std=c++17 scripts/hugepage_probe.cpp \
  -o tmp/bin/hugepage_probe
tmp/bin/hugepage_probe | tee "tmp/logs/hugepage-${SLURM_JOB_ID}.txt"
