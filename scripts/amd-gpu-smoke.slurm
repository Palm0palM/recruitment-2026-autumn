#!/usr/bin/env bash
#SBATCH --job-name=amd-r9700-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/amd-r9700-%j.out
#SBATCH --error=tmp/logs/amd-r9700-%j.err

# Opportunistic use only: node0 has NO Slurm GRES entry for this GPU.
# First ordinary job to occupy node0 gets the device; do not run on the login shell.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs tmp/amd-gpu
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$(grep Cpus_allowed_list /proc/self/status)"
lspci -nn | grep -Ei 'vga|display|amd/ati' || true
"$ROCM_PREFIX/usr/bin/rocminfo" > "tmp/logs/amd-rocminfo-${SLURM_JOB_ID}.txt"
python3 "$ROCM_PREFIX/usr/libexec/rocm_smi/rocm_smi.py" \
  --showproductname --showuniqueid --showmeminfo vram --showhw \
  > "tmp/logs/amd-rocm-smi-${SLURM_JOB_ID}.txt"
"$ROCM_PREFIX/usr/bin/hipcc" --offload-arch=gfx1201 -O3 \
  --rocm-path="$ROCM_PATH" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
  scripts/hip_probe.cpp -o tmp/amd-gpu/hip_probe
tmp/amd-gpu/hip_probe | tee "tmp/logs/amd-hip-probe-${SLURM_JOB_ID}.txt"
