#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/amd-smoke-%j.out
#SBATCH --error=tmp/logs/amd-smoke-%j.err

# hepnode0 has no Slurm GPU GRES. A normal exclusive node allocation is the
# only supported way to use the R9700; never run this on the login shell.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/amd-gpu
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0

echo "host=$(hostname) job=$SLURM_JOB_ID"
grep Cpus_allowed_list /proc/self/status
lspci -nn | grep -Ei 'vga|display|amd/ati' || true
"$ROCM_PREFIX/usr/bin/rocminfo" > "tmp/logs/amd-rocminfo-${SLURM_JOB_ID}.txt"
"$ROCM_PREFIX/usr/bin/rocm-smi" --showproductname --showuniqueid \
  --showmeminfo vram > "tmp/logs/amd-rocm-smi-${SLURM_JOB_ID}.txt" 2>&1 || \
  python3 "$ROCM_PREFIX/usr/libexec/rocm_smi/rocm_smi.py" \
    --showproductname --showuniqueid --showmeminfo vram \
    > "tmp/logs/amd-rocm-smi-${SLURM_JOB_ID}.txt"

"$HIPCC" --offload-arch="$HIP_ARCH" -O3 \
  --rocm-path="$ROCM_PATH" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
  -I"$ROCM_PATH/include" -I"$ROCM_PATH/include/hipfft" \
  scripts/hip_probe.cpp -o tmp/amd-gpu/hip_probe \
  -L"$ROCM_PATH/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"$ROCM_PATH/lib/x86_64-linux-gnu" -lamdhip64 -lhipfft
tmp/amd-gpu/hip_probe | tee "tmp/logs/amd-hip-probe-${SLURM_JOB_ID}.txt"
