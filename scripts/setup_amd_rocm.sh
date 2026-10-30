#!/usr/bin/env bash
# Rootless user-space ROCm 7.1/HIP for hepnode0 Radeon AI PRO R9700 (gfx1201).
# Downloads Ubuntu .deb packages and extracts them into a private prefix.
# This does NOT install kernel drivers (amdgpu/KFD are already present) and
# does NOT modify the shared Python environment or /opt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
BASE="${AMD_ROCM_PREFIX:-$ROOT/tmp/amd-gpu}"
PKGS="$BASE/pkgs"
PREFIX="$BASE/root"
mkdir -p "$PKGS" "$PREFIX"

PACKAGES=(
  rocminfo libhsa-runtime64-1 libhsakmt1 librocm-core1
  rocm-smi librocm-smi64-7 libamd-smi26
  libamdhip64-7 libamdhip64-dev libamd-comgr3 rocm-device-libs-21
  hipcc clang-21 llvm-21 lld-21 clang-tools-21
  libclang-rt-21-dev libclang-common-21-dev libstdc++-16-dev
)
if [ ! -x "$PREFIX/usr/bin/hipcc" ]; then
  cd "$PKGS"
  # apt download works without root; packages come from configured Ubuntu/ROCm repos.
  apt download "${PACKAGES[@]}"
  for deb in ./*.deb; do dpkg-deb -x "$deb" "$PREFIX"; done
else
  echo "Reuse extracted ROCm user space: $PREFIX"
fi

DEVICE_LIB="$PREFIX/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode"
cat > "$BASE/env.sh" <<EOF
# Usage: source <worktree>/tmp/amd-gpu/env.sh
export ROCM_PREFIX="$PREFIX"
export HIP_PATH="\$ROCM_PREFIX/usr"
export ROCM_PATH="\$ROCM_PREFIX/usr"
export HIP_DEVICE_LIB_PATH="$DEVICE_LIB"
export PATH="\$ROCM_PREFIX/usr/bin:\$ROCM_PREFIX/usr/lib/llvm-21/bin:\$PATH"
export LD_LIBRARY_PATH="\$ROCM_PREFIX/usr/lib/x86_64-linux-gnu\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LIBRARY_PATH="\$ROCM_PREFIX/usr/lib/x86_64-linux-gnu\${LIBRARY_PATH:+:\$LIBRARY_PATH}"
EOF
chmod +x "$BASE/env.sh"
echo "ROCm prefix: $PREFIX"
echo "source $BASE/env.sh"
echo "HIPCC=$PREFIX/usr/bin/hipcc"
