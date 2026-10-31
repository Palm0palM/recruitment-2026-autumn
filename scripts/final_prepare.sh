#!/bin/bash
# Materialize frozen controls from Git, never from another worktree's tmp/bin.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp/control tmp/bin tmp/build tmp/logs tmp/scratch
git show e448c989c921762884582804efe23d6607678ff9:src/melspec.cpp > tmp/control/cpu-frozen.cpp
git show af56878fc2d750d43791506e2d341c16378d64f9:src/melspec_cpu.cpp > tmp/control/cpu-hybrid.cpp
git show af56878fc2d750d43791506e2d341c16378d64f9:src/melspec.cpp > tmp/control/dispatcher-v4.cpp
git show 3e11194688b4b581b8ca527c5d292dbee451b0f1:src/melspec.cpp > tmp/control/cuda-math.cpp
git show 3e11194688b4b581b8ca527c5d292dbee451b0f1:src/melspec_cuda.cu > tmp/control/cuda-frozen.cu
git show eff3ed135564dcfcdb7498c3756048b48a0dd9f4:src/melspec.cpp > tmp/control/amd-math.cpp
git show eff3ed135564dcfcdb7498c3756048b48a0dd9f4:src/melspec_hip.cu > tmp/control/hip-frozen.cu
git show eff3ed135564dcfcdb7498c3756048b48a0dd9f4:include/melspec_hip.h > tmp/control/melspec_hip.h
git rev-parse HEAD > tmp/source-commit.txt
sha256sum tmp/control/* > tmp/control-sha256.txt
