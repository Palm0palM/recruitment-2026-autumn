# Standalone functional test record

Date: 2026-10-30. Local functional test only; no benchmark, production binary,
official validation, remote operation, or Slurm job was run.

Working directory for every command:

```text
/home/palm/projects/heptagon/melspec-v3-cuda/tmp/v3-cuda-lab/pipeline
```

Toolchain query:

```bash
command -v nvcc && nvcc --version
```

Result:

```text
/opt/cuda/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Sep_01_08:45:21_PDT_2026
Cuda compilation tools, release 13.4, V13.4.92
Build cuda_13.4.r13.4/compiler.38855100_0
```

Build command:

```bash
mkdir -p build && nvcc -std=c++17 -O2 -arch=native -I. correctness.cu -o build/correctness -lcufft
```

Build completed with no compiler output and exit status 0.

Run command:

```bash
./build/correctness
```

Exit status 0; complete stdout:

```text
float: n_fft=256 hop=128 frames=13 batch=5 tail=3
  separate native              PASS  max_abs=1.148e-06 failures=0
  separate host float output   PASS  max_abs=1.148e-06 failures=0
  fused native                 PASS  max_abs=1.148e-06 failures=0
  fused host float output      PASS  max_abs=1.148e-06 failures=0
double: n_fft=256 hop=128 frames=13 batch=5 tail=3
  separate native              PASS  max_abs=4.441e-15 failures=0
  separate host float output   PASS  max_abs=1.947e-07 failures=0
  fused native                 PASS  max_abs=4.441e-15 failures=0
  fused host float output      PASS  max_abs=1.947e-07 failures=0
GPU pipeline correctness: PASS
```

These are numerical errors against the harness's direct-DFT CPU reference, not
performance results.
