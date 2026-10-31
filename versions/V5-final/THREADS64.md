# L40 64-worker rapid follow-up

Base: final `cb59189` (merged into master via `bf0fc20`). No FFT, SIMD,
CUDA pipeline, public API, protected CLI, validation or data changes.

## Scope

`make CPU_THREADS=64 L40_THREADS64=1` enables an explicitly limited
four-configuration L40 profile. The original default remains 16 workers;
this experiment does not silently replace the validated master delivery.

- CPU worker count is no longer hard-capped at 16. It is the minimum of
  requested workers, Linux process affinity, and the existing short-input cap.
- Runtime override: `MELSPEC_CPU_THREADS=1..1024` (invalid values error).
- Linux cpuset is respected: requesting 64 in a 16-CPU allocation does not
  launch 64 compute workers.
- Numerical algorithms remain frozen; partition boundaries can change.
- The four-config table chooses only between CPU64 and CUDA. It is generated
  from the first official-sized A/B's measured full-process medians, not a
  fitted general model, not a claim of independent generalization.
- Rules require a 64-worker request and at least 64 allowed CPUs. They match
  only `sr/n_fft/hop/n_mels/f_min/f_max`; no PCM contents or filenames are used.
  Forced CPU/CUDA continue to work. Typed no-device fallback is preserved.
- Other parameters still use the original model, **not recalibrated for 64
  workers**. This expedited profile is validated only for the four measured
  configurations at the official input size; no optimality claim elsewhere.

## Hardware and measurement

hepnode2 has **2 sockets × 32 physical cores**, SMT disabled, four NUMA
domains: 0–15 / 16–31 / 32–47 / 48–63. This is 64 physical cores exposed by
the node, not one socket's 32 cores × SMT2. Job requests 64 CPUs, gets 0–63.

Same job, same source/binary, CPU16 pinned 0–15, CPU32 pinned 0–31,
CPU64 allowed 0–63. Old auto is pinned 0–15 with 16 workers; CUDA is unchanged.
Node-local output; complete process wall time; all output reads/checks outside
timer; every measured output checked against original expected files.

First measurement Job 461, 3 interleaved rounds, medians (s):

| Case | CPU16 | CPU32 | CPU64 | CUDA | Original auto16 | Selected CPU64/CUDA |
|---|---:|---:|---:|---:|---:|---|
| default | .67716 | .69685 | .67612 | .82819 | .68703 | CPU64 |
| small_mels | .43310 | .45573 | .43134 | .56569 | .43841 | CPU64 |
| large_fft | .90986 | .87733 | .88179 | .83369 | .82267 | CUDA |
| narrow_band | .83006 | .86542 | .83628 | .84011 | .82366 | CPU64 |

**More threads did not produce a large end-to-end improvement.** CPU64 vs
CPU16 changes are approximately +0.15% / +0.41% / +3.18% / -0.75% speedup;
the last CPU64/CUDA difference is just 0.46%, within observed run variation.
No assumption of 4× scaling is justified. Whole-process results include serial
CLI decoding/output and NUMA effects; no new phase profile was collected to
attribute the small net changes to one specific bottleneck.

Jobs 459/460 stopped before measurements because the new isolated deployment
lacked its shared-data setting / copy of validate.py. These were deployment
issues, not correctness failures; shared files were never changed.

## Build and reproduce

```bash
# L40 toolkit already available; do not install shared dependencies.
export CUDA_HOME=/home/zsx/melspec-v3-cuda-agent/toolkit/cuda-13.4
make CUDA=1 HIP=0 CPU_THREADS=64 L40_THREADS64=1 CUDA_ARCH=89 \
  EXEC=tmp/bin/final64 -B -j16
# Use a 64-CPU Slurm allocation. Default mode is auto.
MELSPEC_DISPATCH_LOG=1 tmp/bin/final64 data/audio.pcm tmp/output.bin \
  --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 --f_min 0 --f_max 22050

# Measurement and subsequent fixed-rule verification, in a separate deployment:
sbatch scripts/final_threads64.slurm
python scripts/final_threads64_select.py RAW_AB.json --header include/melspec_l40_threads64.h
sbatch scripts/final_threads64_verify.slurm
```

The header is frozen after the first measurement. Subsequent verification is
not fed back into selection. Original 16-worker build:
`make CUDA=1 HIP=0 CPU_THREADS=16 L40_THREADS64=0 ...`.

## Verification

- Local 68 synthetic cases vs frozen CPU and independent librosa: all PASS.
- Local ASan/UBSan, 24 reused-vector/arbitrary-double checks: PASS.
- Local 1/16/32/64 request + affinity limit + invalid override contract: PASS.
- Target-node actual 64-worker contract, 5-round fixed-rule comparison and
  original official validation: results to be appended after job completion.

Evidence: `raw/threads64/`. Original 16-worker delivery results remain in
`FINAL_DELIVERY.md`; do not replace those figures with differently allocated
64-worker results without labeling the resource configuration.
