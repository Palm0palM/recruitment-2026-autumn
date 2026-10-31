# F3 — HIP startup policy and preserved AMD algorithm

Source: `v3/amd` eff3ed135564dcfcdb7498c3756048b48a0dd9f4, frozen HIP
source SHA256 d9c466af6a18e95eb2e74cea62056fce83dc28f1693bf4e187b10449664fe544.
Rootless ROCm 7.1.1, gfx1201 Radeon AI PRO R9700 on hepnode0.

Preserved: FP32 hipFFT, batch512/256 for small/large FFT, fused Mel, whole-input
async H2D, direct double output, inward-aligned host THP hints, output resize
thread, padded final batch, stream/events. No FFT redesign or MPI.

## Build variants

- `make HIP=1 CUDA=0`: CPU+HIP. No global dynamic HIP initializer. Queue warmup
  begins only inside a selected HIP call, overlapping backend host preparation.
- `make HIP=1 CUDA=0 HIP_ONLY=1`: AMD dedicated. Retains before-main queue
  warmup overlapping protected PCM read/normalization. Auto means HIP, not a
  trial compute; CPU/CUDA requests explicitly rejected. Eager warmup is skipped
  when an unsupported/CPU mode is requested.
- Eager prewarm in CPU+HIP is a build error; no accidental CPU-startup penalty
  from an unconditional HIP thread. A CPU-only binary links no GPU libraries.

Prewarm thread errors are now captured and rethrown on join, with scratch
managed by RAII. Only no-device/insufficient-driver during initialization is
typed BackendUnavailable; real allocation, memset, queue, FFT, kernel or copy
failures are never converted into CPU fallback.

Mixed-build model is calibrated on the LAZY startup policy, not on historical
eager scores. The placeholder profile keeps auto on CPU until the training
session has completed; the final shipped profile must be generated from it.

## Deployment isolation

Read-only ROCm toolchain:
`/home/zsx/melspec-v3-amd/tmp/amd-gpu/root` (no installs or writes there).
V5 AMD deployment and artifacts: `~/melspec-v5-final/amd/`.
Private cache: `$PWD/tmp/rocfft-cache/warm.db`. No cache copied from another
experiment. First cache population runs are recorded separately from steady
timings; every process still pays its own runtime initialization.

Slurm: hepnode0, 1 task, 16 bound CPU cores, exclusive; no GPU GRES and no mem
request. Benchmarks use private node-local V5 scratch, expected checks after
the timer, all repetitions/min/median retained. CUDA deployment is unaffected.
