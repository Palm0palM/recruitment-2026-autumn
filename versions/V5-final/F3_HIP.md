# F3 — HIP startup policy and preserved AMD algorithm

Source: `v3/amd` eff3ed135564dcfcdb7498c3756048b48a0dd9f4, frozen HIP
source SHA256 d9c466af6a18e95eb2e74cea62056fce83dc28f1693bf4e187b10449664fe544.
Rootless ROCm 7.1.1, gfx1201 Radeon AI PRO R9700 on hepnode0.

Preserved: FP32 hipFFT, batch512/256 for small/large FFT, fused Mel, whole-input
async H2D, direct double output, inward-aligned host THP hints, output resize
thread, padded final batch, stream/events. No FFT redesign or MPI.

## Build variants

- `make HIP=1 CUDA=0`: CPU+HIP. No application-level global HIP queue prewarm. Queue warmup
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
eager scores. The training placeholder has been replaced by the Job 436 profile.
Final auto additionally requires an explicitly configured nonempty rocFFT cache
file before selecting HIP; missing cache stays on CPU without a GPU trial.

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

## Training / integration result (Job 436)

Xeon Gold 6338 + R9700; requested 16 cores, actual srun core-affinity mask
`0-15,64-79` (SMT siblings included); every candidate used that SAME mask.
Do not confuse this with the L40 node's 0-15 mask. 68 independent synthetic
cases vs frozen B16 and librosa PASS. 44 training points, five rounds each,
every measured output checked: all PASS. The private rocFFT cache was created
by this experiment; no previous cache was copied.

Six-round official same-job A/B, medians in seconds:

| Case | Upgraded CPU | Lazy HIP | Dedicated HIP | Frozen AMD | Frozen B16 |
|---|---:|---:|---:|---:|---:|
| default | .69591 | .88043 | .69468 | .69531 | .85769 |
| small_mels | .44148 | .64347 | .46333 | .45839 | .49999 |
| large_fft | .89424 | .93449 | .75209 | .75066 | 1.07416 |
| narrow_band | .85142 | .92534 | .74685 | .74367 | 1.03226 |

The dedicated adapter preserves frozen AMD performance (~0–1% difference).
Lazy startup pays ~180–200 ms that cannot overlap the protected main's input
read. It would be dishonest to inherit the eager benchmark numbers for the
mixed build. For official-sized AMD workloads, recommend dedicated HIP for
large FFTs, CPU-only for small_mels, and retain mixed auto for varied inputs.
For small tasks, neither HIP variant beats SIMD CPU's much smaller startup.

Mixed-profile NNLS on the 44 training points only:

```
Tcpu = .04495945 + .000955393*W + .002640256*I + .001511670*O + .000235520*C
Thip = .32306489 + .000201601*W + .004047886*I + .001580029*O + .000031225*C
```

W/I/O/C as in F2. Training winner accuracy 44/44, leave-one-out 43/44.
CPU MAE .0170 s, HIP MAE .00479 s. Header `melspec_model_hip.h` is generated
from these data, replacing the conservative placeholder. Independent holdout,
real no-device fallback, startup traces, cold-cache and profile evidence
are measured in F4; they never enter the fit.

## Final validation

Job 444: protected official script auto/cpu/hip/dedicated each 4/4 PASS;
14-point holdout all outputs PASS, winner accuracy 13/14 (one 2.52% miss).
CPU and CPU-selected auto traces open no GPU device node. Hidden-device auto
is bitwise CPU-equivalent, forced HIP errors clearly. Private cold DB first
processes were 2.5855 s (8192) / 2.0446 s (16384), versus warm .6985/.7456 s.
Job 445 verifies the final cache guard and corrected diagnostic labels. The
inherited async_h2d/staging_rows printf arguments were swapped in old logs;
only diagnostic argument order changed, not numerical code. Corrected default
prewarm wait is 171.45 ms lazy vs 11.46 ms dedicated.

Known inherited ROCm shutdown warning: SharedSignalPool reports 77 signals.
The frozen Job 395 logs have the same warning (copied to raw/provenance);
it is not hidden or claimed fixed. CPU paths remain clean; long-running
service resource ownership is outside this one-process delivery scope.
