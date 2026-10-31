# V5 integration / conflict analysis (before code changes)

Base: `origin/v4/hybrid` = `af56878fc2d750d43791506e2d341c16378d64f9`.
All four source worktrees were present, clean, and read-only during inspection.
No branch merge or cherry-pick is used.

| Source | Identity | Adopt / boundary |
|---|---|---|
| exp/cpu-pipeline-simd | e14647f, algorithm cbf214d | Selective slab reuse, inward-aligned THP advice, empty-batch UB fix only. Preserve B16/B8/FP64 and all FFT arithmetic. |
| v3/cuda | branch 58bca04, algorithm 3e111946 | Keep current hybrid CUDA file byte-identical (only original backend availability adapter differs from frozen source). |
| v3/amd | eff3ed1, frozen source in that commit | FP32 hipFFT 512/256, fused Mel, direct-double output, async H2D, THP, resize overlap. Separate prewarm policy from numerical algorithm. |
| v4/hybrid | af56878 | Dispatcher/private backend interfaces and cost features retained; coefficients must be recalibrated. |

## Conflicts and decisions

1. CPU patch targets src/melspec.cpp in its experiment; port its exact hunks to
   src/melspec_cpu.cpp while retaining the renamed CPU entry and private header.
   Both performance flags explicitly enabled by the final build, individually
   switchable for A/B. Safety fix unconditional. B8/generic regression required.
2. CPU-pipeline published timing used NUMA-local affinity, while old hybrid used
   scattered cpusets. This integration records affinity, binds consistently,
   and compares frozen and final binaries in the SAME allocation.
3. Old hybrid timed() calls run(), which reads output with numpy before stopping
   the clock. Correct this before any new measurement: subprocess wall only;
   correctness readback outside the timed interval. Historical numbers are not
   a calibration baseline for V5.
   Initial V5 Job 419 used home/ZFS output scratch; archive it separately as a
   storage diagnostic. Production calibration uses V5-prefixed node-local
   scratch linked under tmp/scratch/, same storage for every A/B participant.
4. HIP global dynamic-initialization prewarm creates queues before Dispatcher
   can decide. Default CPU+HIP disables global prewarm; explicit AMD dedicated
   build retains it. Measure the performance penalty, do not claim the lazy
   build inherits the old AMD score. CPU predictions must never invoke HIP APIs.
5. rocFFT JIT cache must be private to V5 and explicitly configured. Record cold
   and warm runs separately; no cache/database/binary committed.
6. CPU-only, CPU+CUDA and CPU+HIP are supported variants. CUDA+HIP simultaneous
   linkage rejected. No GPU work on login nodes; NVIDIA and AMD tests serial.
7. A selected GPU compute failure propagates. Only device unavailability allows
   auto fallback. No speculative full compute, no filename/content dispatch.

## Stages / acceptance

- F1: CPU patch, independent reference+librosa, strict sanitizer/API, B8/generic.
- F2: L40 old hybrid/frozen CUDA vs upgraded CPU/unified forced modes; new
  38-point model, holdout auto evaluation, independent stable commit.
- F3: HIP adapter, lazy CPU+HIP and dedicated prewarm builds; AMD official and
  representative synthetic tests, private cache, source/profile evidence.
- F4: per-platform same-job frozen vs final A/B; final model and deployment
  recommendations in FINAL_DELIVERY.md. Protected hashes and clean git audit.

## F2 adapter refinement after same-job evidence

Job 422 found 65–90 ms extra end-to-end CUDA startup from the inherited serial
cudaGetDeviceCount pre-probe (frozen CUDA 0.787/0.804 s vs integrated
0.878/0.870 s for default/large). Remove that pre-probe for actual FFTs.
At the existing overlapped context-initialization boundary, classify ONLY
cudaErrorNoDevice and cudaErrorInsufficientDriver as BackendUnavailable.
Auto may catch only that type and fall back before any GPU kernel; other
initialization, allocation, FFT, transfer and kernel errors still propagate.
CUDA numerical kernels, batch/plan, host conversions, buffer/resize and stream
pipeline stay frozen. No background allocation or sprint experiments added.
The n_fft=1 forced-CUDA mathematical shortcut still probes (otherwise it
would claim success without any device). Re-measure the adapter and fit only
post-refinement data. Job 422 remains an explicit before-adapter control.

## Completed checkpoints

- F1: acfe1c6, ported CPU; strict B8/generic and remote B16 correctness/sanitizer.
- F2: 8e1659a + 45b4093, L40 44 training points and independent 12-point holdout,
  three-mode official 4/4. Independently deliverable without any HIP dependency.
- F3: 04962ce + c00b3eb, R9700 lazy/dedicated variants, 68 independent cases,
  44-point calibration, frozen AMD same-job comparison.
- F4: Jobs 443/444, final 14-point holdouts, all protected official modes,
  real unavailability, strace, profiles, private cold/warm cache comparison.
- Final safety/diagnostic refinement 4227f51, verified by Job 445: missing
  rocFFT cache keeps mixed auto on CPU, diagnostic printf labels corrected,
  invalid direct-double/synchronous-output build combination rejected.
  CPU-only/CUDA preprocessing still exactly matches c00b3eb.
