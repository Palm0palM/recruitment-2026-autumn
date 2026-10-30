# Four experiments: mathematics, engineering, and interactions

## Shared task semantics

For FFT size N, hop H and K=N/2+1, the unscaled transform is

```
w[i]   = (1 - cos(2*pi*i/N))/2                 # periodic Hann, double
X[t,k] = sum_i x[t*H+i] * w[i] * exp(-2*pi*j*k*i/N)
P[t,k] = real(X[t,k])^2 + imag(X[t,k])^2
S[m,t] = sum_k W[m,k] * P[t,k]                # HTK mel, Slaney area weights
```

Input is raw PCM16 converted to double, not normalized by 32768. Output remains `[mel][frame]`, then
the protected main converts double output to float32 for writing. Sparse traversal skips only exactly-zero
filter weights, retaining each frame's increasing-bin double Mel accumulation order. N=1 preserves the
original zero periodic Hann coefficient. No change to the protected main, validator or task data.

## Wisdom

FFTW plans choose decompositions, copies and ISA-specific codelets. MEASURE/PATIENT spend time benchmarking
algorithm choices; execution does not need that search repeated. The generator receives N and library/CPU
identity, never PCM. Runtime imports an exact matching envelope and asks WISDOM_ONLY|ESTIMATE, so it can
instantiate the premeasured plan without any cold online MEASURE/PATIENT. Unusable wisdom falls back to
ESTIMATE; scored wisdom candidates must explicitly report loaded, not silently take the fallback.

Within a compute call, a single plan is created before thread launch and shared read-only by at most 16
workers using new-array execution with private FFTW-aligned buffers. FFTW execution threading is disabled.
Planner/destructor calls are serialized. Caching a live plan across CLI invocations would require a persistent
service and is not available in this task; persisted wisdom is the useful cross-process reuse mechanism.

Observed float offline planning seconds on hepnode2 (job 151):

| N | MEASURE | PATIENT |
|---|---:|---:|
| 8192 | 0.379044 | 2.294830 |
| 16384 | 0.530984 | 3.628240 |
| 65536 | 1.217840 | 11.885800 |

These costs are NOT hidden inside scored process times: they are explicit prior preparation. If this
preparation were repeated for every timed input, the wisdom deployment conclusion would not apply.
PATIENT is not assumed better merely because it searches longer; compare the complete process directly.
Retained text wisdom and SHA-256 manifest bind the measured planning outcome, not just a regenerable recipe.

## Batch

`plan_many` expresses B independent R2C transforms in one execution call. It reduces call/dispatch overhead
and can expose cross-frame SIMD, but also increases the live working set and changes planner choices.

| Layout | Real input index | Complex output index | FFTW stride/distance |
|---|---|---|---|
| Original rows | b*N+i | b*K+k | istride=1, idist=N, ostride=1, odist=K |
| Padded rows | b*N+i | b*Kpad+k | same except odist=Kpad, Kpad=round_up(K,4) |
| Interleaved | i*B+b | k*B+b | istride=B, idist=1, ostride=B, odist=1 |

Each worker owns its batch; a partial tail batch is zero-filled and only valid frames reach Mel/output.
This is mathematically independent across frames, not a longer FFT or a change to overlap/hop. Unpadded
double-complex rows have K odd for ordinary power-of-two N, shifting each output row's 64-byte address phase.
Padding keeps row phases consistent; interleaving makes adjacent transforms contiguous at a sample/bin.
Neither layout guarantees faster code: pack, strided Mel reads, cache capacity and plan decomposition matter.

Explicit array scratch per worker (excluding FFTW internal plan/execute storage, y, Mel matrix and output):

```
A2 double:       8*N + 24*K
Original batch:  8*B*(N + 3*K)
Padded rows:     8*B*(N + K + 2*Kpad)
Interleaved:     8*B*(N + 3*K)
Float FFT:       4*N + 16*K
```

For N=8192 and 16 workers: A2 is about 2.500 MiB, float 1.500 MiB, original B=4 10.001 MiB,
and original B=32 80.012 MiB. Larger live sets can spill caches; this is an engineering hypothesis consistent
with regressions, not a claim of measured cache/PMU counters. Whole-process peak RSS is a DIFFERENT metric:
the protected main's final write conversion can dominate after FFT scratch has already been destroyed.

## Precision

Only the windowed samples and FFT use float. Both power choices retain double Mel weights/accumulation:

1. Convert float FFT components to double, then form double power.
2. Form power in float, round that float result, then convert to double power storage/Mel.

For spectrum perturbation e, power error is bounded by

```
abs(delta_P) <= 2*abs(X)*abs(e) + abs(e)^2
abs(delta_S[m]) <= sum_k W[m,k] * abs(delta_P[k])   # W >= 0
```

Float power adds its own rounding term; double power cannot undo input quantization/FFT rounding.
Large reference energy permits larger absolute error under the original relative tolerance. Low-energy
bins near cancellation need the absolute term and deliberate adversarial tests; max absolute error alone
does not establish either success or failure. Test the actual final float32 output against the unchanged
budget `abs(error) <= 1e-4 + 1e-3*abs(reference)`.

The 54-case falsification suite spans N=256..65536, exact-period/full-scale/DC+1LSB signals, near-Nyquist,
bin/off-bin and quiet tones, strong+quiet multitone and impulses. It is evidence, not a universal error proof.
Job 151 passed every such comparison for both power choices and MEASURE/PATIENT wisdom. Worst normalized
error was about 0.03483 for MEASURE combination and 0.01574 for PATIENT combination (threshold is 1).
Float is therefore an explicit tested task-domain approximation, NOT a silent change to the default double API.

## ISA/codelets

Enabled SIMD != selected codelets != end-to-end speedup. AVX-512 can reduce instruction count but interact
with cache behavior, decomposition, copies, frequency and 16-worker concurrency. The controlled libraries
use identical scalar `-O3 -march=x86-64-v3`; only enabled codelet sets differ. Application flags are identical.
FFTW adds ISA options to codelet units; explicitly placing -mno-avx512f later in CFLAGS defeated those options
and caused a documented failed build. Plain x86-64-v3 avoids accidental scalar AVX-512 without that override.

Inspect actual `fftw_sprint_plan` output and disassembly, not only configure switches or archive member names.
The AVX2 archive still contains a file named avx512.o, which is not evidence of executing AVX-512 instructions.
Job 126 shows the existing AVX-512-enabled native library is already a good control; AVX2-only regresses.
Single-thread copy+FFT microbenchmarks are not 16-worker Mel/IO process timings.

## Interactions and selection

Wisdom is specific to library, precision, shape/layout/alignment and CPU configuration. Single-transform
double wisdom is not a float or plan-many optimization. Batch can change vectorization and memory traffic;
float can change SIMD width and optimal decomposition; a single-thread planner search may not select the
best kernel under concurrent workers. Thus independent gains cannot be multiplied.

The final experiment directly compares A2, both float power choices, both wisdom rigors and both revised
batch layouts using the same release harness/node. Ranking is geometric mean of four same-job A2-relative
median speedups, not mixed official minima. Only tested combinations are ranked; no global optimum or
exhaustive Cartesian-product search is claimed. Keep a no-wisdom float deployment option and double control.
