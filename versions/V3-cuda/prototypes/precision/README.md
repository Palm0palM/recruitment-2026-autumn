# V3 CUDA precision/correctness and memory plan

This directory is a **correctness-only prototype harness**. It does not implement a
CUDA candidate, change production code, time execution, or make a performance claim.
The acceptance comparison is always

```text
np.allclose(candidate, reference, rtol=1e-3, atol=1e-4, equal_nan=False)
```

per element, with the original shape and float32 output contract. Neither tolerance
is configurable.

## Files

- `precision_harness.py`: deterministic synthetic tests, V1/librosa comparisons,
  and memory-bounded official-audio output checks.
- `memory_plan.py`: conservative out-of-place CUDA capacity arithmetic.
- `test_local.py`: CPU-only helper and contract unit tests.
- `python.sh`: uses the existing environment and redirects all Python/Numba/MPL/temp
  caches into this task directory.
- `TEST_RESULTS.md`: exact local commands and observed results.

Generated PCM, outputs, reports, caches, and runs stay below this directory.

## 1. Correctness contract and oracle policy

The candidate command must remain:

```text
PROGRAM INPUT_PCM OUTPUT_BIN --sr SR --n_fft N --hop_length H \
  --n_mels M --f_min FMIN --f_max FMAX
```

Input is signed little-endian int16 mono and is normalized by `1/32768`. There is
no centering or end padding:

```text
F = 1 + floor((N - n_fft) / hop),   N >= n_fft
frame[t] = y[t*hop : t*hop+n_fft]
```

Output must contain exactly `4*n_mels*F` bytes: little-endian float32, row-major
`[n_mels, F]`. Extra bytes, truncation, NaN, a process failure, or a timeout fails.
The harness performs no retries and records each exact command, return code,
stdout/stderr, shape, first failure, and error metrics in `summary.json`.

For synthetic cases:

1. **librosa is the official oracle**, using float32 PCM normalized by 32768,
   `center=False`, `htk=True`, and `power=2.0`.
2. The fixed V1 executable (`ROOT/melspec` by default) is a second diagnostic
   oracle. It is executed on exactly the same PCM.
3. If V1 and librosa agree at the unchanged tolerances, the candidate must match
   both. If the references themselves differ, the report says `ORACLE-DIFF`;
   librosa still decides official correctness. This is not a tolerance waiver: a
   candidate cannot be required to match two incompatible outputs simultaneously.
4. `--skip-v1` exists for isolation/debugging, but the main agent should run the
   full suite with V1 enabled.

For official audio, the supplied expected `.bin` files are the oracle. They are
read through chunked memmaps; librosa is not recomputed and no timing is done.

### Length-one/length-two warning

V1 evaluates `0.5*(1-cos(2*pi*n/n_fft))`, hence its `n_fft=1` window is zero.
SciPy/librosa's special length-one periodic Hann window is one. The unit test
records this real algorithmic distinction. With ordinary `f_min=0` HTK triangles,
DC is on the left edge and often has zero weight; for very small transforms many
or all filters can be empty. Thus an all-zero `n_fft=1/2` Mel output can hide the
window/FFT difference. These tests still check framing, shape, finite output, and
robustness, but are **not sufficient evidence of FFT accuracy**. `n_fft=8`, the
large synthetic cases, impulses, and official audio must also pass.

## 2. Experiment matrix

The default synthetic suite has 74 cases. `--quick` has 30 and is only a smoke
test. Every random input is deterministic and generated as full-range int16 PCM,
not as content-dependent audio.

### Core transforms: `n_fft = 1, 2, 8`

For each transform size, the full suite includes:

- seeded random PCM;
- exact zeros;
- low amplitude PCM (`0, +/-1, +/-2` LSB);
- constant `32767` (positive full scale) and `-32768` (negative full scale);
- constant `16384` DC;
- alternating `32767/-32768` Nyquist;
- alternating-sign impulses at the first/second/center/last window samples and
  around hop/frame boundaries;
- lengths `n_fft`, `n_fft+hop-1`, `n_fft+hop`, and `n_fft+2*hop`, checking exact
  frame counts and discarded incomplete tails (duplicate lengths are deduplicated).

Core parameters use `sr=96000`, `n_mels=8`, `f_min=0`, `f_max=22050`, with hops
1, 1, and 3 respectively.

### Official parameter sets on bounded synthetic inputs

All four official configurations are applied to the same signal families and to
one/two-frame exact-boundary inputs. Synthetic signals have at most three frames,
so FP64 librosa references fit comfortably in the local host's approximately
15 GiB RAM:

| name | n_fft | hop | n_mels | f_min | f_max |
|---|---:|---:|---:|---:|---:|
| default | 8192 | 128 | 256 | 0 | 22050 |
| small_mels | 8192 | 128 | 64 | 0 | 22050 |
| large_fft | 16384 | 128 | 256 | 0 | 22050 |
| narrow_band | 16384 | 128 | 256 | 300 | 8000 |

Finally, `--suite official` runs all four full official inputs and supplied
references without computing librosa or benchmarking.

## 3. Exact tests for the main agent

From `/home/palm/projects/heptagon/melspec-v3-cuda`, substitute the actual candidate
paths. Run FP64 first as a pipeline/layout diagnostic, then FP32. A quick run is
useful only before the full run.

```bash
# CPU helper tests
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/test_local.py

# Optional smoke test, not signoff
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py /path/to/melspec-cuda-fp64 --quick

# Required bounded synthetic suite, both precisions
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py /path/to/melspec-cuda-fp64
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py /path/to/melspec-cuda-fp32

# Required full official inputs, chunked comparison and no timing in this harness
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py /path/to/melspec-cuda-fp64 \
  --suite official --timeout 3600
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py /path/to/melspec-cuda-fp32 \
  --suite official --timeout 3600
```

Use `--case REGEX` and `--fail-fast` for debugging, then rerun the unfiltered suite.
For example:

```bash
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py /path/to/melspec-cuda-fp32 \
  --case 'core/nfft8|official-params/narrow_band' --fail-fast
```

After both correctness modes pass, the main agent alone should run the unchanged
official validator (it also times execution):

```bash
./tmp/v3-cuda-lab/precision/python.sh validate.py /path/to/melspec-cuda-fp32 --runs 1
./tmp/v3-cuda-lab/precision/python.sh validate.py /path/to/melspec-cuda-fp64 --runs 1
```

The official validator imports and executes librosa for timing and therefore needs
substantially more host memory than this harness's official mode. Do not interpret
a synthetic-only pass as an official pass.

## 4. Near-zero and error propagation

For reference value `r` and candidate `c`, acceptance is elementwise:

```text
|c-r| <= 1e-4 + 1e-3*|r|.
```

At zero the entire allowance is `1e-4`; relative error is undefined and is not an
acceptance metric. At `r=1e-6`, even a large relative error can be hidden by the
absolute allowance. At `r=1`, the allowance is `0.0011`. This is why the report
includes absolute error, error/tolerance ratio, and nonzero-reference relative
error as diagnostics but uses only `allclose` for the verdict.

One PCM LSB has normalized magnitude `2^-15`; its unwindowed squared magnitude is
`2^-30 ~= 9.313e-10` before window/FFT scaling. Small Mel weights can make the
result still smaller. Consequently zero/low-amplitude tests are excellent for
catching uninitialized memory, NaNs, wrong shape, and spurious absolute energy,
but weak at detecting relative FP32 FFT error.

For an FFT coefficient `X` with error `dX`:

```text
P = |X|^2
P+dP = |X+dX|^2
 dP = 2*Re(conj(X)*dX) + |dX|^2.
```

The cross term generally dominates when `X` is large, so power approximately
doubles relative FFT error; near `X=0`, the quadratic term dominates. FP32
rounding in the window, twiddles, FFT, and power therefore cannot be assessed only
from the final output's float32 quantum.

For Mel row `m`:

```text
M_m = sum_k w[m,k]*P[k]
dM_m = sum_k (w[m,k]*dP[k] + dw[m,k]*P[k] + dw[m,k]*dP[k]).
```

Power and valid Mel weights are nonnegative, so Mel summation has no subtractive
cancellation, but sparse support does **not** remove FFT or weight errors. A few
large bins can dominate a row. An FP32 frequency/triangle-boundary discrepancy can
also change an exact zero into a nonzero weight (or vice versa), changing the
support itself. DC and Nyquist are single `rfft` bins with no factor-of-two power
scaling.

## 5. Why FP32 can fail after passing official audio

Official audio is one input distribution and four parameter sets. Passing it does
not prove a general FP32 bound. FP32 may still fail because:

- coherent full-scale DC/Nyquist/tonal frames accumulate FFT error differently
  from ordinary audio; larger transforms have more stages and larger coefficients;
- squaring introduces the cross term above and can amplify a previously acceptable
  coefficient error;
- sparse or narrow Mel rows may be dominated by one erroneous bin rather than
  averaging many independent errors;
- weight generation in FP32 can shift triangle edges/support, especially for tiny
  transforms and closely spaced high-Mel filters;
- different FFT/Mel reduction orders and any reduced-precision tensor mode can
  exceed FP32 assumptions; an official pass must not be used to justify TF32 or a
  relaxed tolerance;
- zero and 1-LSB signals are mostly protected by `atol`, hiding large relative
  errors that a mixed/high-energy signal would expose;
- small-transform special cases (length one/two, DC/Nyquist reconstruction,
  impulse at Hann zero) and exact frame tails are not covered by official audio.

FP64 is a useful diagnostic candidate, not a bit-for-bit cuFFT/V1 guarantee. It
must pass the same tests and tolerances.

## 6. Float32 file output versus host `std::vector<double>` semantics

V1's `compute_melspectrogram` contract returns a host
`std::vector<double>& output` of length `n_mels*F`, row-major. `write_output` then
allocates a full float32 vector, casts each double, and writes those float32 bytes.
A CUDA implementation must preserve that host interface even if device computation
or device staging is FP32:

```text
device f32 values -> host f32 staging -> host double output -> write_output cast f32
```

For finite float32 values, the `float32 -> double -> float32` round trip is exact.
Do not reinterpret a float buffer as doubles, write the double vector directly, or
change the file to float64. A device FP64 path may copy doubles directly into the
host output, followed by the same final cast. The extra host double vector and the
writer's simultaneous float32 vector must be included in host capacity planning.

## 7. GPU and host memory formulas

Let:

- `N`: samples; `F`: frames; `H`: hop; `L=n_fft`; `K=L/2+1`; `M=n_mels`;
- `B`: resident frames per batch; `p=4` for FP32 or `8` for FP64;
- `W_plan`: actual cuFFT workspace reported by the plan API;
- `W_runtime`: context, library, allocator, alignment, and other allocations.

A conservative **out-of-place** batch with separate retained power and separate
native-precision Mel accumulation uses:

| item | bytes |
|---|---:|
| streaming y slice | `p*(L+(B-1)*H)` |
| real FFT input | `p*B*L` |
| complex FFT output | `2*p*B*K` |
| optional retained power | `p*B*K` |
| periodic Hann | `p*L` |
| dense Mel weights | `p*M*K` |
| native-precision Mel accumulation | `p*B*M` |
| final device float32 batch output | `4*B*M` |
| plan/runtime | `W_plan + W_runtime` |

Thus the modeled allocation, excluding unknown `W_runtime`, is

```text
G(B) = W_plan + p*L + p*M*K + p*(L+(B-1)*H)
       + B*(p*(L+3*K+M) + 4*M).
```

The `3*K` term is complex FFT output (`2*K`) plus retained power (`K`). If power is
not retained, remove `p*B*K`; if accumulation writes directly to final float32,
remove `p*B*M`. Do not subtract either unless that is the actual implementation.
If Mel weights remain float32 in an FP64 pipeline, use `4*M*K` for that row.

For in-place R2C cuFFT, real input needs padded layout (typically `2*K` real values
per transform), not merely `L`; naive `B*L` allocation is unsafe. The out-of-place
model above avoids that assumption. Also query plan workspace for the actual batch
and transform instead of estimating it from `L`.

With usable budget `G_budget`, the model's maximum is

```text
B_max = floor((G_budget - W_plan - p*L - p*M*K - p*(L-H))
              / (p*(H+L+3*K+M) + 4*M)).
```

Use `B=min(F,4096,B_max)` initially. The 4096 cap is a conservative starting
choice, **not a performance optimum**. If `B_max<1`, reduce resident state or the
batch/workspace requirements rather than allocating anyway. Query actual free
memory immediately before planning, leave headroom, and check every allocation.
Overlap slices exactly: batch starts are `t0*H`, and the next batch reuses the
`L-H` sample halo; never drop the incomplete final frame tail.

The L40 record is 46068 MiB (about 45 GiB), despite the informal "46 GB" label.
`memory_plan.py` therefore defaults to a 45 GiB total, an 80% (36 GiB) planning
budget, and a 2 GiB **placeholder** plan reserve. Replace that reserve with actual
plan queries. With official `N=24634880`, the model gives:

- default FP32 (`L=8192,M=256`): full-resident model 17.14 GiB, B=4096 model
  2.33 GiB including the placeholder reserve;
- large/narrow FP64 (`L=16384,M=256`): full-resident model 61.45 GiB, B=4096
  model 3.28 GiB including the placeholder reserve.

These are arithmetic estimates, not measured device consumption or benchmark
results. FP64 official large transforms should remain batched by default.

### Host (approximately 15 GiB)

A conservative V1-compatible host peak includes:

```text
2*N                    original int16 PCM
8*N                    V1 host double y
8*M*F                  required host double output
4*M*F                  writer's simultaneous float32 output
p*(L+(B-1)*H)          optional native batch y staging
p*B*M                  optional native batch output staging
+ allocator/I/O/runtime slack
```

For official default this model is about 0.79 GiB for the candidate's owned
arrays, before unrelated processes and reference data. It does not include a full
librosa float64 power/FFT matrix. Do not materialize full power or all FFT outputs
on the host. The official harness compares output and expected files in chunks;
production only needs to produce its output, not load the expected file.

### Capacity calculator examples

```bash
./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/memory_plan.py \
  --n-samples 24634880 --n-fft 8192 --hop 128 --n-mels 256 --precision fp32

./tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/memory_plan.py \
  --n-samples 24634880 --n-fft 16384 --hop 128 --n-mels 256 --precision fp64
```
