# V3 CUDA GPU-pipeline prototype (Subagent B)

This directory is self-contained. It does not modify or include production
sources. The prototype implements this device path:

```text
host normalized PCM y
  -- one H2D copy of the complete y --> device y
  --> gather overlapping frames + periodic Hann (batch)
  --> cuFFT R2C/D2Z, compact [batch][n_fft] real input
  --> either (power kernel, then sparse-range Mel kernel)
         or  (fused power+Mel kernel)
  --> row-major [n_mels,n_frames] output
  -- D2H --> optional host double-to-float conversion
```

Files:

- `gpu_pipeline.cuh`: actual `float`/`double` CUDA kernels and launch helpers.
- `correctness.cu`: standalone cuFFT host driver and direct-DFT CPU correctness
  harness. It tests both algorithms in both precisions, hop 128, a final partial
  batch, row-major output, and host float conversion. It contains no timing code.

## Numerical and layout contract

- The host supplies normalized samples `y = int16_pcm / 32768`, converted once to
  the chosen compute type. No CPU matrix of overlapping frames is constructed.
- `gather_window_kernel` reads sample `y[(frame_begin+t)*hop+n]` and multiplies
  by the precomputed periodic Hann value
  `0.5 - 0.5*cos(2*pi*n/n_fft)`. The denominator is `n_fft`, not `n_fft-1`.
- cuFFT input is contiguous real `[batch][n_fft]`; R2C/D2Z output is contiguous
  complex `[batch][K]`, `K=n_fft/2+1`. There is no FFT scaling in the kernels,
  matching the reference's unnormalized `|X|^2`.
- `mel_weights` is row-major `[n_mels][K]`. `first[m]` inclusive and `last[m]`
  exclusive delimit nonzero support; exact-zero endpoints are omitted. Empty
  ranges produce zero. The projection remains sparse-range even though weights
  use dense row storage.
- One warp computes one frame/Mel dot product. Lanes read consecutive support
  bins and use a warp reduction. Output index is
  `m*n_frames + frame_begin + local_frame`, so device output is directly
  row-major `[n_mels,n_frames]`, independent of batching.
- All flat indexing and launch work counts use `size_t`; capped grid-stride loops
  avoid 32-bit flat-index overflow on long inputs.
- If compute type is `double`, D2H the double matrix and cast each element to
  `float` on the host immediately before the existing binary write. If compute
  type is `float`, D2H can fill the output buffer directly. No device-side
  downcast is required.

## Separate power versus fused power+Mel

Let, for one batch:

- `B`: actual batch size;
- `K`: `n_fft/2+1`;
- `M`: `n_mels`;
- `L`: `sum_m (last[m]-first[m])`, the number of Mel taps per frame;
- `S`: `sizeof(T)`;
- `C`: `2*S`, bytes in one cuFFT complex value.

The counts below describe explicit post-FFT kernel accesses/arithmetic, not
cuFFT internals, caches, gather, Hann, or H2D/D2H transfers.

| Quantity | Separate power + Mel | Fused power+Mel |
|---|---:|---:|
| Complex FFT values read | `B*K` | `B*L` |
| Squared norms evaluated | `B*K` | `B*L` |
| Power values written | `B*K` | `0` |
| Power values read by Mel | `B*L` | `0` |
| Weight values read | `B*L` | `B*L` |
| Output values written | `B*M` | `B*M` |

Ignoring cache effects, the corresponding post-FFT byte counts (including the
common weight and output traffic) are:

```text
separate = B*K*(C+S) + 2*B*L*S + B*M*S
fused    = B*L*(C+S)           + B*M*S
```

The fused kernel removes the power allocation, its `B*K` stores, and its `B*L`
loads. In exchange, each shared FFT bin is read as a complex value and its norm
is recomputed for every overlapping Mel filter that uses it. If `L>K`, fused
performs `B*(L-K)` more norm evaluations than separate; if narrow bands leave
bins unused, `L` can instead be smaller than `K`. Separate also materializes all
bins, including bins outside every filter. Both perform the same `B*L` Mel
multiply/add work in this prototype. These are work/tradeoff formulas only;
cache behavior and measured performance are left to the main agent.

## Batch and memory handling

Use `capacity=min(requested_capacity,n_frames)` and iterate:

```cpp
for (int frame_begin = 0; frame_begin < n_frames; frame_begin += capacity) {
    int B = std::min(capacity, n_frames - frame_begin);
    gather_window_kernel<<<..., stream>>>(..., B, frame_begin, ...);
    cufft_exec(plan_for_exact_B, ...);  // R2C or D2Z
    launch_separate_or_fused(..., B, frame_begin, ...);
}
```

A cuFFT plan has a fixed batch. Create/cache a full-capacity plan and, when
`n_frames % capacity != 0`, another plan whose batch is exactly the tail size.
Never run the tail with the full plan: that would access uninitialized or
out-of-bounds batch slots. All custom kernels receive the actual `B`, so they do
not process padded frames. The harness uses 13 frames and capacity 5, producing
batches 5, 5, and 3.

For this full-device-output implementation, define `W_plans` as cuFFT's actual
plan/work-area allocations (full plus tail where applicable). Ignoring allocator
alignment, device storage is:

```text
common = S * (N + n_fft + M*K + M*F + capacity*(n_fft + 2*K))
         + 2*M*sizeof(int) + W_plans
separate = common + S*capacity*K
fused    = common
```

Here `N=n_samples` and `F=n_frames`. The terms are respectively complete `y`,
window, dense Mel weights, full output, batch real FFT input, batch complex FFT
output, and two support arrays. cuFFT work area is implementation-dependent;
query/report it rather than guessing. Host input/output and host conversion
buffers are additional. Batching controls FFT/power storage, but not this
implementation's full `M*F` device output. Multiple streams need separate batch
buffers and plans per stream; the static `y`, window, and filterbank are shared.

## Stream and asynchronous options

The harness uses a non-blocking stream and `cudaMemcpyAsync`, with a final stream
synchronization before host output is consumed or device storage is released.
For genuinely asynchronous host copies in an integrated implementation, use
pinned host staging/output buffers; pageable host memory may be staged or make a
copy effectively synchronous.

A single stream is the simplest correct option: upload static arrays once, then
run each batch in order and D2H after all batches. To overlap independent batch
compute/copy work:

1. Keep one read-only device `y`, window, weights, and support arrays.
2. Record an event after their uploads and make each worker stream wait on it.
3. Give every worker stream its own FFT input/output, optional power buffer,
   full/tail cuFFT plans, events, and pinned output staging. Do not concurrently
   execute one cuFFT plan on multiple streams; bind each plan with
   `cufftSetStream`.
4. Dispatch disjoint `frame_begin` ranges. All output writes are disjoint.
5. Synchronize events before reusing per-stream buffers, and synchronize all
   streams before host conversion/write or cleanup.

The current kernels write into the shared full output matrix. Per-batch D2H of
that row-major matrix requires a strided/2D copy (one batch-column segment per
Mel row) or a separate transposition/staging design; a single contiguous copy of
a batch is not layout-correct. The harness intentionally copies the full matrix
once at the end.

## Exact main-agent standalone build/test commands

No toolkit installation or environment modification is needed by this prototype.
Use an already authorized CUDA toolkit. From any worktree checkout, all generated
artifacts remain in this directory's `build/`.

For a local GPU with CUDA 13.4 or another toolkit supporting `-arch=native`:

```bash
cd /home/palm/projects/heptagon/melspec-v3-cuda/tmp/v3-cuda-lab/pipeline
mkdir -p build
nvcc -std=c++17 -O2 -arch=native -I. correctness.cu -o build/correctness -lcufft
./build/correctness
```

For the documented L40 (`sm_89`) and the main agent's authorized private CUDA
13.4 toolkit, run these commands inside the already established GPU execution
environment (replace only `CUDA_HOME` with that toolkit's absolute path):

```bash
cd /home/palm/projects/heptagon/melspec-v3-cuda/tmp/v3-cuda-lab/pipeline
mkdir -p build
CUDA_HOME=/absolute/path/to/authorized/private/cuda-13.4
"$CUDA_HOME/bin/nvcc" -std=c++17 -O2 -arch=sm_89 -I. correctness.cu \
  -o build/correctness-sm89 -L"$CUDA_HOME/lib64" \
  -Wl,-rpath,"$CUDA_HOME/lib64" -lcufft
./build/correctness-sm89
```

Success requires every precision/mode check and the final
`GPU pipeline correctness: PASS`, with process exit status 0. A direct-DFT CPU
reference is computed independently of cuFFT and both GPU post-processing paths.
This is a functional test, not a benchmark. Production integration, official PCM
validation, and all performance measurements remain with the main agent.
