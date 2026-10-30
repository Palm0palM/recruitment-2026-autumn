# Standalone cuFFT batched-layout probe

`cufft_layout_probe.cu` is an independent CUDA/cuFFT prototype. It does not read
PCM, use project sources, or depend on Python. It covers:

- FP64 `CUFFT_D2Z` and FP32 `CUFFT_R2C` forward transforms;
- `n_fft = 8192` and `16384`;
- batch sizes `1, 4, 16, 64, 128, 256, 512, 1024`;
- dense natural `[batch][n_fft]` real input and
  `[batch][n_fft/2+1]` complex output;
- separate plan-creation timing, reported plan workspace, untimed warm-up,
  and CUDA-event timing of individual warm executions;
- a deterministic 8-point direct-DFT correctness check, including a separate
  final-partial-batch plan;
- RAII cleanup of plans, device allocations, and events.

No measured results are included here. The main agent must build and run the
probe on the L40 before making any performance claim.

## Exact build and run commands for the main-agent Slurm job

These commands are intended for the existing one-GPU Slurm job/allocation, not
as a new submission. They follow the existing smoke-job convention that the
job is submitted from the remote worktree root and that CUDA 13.4 is private to
that worktree. If the main agent uses another remote location, change only
`SLURM_SUBMIT_DIR`/`CUDA_HOME` setup to the actual private paths.

```bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}/tmp/v3-cuda-lab/cufft"
export CUDA_HOME="${CUDA_HOME:-${SLURM_SUBMIT_DIR}/toolkit/cuda-13.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$CUDA_HOME/bin/nvcc" -O2 -std=c++17 -arch=sm_89 \
  -I"$CUDA_HOME/targets/x86_64-linux/include" \
  cufft_layout_probe.cu -o cufft_layout_probe \
  -L"$CUDA_HOME/lib64" \
  -Xlinker -rpath,"$CUDA_HOME/lib64" \
  -lcufft -lcudart

# Optional fast functional gate. The full command runs these checks again.
./cufft_layout_probe --self-test-only

# Required sweep; pipefail above preserves a failing probe's exit status.
./cufft_layout_probe --warmup 3 --iters 10 2>&1 | \
  tee cufft_layout_probe.log
```

The binary defaults to the same full sweep with three warm-up and ten measured
executions per case. `--warmup N` and `--iters N` require positive integers.
`--self-test-only` skips the large sweep. Run with one visible GPU; the probe
uses the current CUDA device and the default stream.

## Data layout and plan construction

For each precision, size, and batch, the probe calls `cufftPlanMany` with:

| Parameter | Value | Meaning |
|---|---:|---|
| rank | 1 | one-dimensional transform |
| `istride`, `ostride` | 1, 1 | adjacent samples/bins within a signal |
| `idist` | `n_fft` | next real input signal, in elements |
| `odist` | `n_fft/2+1` | next complex output spectrum, in elements |
| `inembed`, `onembed` | `nullptr`, `nullptr` | no padded embedded layout |
| batch | current sweep value | independent transforms per execution |

Thus host and device indexing is exactly:

```text
input[b * n_fft + t]                 0 <= t < n_fft
output[b * (n_fft/2 + 1) + k]        0 <= k <= n_fft/2
```

Distances are element counts, not bytes. FP32 output elements are
`cufftComplex`; FP64 output elements are `cufftDoubleComplex`. Allocations use
`batch*n_fft` inputs and `batch*(n_fft/2+1)` outputs. There is no padding
between signals.

For an eventual input containing `N` signals and a chosen full batch `B`, use
`q = N/B` executions with the batch-`B` plan. If `r = N%B` is nonzero, construct
or retrieve a distinct plan whose batch is exactly `r` for the final `r`
signals. Never execute the batch-`B` plan on storage allocated for only `r`
signals. The self-test explicitly processes five signals as a batch of three
followed by a separately planned batch of two. The timing sweep itself contains
only full batches.

## Correctness check

Before any large timing case, both FP32 and FP64 paths run an 8-point transform
against a CPU direct DFT. Five deterministic, distinct signals are checked over
all retained bins in natural output order, using a full batch of three and a
tail batch of two. Errors are scaled by `max(1, abs(reference_bin))`; thresholds
are `2e-4` for FP32 and `2e-11` for FP64. A mismatch, non-finite result, CUDA
error, or cuFFT error makes the program exit nonzero.

The large `8192/16384` cases are execution/layout probes only; they are not
checked against a large reference DFT. Their input is deterministic synthetic
integer-hash data generated independently of PCM.

## Timing and output

The log starts with `#` metadata and two `self_test ... status=PASS` lines. It
then emits one CSV header and 32 case rows in this order: both precisions for
`n_fft=8192`, then both precisions for `n_fft=16384`, with ascending batches
within each precision.

CSV columns:

```text
precision,n_fft,batch,plan_create_ms,workspace_bytes,
warmup_iterations,measured_iterations,
exec_ms_min,exec_ms_mean,exec_ms_max,
input_bytes,output_bytes,device_buffer_bytes
```

- `plan_create_ms` is host `steady_clock` elapsed time around only
  `cufftPlanMany`, with `cudaDeviceSynchronize` immediately before and after.
  Workspace query, allocation, input generation/copies, warm-up, and execution
  are outside this interval.
- `workspace_bytes` is obtained from `cufftGetSize` after successful plan
  creation. It is cuFFT's reported plan work-area size, not total GPU memory.
- Warm-up executions are enqueued first and completed with a device-wide
  synchronization. None is included in execution statistics.
- Each measured iteration records start/stop CUDA events around one batched
  `cufftExec*`, synchronizes the stop event, and obtains elapsed time with
  `cudaEventElapsedTime`. Min/mean/max are computed from those individual event
  intervals. Input generation and H2D copies are excluded.

## Memory-safety bounds and cleanup

Cases run sequentially. Each case destroys its events, device buffers, host
input, and plan before the next case starts. Element and byte products are
checked for `size_t` overflow. There is no allocation for all batch cases at
once.

At the largest case (`n_fft=16384`, `batch=1024`), input plus output device
buffers are approximately 256 MiB for FP64 and 128 MiB for FP32, excluding the
reported plan workspace and any CUDA/cuFFT internal allocations. FP64 also has
a 128 MiB host input vector at that case. Allocation failure is fatal rather
than silently skipping a case.

## Design risks and interpretation limits

1. This is a standalone forward-FFT layout/timing prototype, not a production
   integration and not a mel-spectrogram or PCM correctness test.
2. Synthetic data, repeated execution of unchanged input, default-stream
   serialization, and per-case plan creation/destruction may differ from a
   production pipeline's plan cache, streams, overlap, and data reuse.
3. Plan timing uses device-wide synchronization and assumes an otherwise
   exclusive GPU. Concurrent unrelated work would contaminate both host and
   event observations.
4. `cufftGetSize` does not account for input/output buffers, host memory, CUDA
   context overhead, or every possible library-internal allocation.
5. The small direct-DFT test checks layout and numerical sanity for both APIs,
   but does not establish numerical accuracy at the two large transform sizes.
6. The probe measures cuFFT execution intervals only; it does not include
   production framing, windowing, mel filtering, data transfer, or end-to-end
   throughput.
