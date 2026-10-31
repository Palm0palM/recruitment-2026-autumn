# F2 — upgraded CPU + frozen CUDA pipeline

## Sources / boundaries

CPU: F1 selective cbf214d port on e448c989 SIMD. CUDA math, kernels, batch512,
cuFFT, async output, resize and context-overlap pipeline: 3e111946.
No CUDA sprint background allocation or GPU input-conversion experiment used.

The only extra CUDA adapter change is typed initialization unavailability:
cudaErrorNoDevice / cudaErrorInsufficientDriver become BackendUnavailable at
the existing context initialization boundary. Auto catches ONLY that type,
before all numerical GPU kernels; forced CUDA reports it. All other errors
propagate. This removes the inherited serial cudaGetDeviceCount pre-probe.
Forced n_fft=1 still checks a device because its frozen mathematical shortcut
otherwise performs no CUDA operation. CPU decisions call no CUDA API.

## Job 425: same-node A/B

hepnode2 / Xeon Gold 6338 / 16 bound cores 0–15 / L40 sm89 / CUDA 13.4.92 /
GCC 15.2 / THP madvise / private node-local output scratch. Process-only timer;
all output readback and allclose outside clock. Every measured output checked.
Six interleaved rounds; seconds, median (all raw/min in raw/f2/).

| Official case | Frozen B16 | Final CPU (unified) | CPU reduction | Frozen CUDA | Final CUDA | GPU overhead |
|---|---:|---:|---:|---:|---:|---:|
| default | .83137 | .66764 | 19.69% | .79900 | .80132 | .29% |
| small_mels | .48468 | .42305 | 12.72% | .53909 | .54502 | 1.10% |
| large_fft | 1.03222 | .85077 | 17.58% | .80198 | .81253 | 1.32% |
| narrow_band | 1.00540 | .80788 | 19.65% | .80413 | .81569 | 1.44% |

All cases and all rounds PASS. Before the adapter (Job 422), serial pre-probe
added ~65–90 ms on default/large; its raw data is retained in raw/f2-preprobe.
Job 419 home/ZFS/unbound results are diagnostics only, never calibration data.

## Model calibration (training only)

44 points: original hybrid 38-point grid, plus n_fft=32768/65536 at
4096/32768/65536 frames. Five interleaved runs; per-output checks.

```
W = frames*n_fft*log2(n_fft)/1e8
I = input_pcm_bytes/1e6
O = frames*n_mels*4/1e6
C = W*max(0, n_fft/16384 - 1)

Tcpu  = .0306936572 + .0005203894*W + .0040032038*I + .0017894085*O + .0012900750*C
Tcuda = .2440087176 + .0000568870*W + .0039520393*I + .0016898423*O + .0000673957*C
```

C is an empirical large-working-set pressure feature, motivated by the B16
scratch footprint and measured slope change at 32K/64K. It is not a claim of
measured cache misses or a changed FFT complexity. The plain 4-feature fit over
all 44 points had CPU MAE .127 s and 40/44 winners; this fit has MAE .0205 s
and 43/44 winners, also 43/44 leave-one-out. One miss: 64K FFT / 4096 frames.
Selection is argmin, no case name, audio content, threshold exceptions or trial.
Generated header include/melspec_model_cuda.h and fitting script are shipped.

Independent holdout and final official validation are recorded separately;
they are not included in fitting. Recalibrate when changing affinity,
hardware, THP policy, input/output storage, compiler or CPU optimization flags.
