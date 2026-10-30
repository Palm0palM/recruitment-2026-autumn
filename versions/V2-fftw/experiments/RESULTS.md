# Kimi worker round — results in progress

Production remains A2 double ESTIMATE until a separately validated integration.
Worker model/call evidence: `ORCHESTRATION.md`, `agent-verification.json`, `session-verification.json`.
Initial prototypes/local review: `REVIEW.md`. Do not confuse local screens with cluster conclusions.

## Cluster job 93: independent precision and wisdom

- Commit `f9d2080af32b63c9147e21c2a7cce163ea7ed096`; hepnode2, GCC 15.2.0, cpuset 0–15,
  1 task, 16 CPUs, nomultithread, no memory request; COMPLETED, exit 0, 10:02.
- Uninstrumented `-O3 -march=native` builds; serial rotated-order three-repeat comparison.
- All **48** full-output expected comparisons PASS. Each candidate also passed 39 signal/boundary tests
  and 3 short-input rejection checks. Float never changes the original rtol=1e-3/atol=1e-4.
- FFTW double static SHA-256 `c487eddeb264013f9f051af9fb15ce66d60fc6d15d32ff7eb31501f593554ef5`.
- FFTW float static SHA-256 `15c7fb154e1a65ed84858cf1776ce8d8a18ec46ddc55f556208f12e098fd761a`.
- A2 executable SHA-256 `f1ab166c877e3921ed1a1d330a69ca542e243cd69eaef9184443662e7eb88e64`, identical
  to the first-round validated A2 binary. Full candidate hashes and build commands are in raw cluster logs.

Same-job **median whole-process seconds**, with A2-relative speedups in parentheses:

| Candidate | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| A2 double ESTIMATE | 1.011655 | 0.660833 | 1.363302 | 1.320362 |
| Float FFT + double power | 0.906400 (1.116×) | 0.562626 (1.175×) | 1.213276 (1.124×) | 1.149965 (1.148×) |
| Float FFT + float power | 0.893212 (1.133×) | 0.574579 (1.150×) | 1.193986 (1.142×) | 1.135100 (1.163×) |
| Double MEASURE wisdom | 0.924865 (1.094×) | 0.584478 (1.131×) | 1.243168 (1.097×) | 1.187405 (1.112×) |

Float-power won this job's geometric-mean median ranking; official `validate.py --runs 3` then returned
**4/4 PASS**, minimum seconds **0.907 / 0.573 / 1.212 / 1.155**. These official minima are not the
denominator of the same-job speedups above. The validator's printed speedup is relative to its own reference,
not A2. Float-power binary SHA-256 `2e12fd2f7278613d0151701032cbaf66d3bf4324976016f53e3fad87f1b06ba7`.

Post-exec sampled process VmHWM (KiB), separate non-score invocations:

| Candidate | default | large_fft |
|---|---:|---:|
| A2 | 788288 | 794400 |
| Float-power | 787660 | 796616 |
| Double-power | 787492 | 796704 |
| Double wisdom | 788308 | 794376 |

End-of-process output conversion dominates total peak, so reduced FFT scratch does not imply reduced total
peak RSS. VmHWM sampling is every 5 ms and may miss a very short final peak. Inherited wait4 ru_maxrss is
retained solely as diagnostic evidence and is NOT used as a valid memory metric.

## Additional numerical falsification search (local, parent-only)

Both float candidates passed **54/54** stronger cases spanning N=256 through 65536: exact-period full-scale
signals, bin/off-bin/near-Nyquist tones, quiet bands, DC+1LSB, strong+quiet multitone and sparse impulses.
Worst normalized error abs/(atol+rtol*abs(ref)) was about **0.02255** (failure threshold 1), in a quiet band
of N=65536 strong+quiet multitone. This is evidence from tested inputs, not a proof for all possible PCM.

## Remaining gates

- Controlled AVX2-only versus AVX-512-enabled libraries and one original batch control on cluster.
- Revised padded/interleaved batch layout prototypes and float+wisdom combination, each tested independently.
- Production integration only after review and another full official validation/benchmark of that exact build.
- No push/merge performed in this round. Regressions and invalid measurement attempts are retained.
