# Kimi worker round — completed

Default production build remains A2 double ESTIMATE. Opt-in float and float wisdom are implemented;
the final integrated release comparison is recorded below, separately from prototype tests.
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

## Closeout complete

All independent experiments, production ports, final release scoring and official validation are complete.
After job 151 timed out, the user cancelled pending job 203 and authorized a short closeout. Job **209**
completed **0:0 in 3:35** on hepnode2: both PATIENT configurations passed official **4/4**, `--runs 1`,
unchanged tolerance and all four cases; all 18 separate memory-check invocations also passed. Binaries and
wisdom were preserved, with no rebuild/replanning/full-matrix rerun/profile. Performance conclusions use
the complete five-repeat job 151 data, NOT single-run official times. No push/merge is performed.

## Integration stage 1 — opt-in precision

Parent adapted the reviewed precision prototype into the existing shared buffer/plan abstraction:

```sh
make -B FFT_PRECISION=double FFT_MODE=r2c EXEC=tmp/bin/a2
make -B FFT_PRECISION=float FFT_POWER=float FFTW_PREFIX=/path/to/float-prefix EXEC=tmp/bin/float-fp
make -B FFT_PRECISION=float FFT_POWER=double FFTW_PREFIX=/path/to/float-prefix EXEC=tmp/bin/float-dp
```

Only FFT input/output and optional power use float. Hann, Mel weights, Mel accumulation and API output
remain double. Float is R2C-only; A1 C2C double remains buildable. Default precision is explicitly double,
and object paths include precision/power to avoid cross-variant object reuse. Use `-B` when changing libraries.
This is an approximate numeric option, NOT preservation of the all-double arithmetic contract.

Parent built all four routes (A1/A2 double, float-power/double-power). Each passed 39 valid and 3 rejection
tests; A2 plus both float routes passed the additional 54-case numerical suite. The integrated local default
A2 executable is **byte-identical** to the pre-round local baseline:
`97594fa7f89161aae8e41bb086dee0736ec2839cf83f59556947188c0f57720c`.
Both exact integrated ordinary-float routes later passed full official validation in job 151.

## Failed ISA scheduling stage: job 104

The AVX2 library built and installed successfully, but the parent Slurm script then used an incorrect
`work/install-*` path instead of the builder's actual `prefix/*`. Compilation failed before any benchmark
(`fftw3.h` not found); job FAILED, exit 1, 3:51. This is an orchestration failure, not an ISA performance result.
The corrected script reuses only a library with recorded matching archive and scalar compiler flags.
Review initially tried explicit `-mno-avx512f` for both scalar builds, but job 119 exposed that this overrides
FFTW's codelet flags. The successful job 126 instead uses identical `-O3 -march=x86-64-v3` for both;
only enabled AVX-512 codelet units receive their required extra ISA flags. See `FAILURES.md`. The earlier drafted
AVX-512 variant's global `-mavx512f` was never built in job 104 and must not be used for a codelet-only claim.

New private prototypes received: padded-row/interleaved batch layouts (B=4/8/16) and float MEASURE/PATIENT
wisdom with precision-specific envelope checks. Parent reviewed and compiled all six layout variants and
both combo power variants; each passed 39 valid and 3 rejection cases. This was the initial acceptance
stage; final cluster comparisons below supersede its then-pending performance status.

All 88 full-output checks in the two-repeat local layout/combo screen passed. Its timing data is retained
but **not accepted as speedup evidence**: process times varied strongly (including roughly doubled A2 times
versus earlier local screens), and near-identical float-power variants exchanged implausibly large gaps.
Only controlled cluster results will decide these variants. Both double and float wisdom also passed the
parent's expanded 19-case envelope suite, including valid import with no applicable plan and malformed
FFTW payloads whose outer checksums were deliberately made valid.

## Integration stage 2 — opt-in float wisdom

The parent ported the reviewed float envelope loader/generator helper into `src/fftw_float_wisdom.cpp`
and `include/fftw_float_wisdom.h`. It is linked ONLY with `FFTW_WISDOM=1`; default double and ordinary
float do not gain this dependency. This build option requires float + ESTIMATE and forbids cold MEASURE.
Offline generation is a separate target using `MELSPEC_FFTWF_WISDOM_GENERATOR=1`, never an app code path.

```sh
make -B FFT_PRECISION=float FFT_POWER=float FFTW_WISDOM=1 FFTW_PREFIX=/path/to/float-prefix EXEC=tmp/bin/combo
make FFT_PRECISION=float FFTW_PREFIX=/path/to/float-prefix wisdom-generator
```

At runtime, set `MELSPEC_FFTWF_LIBRARY_ID` to the actual linked archive hash and
`MELSPEC_FFTWF_WISDOM_DIRECTORY` to a directory containing `r2c-<n_fft>.wisdom`. MEASURE and PATIENT
files are separate experimental configurations, not automatically mixed. Missing/mismatched files safely
fall back to ESTIMATE with explicit diagnostics; no PCM-specific lookup or runtime measurement occurs.
Both integrated power routes passed the 39+3 suite; the float-power integrated loader additionally passed
19 negative-envelope tests locally.
The default A2 build is byte-identical locally and on hepnode2. Exact integrated cluster testing in
`scripts/fftw-lab-final.slurm` completed numerical/envelope gates and all scored invocations; the last
two official validators and memory stage were completed by `scripts/fftw-lab-complete.slurm` in job 209.

## Cluster job 126 — controlled ISA and original batch

Commit `05733e6`, hepnode2, same allocation/compiler as job 93. COMPLETED 0:0 in 9:06.
All **60** interleaved full-output checks PASS, plus 39 valid/3 rejection cases for each new candidate.
Both controlled library builds use scalar `-O3 -march=x86-64-v3`; application flags remain identical native.

| Candidate (median seconds) | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| Native A2 | 1.002906 | 0.666656 | 1.373017 | 1.318609 |
| Controlled AVX2-only A2 | 1.020935 | 0.685612 | 1.442778 | 1.389558 |
| Controlled AVX-512-enabled A2 | 1.000335 | 0.659443 | 1.376469 | 1.318518 |
| Original dense-row batch 4 | 1.044420 | 0.702003 | 1.534384 | 1.479806 |
| Float FFT/float-power control | 0.893730 | 0.558928 | 1.201041 | 1.142936 |

Controlled AVX-512-enabled versus AVX2-only observed gains are **1.021 / 1.040 / 1.048 / 1.054×**;
no improvement over the existing native library is established. Retain existing AVX-512-enabled library;
do not change compiler ISA flags merely on an assumption about downclocking. Actual ESTIMATE plan includes
AVX-512 codelets. Disassembly triage found 46565 suspect lines in the enabled archive; the AVX2 archive's
sole matching line is the filename `avx512.o`, NOT an AVX-512 instruction. Text triage is not a complete proof.

Original batch 4 regresses versus same-job native A2: ratios **0.960 / 0.950 / 0.895 / 0.891×**. It is rejected
for integration, not called an algorithmic correctness failure. Revised layouts are covered by job 151 below.
Default/large_fft VmHWM KiB: A2 **788300/794436**, AVX2 **787640/793528**, AVX-512 **788436/794396**,
batch4 **786228/794204**. Again whole-process output conversion dominates peak, not just FFT scratch.

## Final integrated release comparison — job 151

Production-code commit **`d8234a51f37dbe89a222f8101f73deae1f63f4c0`**. Same hepnode2, GCC 15.2.0,
cpuset 0–15, one task/16 CPUs/nomultithread, no --mem. No phase instrumentation, no internal FFTW threads.
All **180** full-output comparisons PASS (9 configurations × 4 cases × 5 rotated-order repeats).
All 80 wisdom-labelled score invocations explicitly confirmed `status=loaded` without fallback.
Ordinary float, revised layouts and both wisdom rigors passed their 39 valid/3 rejection suites.
All original-tolerance 54-case precision stress comparisons passed, including loaded wisdom through N=65536.
Both rigors passed the 19-case envelope/fallback suite. No relaxed tolerances or PCM-dependent tuning.

Median whole-process seconds; parentheses are speedups against the SAME job's A2 median:

| Configuration | default | small_mels | large_fft | narrow_band | Geomean speedup |
|---|---:|---:|---:|---:|---:|
| A2 double ESTIMATE | 1.004733 | 0.667022 | 1.369675 | 1.311581 | 1.000× |
| Float FFT / float power | 0.894187 (1.124×) | 0.559991 (1.191×) | 1.195897 (1.145×) | 1.139348 (1.151×) | 1.153× |
| Float FFT / double power | 0.898369 (1.118×) | 0.564308 (1.182×) | 1.204793 (1.137×) | 1.155094 (1.135×) | 1.143× |
| Float power + MEASURE wisdom | 0.822264 (1.222×) | 0.491602 (1.357×) | 1.079144 (1.269×) | 1.029757 (1.274×) | 1.280× |
| Double power + MEASURE wisdom | 0.835386 (1.203×) | 0.498174 (1.339×) | 1.093682 (1.252×) | 1.034326 (1.268×) | 1.265× |
| **Float power + PATIENT wisdom** | **0.826122 (1.216×)** | **0.491271 (1.358×)** | **1.023727 (1.338×)** | **0.976393 (1.343×)** | **1.313×** |
| Double power + PATIENT wisdom | 0.832120 (1.207×) | 0.497697 (1.340×) | 1.035943 (1.322×) | 0.982572 (1.335×) | 1.300× |
| Padded-row batch 4 | 1.009260 (0.996×) | 0.665665 (1.002×) | 1.458450 (0.939×) | 1.401968 (0.936×) | 0.968× |
| Interleaved batch 4 | 1.216198 (0.826×) | 0.876576 (0.761×) | 2.171612 (0.631×) | 2.099928 (0.625×) | 0.705× |

Select float FFT + float power + PATIENT wisdom as the fastest observed balanced tested configuration;
do not claim a global optimum, significance for sub-percent differences, or multiplied independent gains.
Both revised batch layouts are rejected for integration. Retain existing AVX-512-enabled library.

The best combination's geomean gain over the same-job no-wisdom float route is about **1.139×**,
measured directly; its A2-relative 1.313× is not obtained by multiplying results from separate jobs.
The no-wisdom float route remains a useful deployment option when target-matching wisdom cannot be prepared.
Default double stays unchanged because float is an explicit approximate-arithmetic option.

Candidate identity:
- Combined float-power executable SHA-256: `a9d994473813cdc44b5588a8510007c5b2ee73072ba235221b880440c030030c`.
- No-wisdom float-power SHA-256: `63b254228d78af57f36a3bcad2e004793174a873111813907944c93cc971d4b6`.
- A2 and both FFTW library hashes remain those listed for job 93.
- Exact measured text wisdom is retained under `raw/cluster/final-wisdom/{measure,patient}/`, with
  SHA-256 manifest; deployment paths/overrides are in `raw/cluster/final-environments-151.json`.
  The same executable serves both rigors; the wisdom bytes/environment are part of candidate identity.

Job 151 reached its 30-minute limit AFTER all scoring and four complete official validators, during the
fifth validator (which also runs the slow reference). Its TIMEOUT does not invalidate the completed score
JSON. Official ordinary-float and both MEASURE routes are each 4/4 PASS with --runs 3. The interrupted
fifth invocation is NOT counted as PASS: job 209 subsequently completed BOTH PATIENT official validators
with --runs 1. No new profile was run during user-bounded closeout.

## Final official acceptance and memory — job 209

Same hepnode2 and resource specification, checkout `a44f6eb`; source/include/Makefile were verified
identical to production commit d8234a5, and exact saved executables/wisdom were reused. Job COMPLETED 0:0,
**3 minutes 35 seconds**. No pending benchmark work remains.

Official `validate.py --runs 1`, all four cases and unchanged original tolerance:

| Configuration | default | small_mels | large_fft | narrow_band | Result |
|---|---:|---:|---:|---:|---|
| **Float power + PATIENT wisdom** | **0.828 s** | **0.492 s** | **1.020 s** | **0.976 s** | **4/4 PASS** |
| Double power + PATIENT wisdom | 0.830 s | 0.490 s | 1.026 s | 0.971 s | 4/4 PASS |

These single-run official times establish acceptance, not the ranking or A2-relative ratios; use job 151's
five-repeat same-job medians above. All six integrated float precision/planner configurations now have
complete official four-case PASS results. Peak observed process VmHWM (KiB), separate non-score runs:

| Configuration | default | large_fft |
|---|---:|---:|
| A2 | 788060 | 794244 |
| Float FFT / float power | 787560 | 796632 |
| Float FFT / double power | 787600 | 796308 |
| Float power + MEASURE wisdom | 787640 | 796604 |
| Double power + MEASURE wisdom | 787596 | 796612 |
| **Float power + PATIENT wisdom** | **787508** | **796680** |
| Double power + PATIENT wisdom | 787544 | 796688 |
| Padded-row batch 4 | 786292 | 794336 |
| Interleaved batch 4 | 786136 | 794508 |

The selected candidate is approximately **769.1 / 778.0 MiB** for default/large_fft. There is no meaningful
whole-process memory reduction: the final output conversion dominates, and large_fft is slightly higher
than A2 despite smaller explicit FFT scratch. Sampled VmHWM is approximate; inherited wait4 RSS is invalid
for this purpose and is intentionally not the reported metric. Memory checks all compared full output again.

## Delivery decision

- Best observed tested balanced configuration: native R2C, **float FFT + float power + PATIENT wisdom**,
  double Hann/Mel/accumulation/output API, at most 16 outer workers, no internal FFTW threads.
- Production implementation commit: **d8234a5**; delivery/report commits leave those production files unchanged.
  Source SHA-256 manifest: `production-source-sha256.json`. Frozen target-matching wisdom and its manifest
  are committed as text experiment records, not binaries or audio data.
- Build: `make -B FFT_PRECISION=float FFT_POWER=float FFTW_WISDOM=1 FFTW_PREFIX=<float-prefix> EXEC=tmp/bin/fftw-best`.
  Set `MELSPEC_FFTWF_LIBRARY_ID` to the actual linked archive hash and `MELSPEC_FFTWF_WISDOM_DIRECTORY`
  to the matching PATIENT directory. Rebuild/regenerate on another target; never force a mismatched fingerprint.
- Default remains all-double A2; without prepared wisdom use the explicit no-wisdom float option (~1.153× here).
- Batch and AVX2-only variants are rejected; failed builds, timeout and invalid local/memory measurements
  remain documented. No additional experiments, official reruns or profiling are scheduled.
