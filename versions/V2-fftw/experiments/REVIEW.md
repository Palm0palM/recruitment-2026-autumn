# Orchestrator review / current acceptance status

No production kernel has been changed in this round. A2 ESTIMATE remains the last cluster-validated baseline.

## Precision

Reviewed private implementation: distinct aligned fftwf buffers, shared single-thread plan, double window/Mel,
and two independent power variants. Both built successfully under the parent and passed 39 valid/3 short-input
cases versus freshly rebuilt A2. Both also passed all four full expected comparisons without relaxing tolerance.
The subsequent controlled local 3-repeat screen includes A2 and all batch variants; raw repeats are retained.
These are local tests, NOT complete official cluster validation or a formal speedup claim.

Numerical note correction to the worker's NOTES: exactly at float value 2^26 the forward ULP is 8,
while below it in [2^25,2^26) spacing is 4. The worker's approximate example conflates the boundary;
this does not affect prototype code or test tolerance.

## Batch

Reviewed plan-many layout, disjoint worker ranges and zero-filled tail rows. All B=1/4/8/16/32 built and
passed the worker's frame/batch/worker boundary suite. All 96 full-output local screening invocations passed.
Batch >1 is generally slower on this local screen; do not promote it merely because it is batched.
Scratch formula is theoretical, not RSS. Production integration still requires explicit memory and cluster tests.
Extreme int-limit note: t0 += kBatch could overflow near INT_MAX frames; fix before general-purpose integration,
even though current legal test inputs are far below that range. No such integration has occurred.

## Wisdom

Primary review caught CPUID leaf1 EBX initial APIC ID in the fingerprint. Kimi fixed its private source by
masking bits31:24; without this, same-machine processes on different cores could spuriously reject wisdom.
Correct EBX fields: bits7:0 brand index, bits15:8 CLFLUSH line size, bits23:16 maximum addressable logical IDs.
Worker comments/notes have the middle two descriptions swapped; the mask itself is correct.

Parent archived the prototype and added one small experimental change: MELSPEC_FFTW_WISDOM_DIRECTORY
selects <transform>-<size>.wisdom, supporting all CLI FFT sizes without a Python wrapper distorting timing.
The per-file envelope/CPU/library/precision/layout checks and safe ESTIMATE fallback remain mandatory.
The parent built this archived variant and its offline generator successfully. MEASURE files for
1/2/8/256/2048/8192/16384 were generated using the actual linked library's SHA-256. All 39 valid and
3 short-input cases passed versus fresh A2, with loaded diagnostics for all valid cases. Performance,
negative-envelope tests and cluster validation remain pending; no wisdom speedup is yet claimed.

## ISA

Static research and independent build/probe/audit drafts received; execution not authorized yet.
Enabled/selected ISA, planner decomposition and copy behavior must be separated. No AVX2-only benefit is measured.
Review scripts before any shell execution, and never build an AVX2-only claim with accidental native AVX-512.
Parent adapted the archived ISA builder's output root to this worktree's tmp/fftw-lab/isa/, and the archived
batch test's scratch to TMPDIR: archiving a worker script must not make it emit dependencies/PCM in versions/.

## Local screen (3 repeats, parent-only serialized)

Minimum A2-relative ratios for default/small_mels/large_fft/narrow_band:
- float FFT + double power: approximately 1.017 / 1.129 / 1.178 / 1.185.
- float FFT + float power: approximately 1.071 / 1.150 / 1.180 / 1.183.
- batch 4: approximately 0.937 / 1.030 / 0.896 / 0.889; larger batches generally regress further.
Single-run precision screening is correctness evidence only; use the complete repeats for local comparisons.
No independent speedups may be multiplied to predict a combination.

Wisdom negative tests now pass: valid load, wrong/missing library ID, wrong CPU, wrong size, truncated
envelope and checksum corruption. All fallback outputs match A2 within original tolerance. The local
MEASURE-wisdom 3-repeat screen passes all full expected comparisons but does not show an all-case gain:
approximately 1.002 / 1.078 / 1.062 / 0.972 versus same-screen A2 minima. This is not a cluster result.

Memory measurement failure: the first posix_spawn/wait4 ru_maxrss sequence inherited the large Python
parent's RSS around 1 GB, yielding implausibly identical peaks for distinct children. Those numbers are
explicitly INVALID, not candidate memory evidence. The harness now samples the matching post-exec
/proc/<pid>/status VmHWM; CPU rusage is retained separately. Serial remeasurement completed: total
default peaks approximately 764–767 MiB and large_fft approximately 772–774 MiB. Even large-batch scratch
does not necessarily raise TOTAL peak: the fixed final double-to-float write buffer can dominate peak RSS
after FFT scratch was destroyed. Theoretical scratch and measured whole-process peak are different metrics.
