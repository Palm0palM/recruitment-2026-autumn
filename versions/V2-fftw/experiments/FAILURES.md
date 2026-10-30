# Failed attempts (retained, never scored)

- Early resolved-model diagnostic exposed a credential field; local file sanitized, no credentials committed,
  rotation recommended. Further metadata collection uses a whitelist.
- Initial wait4 ru_maxrss inherited the large Python parent's RSS; INVALID raw numbers retained separately.
  Replaced the reported metric with sampled matching-child post-exec VmHWM.
- Local two-repeat layout/combo screen: all output checks passed, but highly variable timing and roughly
  doubled A2 wall times make it unsuitable for quantitative speedup claims. Raw screen retained.
- Wisdom CPU fingerprint originally included per-logical-CPU APIC ID. Primary review caught it before
  measurement; Kimi masked leaf1 EBX[31:24], and the parent tested corrected load/fallback behavior.
- Cluster 104: AVX2 library built; wrong prefix path in parent script caused compile failure before scoring.
- Cluster 119: explicit global `-mno-avx512f` overrode FFTW's earlier per-codelet `-mavx512f` option.
  AVX-512 codelet compilation correctly refused the unsupported target. FAILED 2:0, 1:56, no scoring.
  Correction: identical scalar `-O3 -march=x86-64-v3` for both libraries, WITHOUT an explicit disabling
  flag that defeats codelet-specific overrides. Local preprocessing confirms `-mavx512f -march=x86-64-v3`
  defines __AVX512F__, while plain -march=x86-64-v3 defines __AVX2__ but NOT __AVX512F__.
  Retry uses a fresh tmp/fftw-lab/isa/v3/ directory, preserving the failed build and both logs. Disassembly
  triage plus actual selected plans remain required; macro checks alone are not an ISA-absence proof.
- Cluster 151: all 180 release comparisons, numerical/envelope gates and the first four complete official
  validators passed. The 30-minute allocation expired during the fifth official run because official
  validation also measures the slow reference. This is a scheduling timeout, NOT a candidate correctness
  failure; partial fifth-run output is not counted as PASS. Preserve the exact binaries/wisdom and release
  score JSON, then run the last two official validators, memory measurements and diagnostics in a separate
  serial allocation. Do not regenerate wisdom or cherry-pick new planning outcomes during completion.
- Pending completion job 203 was cancelled at the user's stop request before receiving any node (0 allocated
  CPUs, elapsed 0). Following renewed authorization, job 209 used --runs 1 for each of the last two complete
  official validators, retained existing five-repeat release scores, omitted profiling, and finished all
  validation/memory work in 3:35. This closes the timeout without changing the measured candidate.
