# F1 — selective CPU upgrade

Source `exp/cpu-pipeline-simd` e14647f (algorithm cbf214d, underlying SIMD
e448c989). Review completed before editing; no merge/cherry-pick. The ported
hunks are archived in raw/f1/ported-cpu.patch.

- Raw overlap slab transposition before double Hann multiplication / float
  narrowing. B16 only, dimension-based cache budget and high-overlap guard.
  All other ISA and nonoverlap paths retained; no FFT arithmetic changes.
- Linux inward-page-aligned MADV_HUGEPAGE on output reserve, followed by normal
  vector resize. No skipped initialization or allocator metadata access.
- Unconditional empty-batch early return fixes the frozen n_fft=1 invalid
  reference binding (including B8 and FP64 SIMD workers).
- Build defaults: CPU_PACK_REUSE=1, CPU_HUGEPAGE=1. Individually disable for
  ablation. Object paths include both flags and ISA batch.

Validation:

| Platform | Check | Result |
|---|---|---|
| Local AVX2 B8 | 68 independent PCM / arbitrary parameter cases vs frozen B8 and librosa | PASS; frozen output bit-exact |
| Local AVX2 B8 | strict ASan+UBSan, 24 arbitrary-double/reused-vector API cases | PASS |
| Local generic x86-64 FP64 | strict ASan+UBSan, same 24 API cases | PASS |
| hepnode2 AVX-512 B16 Job 419 | 68 independent cases vs frozen B16 and librosa | PASS under original rtol=1e-3, atol=1e-4 |
| hepnode2 AVX-512 B16 Job 419 | strict ASan+UBSan, 24 API cases | PASS |

The slab+THP combination inherits the source's non-bit-exact FP64-tail caveat:
some near-zero residuals can differ at ~1e-28 after codegen/layout changes.
Acceptance remains official per-element rtol/atol, never an altered tolerance.
Final performance is measured in F2/F4, not inferred by adding individual gains.
