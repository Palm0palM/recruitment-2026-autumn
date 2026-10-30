# Local functional test evidence

Date: 2026-10-30. Working directory for every command:
`/home/palm/projects/heptagon/melspec-v3-cuda`.

These are local CPU correctness tests of the harness, using the repository's
`melspec.py` as a known non-CUDA candidate and `melspec` as the fixed V1 reference.
No CUDA candidate, official full-audio run, formal benchmark, SSH, Slurm, or
performance measurement was performed by this subagent.

## CPU unit tests

Final command:

```bash
tmp/v3-cuda-lab/precision/python.sh tmp/v3-cuda-lab/precision/test_local.py
```

Result: exit 0, **9/9 tests passed**. Covered strict near-zero comparison, NaN
rejection, exact f32->double->f32 round trip, chunked file comparison, official
frame counts, the length-one V1/SciPy window distinction, and memory
formula/frame/batch invariants.

An earlier 8-test version (before the explicit chunked-file regression test) also
passed before end-to-end testing.

## Reduced end-to-end smoke test

```bash
tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py melspec.py --quick
```

Result: exit 0, **30/30 cases passed**. Artifact:
`runs/20261030-160029-182976/summary.json`.
Librosa emitted its expected `Empty filters detected in mel frequency basis`
warning for small transforms. No timing was collected.

## Full bounded synthetic suite

```bash
tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py melspec.py
```

Result: exit 0, **74/74 cases passed**. Artifact:
`runs/20261030-161259-195976/summary.json`.

A read-only aggregation of that JSON used:

```bash
tmp/v3-cuda-lab/precision/python.sh -c 'import json; p="tmp/v3-cuda-lab/precision/runs/20261030-161259-195976/summary.json"; d=json.load(open(p)); r=d["results"]; print({"cases":len(r),"passed":d["passed"],"candidate_librosa_failures":sum(not x["candidate_vs_librosa"]["passed"] for x in r),"candidate_v1_failures":sum(not x["candidate_vs_v1"]["passed"] for x in r),"oracle_differences":sum(not x["v1_vs_librosa"]["passed"] for x in r),"max_candidate_librosa_abs":max(x["candidate_vs_librosa"]["max_abs_error"] for x in r),"max_candidate_v1_abs":max(x["candidate_vs_v1"]["max_abs_error"] for x in r)})'
```

Observed:

```text
{'cases': 74, 'passed': True, 'candidate_librosa_failures': 0,
 'candidate_v1_failures': 0, 'oracle_differences': 0,
 'max_candidate_librosa_abs': 0.000244140625,
 'max_candidate_v1_abs': 0.03125}
```

Maximum absolute error alone exceeding `atol` is not a failure: the unchanged
criterion also includes relative tolerance and passed every element. The
length-one window distinction was not observed as an output oracle difference in
this matrix because small/empty Mel responses can map both results to zero.

After strengthening requested-V1 execution/load failures to make the harness fail,
two representative cases were rerun on the final harness:

```bash
tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py melspec.py \
  --case 'core/nfft8/impulses|official-params/large_fft/full_scale_negative'
```

Result: exit 0, **2/2 passed**. Artifact:
`runs/20261030-162556-206640/summary.json`.

## Capacity arithmetic (not device measurements)

```bash
tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/memory_plan.py \
  --n-samples 24634880 --n-fft 8192 --hop 128 --n-mels 256 --precision fp32
```

Result: exit 0. Model: `F=192397`, `K=4097`, per-frame 84,492 bytes,
full-resident 17.1436 GiB, recommended B=4096 using 2.3263 GiB including the
2 GiB placeholder plan reserve. Host-owned-array model 0.7858 GiB.

```bash
tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/memory_plan.py \
  --n-samples 24634880 --n-fft 16384 --hop 128 --n-mels 256 --precision fp64
```

Result: exit 0. Model: `F=192333`, `K=8193`, per-frame 331,800 bytes,
full-resident 61.4492 GiB, `B_max=109976`, recommended B=4096 using 3.2816 GiB
including the placeholder reserve. Host-owned-array model 0.7915 GiB.

These numbers are formulas with conservative separate buffers, not CUDA allocation
measurements. Actual cuFFT workspace and free memory still must be queried.

## Not run here

The main agent must run the FP32/FP64 CUDA candidates through the full synthetic
and official-audio commands in `README.md`, then the unchanged `validate.py`.
No claim is made that this Python-candidate harness validation establishes CUDA
correctness.
