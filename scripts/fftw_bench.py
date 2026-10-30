#!/usr/bin/env python3
"""Interleaved full-input V1/A1/A2 process timings and correctness.

No librosa timing here; official validate.py is run separately for the winner.
Output goes to node-local TMPDIR if set, otherwise this worktree's tmp/scratch.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--programs", nargs="+", default=["v1=tmp/bin/v1", "a1-estimate=tmp/bin/c2c-estimate",
                    "a1-measure=tmp/bin/c2c-measure", "a2-estimate=tmp/bin/r2c-estimate", "a2-measure=tmp/bin/r2c-measure"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be positive")
    programs = dict(p.split("=", 1) for p in args.programs)
    if len(programs) != len(args.programs):
        ap.error("duplicate program names")
    cases = {c["name"]: c for c in load_official_cases()}
    if any(n not in cases for n in args.cases):
        ap.error("unknown case")
    data = find_data_dir()
    scratch = Path(os.environ.get("TMPDIR", str(ROOT / "tmp/scratch")))
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.cases:
        tc = cases[name]
        expected = np.fromfile(data / Path(tc["expected"]).relative_to("data"), dtype="<f4")
        flags = [s for k, v in tc["params"].items() for s in (f"--{k}", str(v))]
        records = {n: dict(case=name, program=n, times=[], max_abs_err=0.0, passed=True) for n in programs}
        order = list(programs)
        with tempfile.TemporaryDirectory(prefix="fftw-bench-", dir=scratch) as td:
            out = Path(td) / "out.bin"
            for repeat in range(args.runs):
                # Rotate program order across rounds; no simultaneous workloads.
                for n in order[repeat % len(order):] + order[:repeat % len(order)]:
                    cmd = [str(Path(programs[n]).resolve()), str(data / "audio.pcm"), str(out), *flags]
                    t0 = time.perf_counter()
                    subprocess.run(cmd, check=True, capture_output=True)
                    elapsed = time.perf_counter() - t0
                    got = np.fromfile(out, dtype="<f4")
                    passed = got.shape == expected.shape and bool(np.allclose(got, expected, rtol=1e-3, atol=1e-4))
                    if not passed:
                        raise SystemExit(f"FAIL {name} {n} round={repeat}")
                    err = float(np.max(np.abs(got - expected)))
                    records[n]["times"].append(elapsed)
                    records[n]["max_abs_err"] = max(records[n]["max_abs_err"], err)
                    print(f"{name} {n} round={repeat} elapsed={elapsed:.6f} max_abs_err={err:.8g} PASS", flush=True)
        for r in records.values():
            r["min_s"] = min(r["times"])
            r["median_s"] = float(np.median(r["times"]))
            r["vs_v1"] = min(records["v1"]["times"]) / r["min_s"] if "v1" in records else None
            results.append(r)
        del expected
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print("case | program | min_s | median_s | vs_v1", flush=True)
    for r in results:
        print(f"{r['case']} | {r['program']} | {r['min_s']:.6f} | {r['median_s']:.6f} | {r['vs_v1']}")


if __name__ == "__main__":
    main()
