#!/usr/bin/env python3
"""Interleaved full-input CPU/GPU process timings with official-output checks.

Example: python scripts/cuda_bench.py --programs a2=baseline-a2/tmp/bin/a2-estimate \
         cuda=tmp/bin/cuda-fp32 --runs 3 --out tmp/results/compare.json
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
    ap.add_argument("--programs", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be positive")
    programs = dict(item.split("=", 1) for item in args.programs)
    if len(programs) != len(args.programs):
        ap.error("duplicate program names")
    cases = {c["name"]: c for c in load_official_cases()}
    if any(name not in cases for name in args.cases):
        ap.error("unknown case")
    data = find_data_dir()
    scratch = Path(os.environ.get("TMPDIR", str(ROOT / "tmp/scratch")))
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.cases:
        case = cases[name]
        expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
        records = {n: dict(case=name, program=n, times=[], max_abs_err=0.0, passed=True) for n in programs}
        order = list(programs)
        with tempfile.TemporaryDirectory(prefix="cuda-bench-", dir=scratch) as td:
            out = Path(td) / "out.bin"
            for repeat in range(args.runs):
                rotated = order[repeat % len(order):] + order[:repeat % len(order)]
                for label in rotated:
                    cmd = [str(Path(programs[label]).resolve()), str(data / "audio.pcm"), str(out), *flags]
                    t0 = time.perf_counter()
                    subprocess.run(cmd, check=True, capture_output=True)
                    elapsed = time.perf_counter() - t0
                    got = np.fromfile(out, dtype="<f4")
                    if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-3, atol=1e-4):
                        raise SystemExit(f"FAIL {name} {label} round={repeat}")
                    error = float(np.max(np.abs(got - expected)))
                    records[label]["times"].append(elapsed)
                    records[label]["max_abs_err"] = max(records[label]["max_abs_err"], error)
                    print(f"{name} {label} round={repeat} elapsed={elapsed:.6f} max_abs_err={error:.8g} PASS", flush=True)
        for record in records.values():
            record["min_s"] = min(record["times"])
            record["median_s"] = float(np.median(record["times"]))
            if "a2" in records:
                record["vs_a2"] = min(records["a2"]["times"]) / record["min_s"]
            results.append(record)
        del expected
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    for record in results:
        print(f"{record['case']} | {record['program']} | min={record['min_s']:.6f} | "
              f"median={record['median_s']:.6f} | vs_a2={record.get('vs_a2')}", flush=True)


if __name__ == "__main__":
    main()
