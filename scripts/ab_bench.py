#!/usr/bin/env python3
"""同一作业内交替 A/B 全量测量，官方 expected 校验；不运行 librosa。"""
import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--cases", nargs="+", default=["default", "large_fft"])
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be >= 1")
    programs = [str(Path(args.a).resolve()), str(Path(args.b).resolve())]
    cases = {c["name"]: c for c in load_official_cases()}
    if any(name not in cases for name in args.cases):
        ap.error("unknown case")
    data = find_data_dir()
    pcm = data / "audio.pcm"
    (ROOT / "tmp/bench").mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.cases:
        case = cases[name]
        flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp/bench") as td:
            outputs = [Path(td) / "a.bin", Path(td) / "b.bin"]
            commands = [[program, str(pcm), str(output), *flags]
                        for program, output in zip(programs, outputs)]
            for command in commands:
                subprocess.run(command, check=True, capture_output=True)
            times = [[], []]
            for iteration in range(args.runs):
                order = (0, 1) if iteration % 2 == 0 else (1, 0)
                for index in order:
                    start = time.perf_counter()
                    subprocess.run(commands[index], check=True, capture_output=True)
                    times[index].append(time.perf_counter() - start)
            row = {"case": name, "runs": args.runs, "times": times}
            for index, label in enumerate(("a", "b")):
                got = np.fromfile(outputs[index], dtype="<f4")
                expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
                row[label] = {"program": programs[index], "min_s": min(times[index]),
                              "median_s": statistics.median(times[index]),
                              "pass": bool(got.shape == expected.shape and
                                           np.allclose(got, expected, rtol=1e-3, atol=1e-4)),
                              "max_abs_err": float(np.max(np.abs(got - expected)))
                              if got.shape == expected.shape else None}
            row["median_speedup_b_over_a"] = row["a"]["median_s"] / row["b"]["median_s"]
            results.append(row)
            print(f"{name}: A min/med {row['a']['min_s']:.6f}/{row['a']['median_s']:.6f}s; "
                  f"B min/med {row['b']['min_s']:.6f}/{row['b']['median_s']:.6f}s; "
                  f"median B speedup {row['median_speedup_b_over_a']:.4f}x; "
                  f"PASS A/B={row['a']['pass']}/{row['b']['pass']}", flush=True)
            print("  A runs: " + " ".join(f"{t:.6f}" for t in times[0]))
            print("  B runs: " + " ".join(f"{t:.6f}" for t in times[1]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    if any(not row[label]["pass"] for row in results for label in ("a", "b")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
