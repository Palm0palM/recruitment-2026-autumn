#!/usr/bin/env python3
"""多程序全量交错 benchmark：按排列轮转，校验官方 expected。

runs 达到程序数阶乘时覆盖全部顺序；较少 runs 只取排列序列前缀。
"""
import argparse
import itertools
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("programs", nargs="+", metavar="LABEL=PATH")
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    ap.add_argument("--runs", type=int, default=6, help="每个程序、每组运行次数")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.programs) < 2 or args.runs < 1:
        ap.error("至少两个程序且 --runs >= 1")
    labels, programs = [], []
    for spec in args.programs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path or label in labels:
            ap.error("程序格式为唯一 LABEL=PATH")
        labels.append(label)
        programs.append(str(Path(path).resolve()))
    cases = {case["name"]: case for case in load_official_cases()}
    if any(name not in cases for name in args.cases):
        ap.error("未知用例")
    data = find_data_dir()
    pcm = data / "audio.pcm"
    scratch = Path(os.environ.get("TMPDIR", str(ROOT / "tmp/bench")))
    scratch.mkdir(parents=True, exist_ok=True)
    orders = list(itertools.permutations(range(len(programs))))
    results = []
    for name in args.cases:
        case = cases[name]
        flags = [item for key, value in case["params"].items() for item in (f"--{key}", str(value))]
        expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        with tempfile.TemporaryDirectory(prefix=f"melspec-{name}-", dir=scratch) as td:
            outputs = [Path(td) / f"{label}.bin" for label in labels]
            commands = [[program, str(pcm), str(output), *flags]
                        for program, output in zip(programs, outputs)]
            for command in commands:
                subprocess.run(command, check=True, capture_output=True)
            times = [[] for _ in programs]
            for iteration in range(args.runs):
                for index in orders[iteration % len(orders)]:
                    start = time.perf_counter()
                    subprocess.run(commands[index], check=True, capture_output=True)
                    times[index].append(time.perf_counter() - start)
            row = {"case": name, "runs": args.runs, "order": [list(map(str, order)) for order in orders],
                   "programs": []}
            for index, label in enumerate(labels):
                got = np.fromfile(outputs[index], dtype="<f4")
                passed = got.shape == expected.shape and np.allclose(got, expected, rtol=1e-3, atol=1e-4)
                row["programs"].append({
                    "label": label, "path": programs[index], "times_s": times[index],
                    "min_s": min(times[index]), "median_s": statistics.median(times[index]),
                    "pass": bool(passed),
                    "max_abs_err": float(np.max(np.abs(got - expected))) if got.shape == expected.shape else None})
                print(f"{name}/{label}: min={min(times[index]):.6f}s "
                      f"median={statistics.median(times[index]):.6f}s PASS={passed}", flush=True)
                print("  runs: " + " ".join(f"{value:.6f}" for value in times[index]))
            results.append(row)
        del expected
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    if any(not program["pass"] for row in results for program in row["programs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
