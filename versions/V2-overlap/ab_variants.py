#!/usr/bin/env python3
"""V1/full/half/half-delta 全量轮转 A/B，记录 wall 与 CPU time。"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_benchmark import BASE, build_single_thread
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench import ROOT, find_data_dir, load_official_cases

VARIANTS = [
    ("v1", "tmp/binaries/melspec-v1", None),
    ("full", "tmp/binaries/melspec-overlap-full", "melspec_overlap_prototype.cpp"),
    ("half", "tmp/binaries/melspec-overlap-half", "melspec_overlap_half_prototype.cpp"),
    ("half_delta", "tmp/binaries/melspec-overlap-half-delta", "melspec_overlap_half_delta_prototype.cpp"),
]


def median(values):
    return float(np.median(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="default")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--single-thread", action="store_true")
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.rounds < 1:
        ap.error("--rounds must be >= 1")
    case = next(c for c in load_official_cases() if c["name"] == args.case)
    data = find_data_dir()
    flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
    programs = []
    for name, binary, source_name in VARIANTS:
        path = ROOT / binary
        if args.single_thread:
            if source_name is None:
                source = subprocess.check_output(["git", "show", f"{BASE}:src/melspec.cpp"],
                                                 cwd=ROOT, text=True)
            else:
                source = (ROOT / "versions/V2-overlap" / source_name).read_text()
            path = build_single_thread(f"variant-{name}-single", source)
        programs.append((name, path))
    wall = {name: [] for name, _ in programs}
    cpu = {name: [] for name, _ in programs}
    rows = []
    scratch = ROOT / "tmp/scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        output = Path(td) / "out.bin"
        for round_index in range(args.rounds):
            shift = round_index % len(programs)
            order = programs[shift:] + programs[:shift]
            if round_index % 2:
                order = order[::-1]
            for name, program in order:
                cmd = [str(program), str(data / "audio.pcm"), str(output), *flags]
                if args.single_thread:
                    cmd = ["taskset", "-c", str(args.cpu), *cmd]
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                started = time.perf_counter()
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elapsed = time.perf_counter() - started
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu_elapsed = (after.ru_utime - before.ru_utime +
                               after.ru_stime - before.ru_stime)
                wall[name].append(elapsed)
                cpu[name].append(cpu_elapsed)
                row = dict(round=round_index, variant=name, wall=elapsed, cpu=cpu_elapsed)
                rows.append(row)
                print(json.dumps(row), flush=True)
        correctness = {}
        expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        for name, program in programs:
            cmd = [str(program), str(data / "audio.pcm"), str(output), *flags]
            if args.single_thread:
                cmd = ["taskset", "-c", str(args.cpu), *cmd]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            got = np.fromfile(output, dtype="<f4")
            passed = got.shape == expected.shape and bool(np.allclose(got, expected, rtol=1e-3, atol=1e-4))
            correctness[name] = dict(passed=passed,
                max_abs_err=float(np.max(np.abs(got - expected))) if got.shape == expected.shape else None)
            if not passed:
                raise SystemExit(f"{name}: output FAIL")
            del got
        del expected
    ratios = {}
    for name in ("full", "half", "half_delta"):
        ratios[f"{name}_over_v1_cpu"] = [x / v for x, v in zip(cpu[name], cpu["v1"])]
        ratios[f"{name}_over_v1_wall"] = [x / v for x, v in zip(wall[name], wall["v1"])]
    ratios["delta_over_half_cpu"] = [x / v for x, v in zip(cpu["half_delta"], cpu["half"])]
    ratios["delta_over_half_wall"] = [x / v for x, v in zip(wall["half_delta"], wall["half"])]
    result = dict(case=args.case, rounds=args.rounds, single_thread=args.single_thread,
                  wall=wall, cpu=cpu, correctness=correctness,
                  median_wall={k: median(v) for k, v in wall.items()},
                  median_cpu={k: median(v) for k, v in cpu.items()},
                  ratios=ratios, median_ratios={k: median(v) for k, v in ratios.items()},
                  rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
