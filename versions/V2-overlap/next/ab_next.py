#!/usr/bin/env python3
"""下一轮候选的全量多线程轮转 A/B，保留逐次 wall/CPU time。"""
from __future__ import annotations
import argparse, json, resource, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from bench import ROOT, find_data_dir, load_official_cases

PROGRAMS = {
    "v1": "tmp/binaries/melspec-v1",
    "half_delta": "tmp/binaries/melspec-overlap-half-delta",
    "pruned": "tmp/binaries/melspec-overlap-pruned-power",
    "fftw": "tmp/binaries/melspec-overlap-fftw",
    "codelet": "tmp/binaries/melspec-overlap-codelet",
    "fftw_pruned": "tmp/binaries/melspec-overlap-fftw-pruned",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="default")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--programs", default="v1,half_delta,pruned,fftw,codelet,fftw_pruned")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    names = args.programs.split(",")
    if args.rounds < 1 or any(n not in PROGRAMS for n in names):
        ap.error("invalid rounds or program list")
    case = next(c for c in load_official_cases() if c["name"] == args.case)
    data = find_data_dir()
    flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
    wall = {n: [] for n in names}
    cpu = {n: [] for n in names}
    rows = []
    scratch = ROOT / "tmp/scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        output = Path(td) / "out.bin"
        for round_index in range(args.rounds):
            shift = round_index % len(names)
            order = names[shift:] + names[:shift]
            if round_index % 2:
                order = order[::-1]
            for name in order:
                cmd = [str(ROOT / PROGRAMS[name]), str(data / "audio.pcm"), str(output), *flags]
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                started = time.perf_counter()
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elapsed = time.perf_counter() - started
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu_elapsed = after.ru_utime - before.ru_utime + after.ru_stime - before.ru_stime
                wall[name].append(elapsed)
                cpu[name].append(cpu_elapsed)
                row = dict(round=round_index, variant=name, wall=elapsed, cpu=cpu_elapsed)
                rows.append(row)
                print(json.dumps(row), flush=True)
        expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        correctness = {}
        for name in names:
            subprocess.run([ROOT / PROGRAMS[name], data / "audio.pcm", output, *flags],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            got = np.fromfile(output, dtype="<f4")
            passed = got.shape == expected.shape and bool(np.allclose(got, expected, rtol=1e-3, atol=1e-4))
            correctness[name] = dict(passed=passed,
                max_abs_err=float(np.max(np.abs(got - expected))) if got.shape == expected.shape else None)
            if not passed:
                raise SystemExit(f"{name}: FAIL")
            del got
        del expected
    ratios = {}
    for name in names:
        if name == "v1":
            continue
        for metric, values in (("wall", wall), ("cpu", cpu)):
            ratios[f"{name}_over_v1_{metric}"] = [x / v for x, v in zip(values[name], values["v1"])]
            if "half_delta" in names and name != "half_delta":
                ratios[f"{name}_over_delta_{metric}"] = [x / v for x, v in zip(values[name], values["half_delta"])]
    result = dict(case=args.case, rounds=args.rounds, wall=wall, cpu=cpu,
                  correctness=correctness, ratios=ratios,
                  median_ratios={k: float(np.median(v)) for k, v in ratios.items()},
                  median_wall={k: float(np.median(v)) for k, v in wall.items()},
                  median_cpu={k: float(np.median(v)) for k, v in cpu.items()}, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
