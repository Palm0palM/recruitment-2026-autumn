#!/usr/bin/env python3
"""V1/V2 全量交错 A/B；只读公共数据，产物写 worktree 私有 tmp/。"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench import ROOT, find_data_dir, load_official_cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=ROOT / "tmp/binaries/melspec-v1")
    ap.add_argument("--candidate", type=Path, default=ROOT / "melspec")
    ap.add_argument("--case", default="default")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--single-thread", action="store_true")
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.pairs < 1:
        ap.error("--pairs must be >= 1")
    case = next(c for c in load_official_cases() if c["name"] == args.case)
    data = find_data_dir()
    scratch = ROOT / "tmp/scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
    programs = [("v1", args.baseline.resolve()), ("v2", args.candidate.resolve())]
    times = {name: [] for name, _ in programs}
    rows = []
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        output = Path(td) / "out.bin"
        for pair in range(args.pairs):
            # 交换每对的先后顺序，降低固定热缓存/负载偏差。
            order = programs if pair % 2 == 0 else programs[::-1]
            for name, program in order:
                cmd = [str(program), str(data / "audio.pcm"), str(output), *flags]
                if args.single_thread:
                    cmd = ["taskset", "-c", str(args.cpu), *cmd]
                started = time.perf_counter()
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elapsed = time.perf_counter() - started
                times[name].append(elapsed)
                row = dict(pair=pair, version=name, seconds=elapsed)
                rows.append(row)
                print(json.dumps(row), flush=True)
        # 最后一对可能由 V1 收尾；正确性始终单独检查 candidate。
        cmd = [str(args.candidate.resolve()), str(data / "audio.pcm"), str(output), *flags]
        if args.single_thread:
            cmd = ["taskset", "-c", str(args.cpu), *cmd]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        got = np.fromfile(output, dtype="<f4")
        expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        passed = got.shape == expected.shape and bool(np.allclose(got, expected, rtol=1e-3, atol=1e-4))
        max_abs_err = float(np.max(np.abs(got - expected))) if got.shape == expected.shape else None
    result = dict(case=args.case, single_thread=args.single_thread,
                  cpu=args.cpu if args.single_thread else None,
                  passed=passed, max_abs_err=max_abs_err,
                  times=times,
                  min={k: min(v) for k, v in times.items()},
                  median={k: float(np.median(v)) for k, v in times.items()},
                  v2_over_v1_min=min(times["v2"]) / min(times["v1"]),
                  rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)
    if not passed:
        raise SystemExit("candidate output FAIL")


if __name__ == "__main__":
    main()
