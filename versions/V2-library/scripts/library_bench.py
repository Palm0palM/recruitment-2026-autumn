#!/usr/bin/env python3
"""Interleaved, correctness-checked process benchmark for library candidates.

A program spec is `name=command template`; `{pcm}` and `{out}` are substituted,
and official case flags are appended. Uses wait4 for exact per-process wall,
user, sys and peak RSS. Set MELSPEC_STATS in the child for optional stage JSON.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from bench import find_data_dir, load_official_cases


def run_child(argv, env, stderr_path):
    with stderr_path.open("wb") as err:
        before = time.perf_counter()
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=err)
        pid, status, usage = os.wait4(proc.pid, 0)
        elapsed = time.perf_counter() - before
        proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, elapsed, usage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("programs", nargs="+", help="name=command template")
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats-dir", type=Path, default=ROOT / "tmp/results/stats")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be positive")
    programs = {}
    for item in args.programs:
        if "=" not in item:
            ap.error(f"program must be name=command: {item}")
        name, command = item.split("=", 1)
        if not name or name in programs:
            ap.error(f"empty or duplicate program name: {name}")
        programs[name] = shlex.split(command)
    cases = {c["name"]: c for c in load_official_cases()}
    if any(n not in cases for n in args.cases):
        ap.error("unknown case")
    all_cases = load_official_cases()
    if any(not any(c["name"] == n for c in all_cases) for n in args.cases):
        ap.error("unknown case")
    data = find_data_dir()
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "tmp/logs"
    logs.mkdir(parents=True, exist_ok=True)
    records = []
    for case_name in args.cases:
        case = cases[case_name]
        expected_path = data / Path(case["expected"]).relative_to("data")
        expected = np.fromfile(expected_path, dtype="<f4")
        flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
        with __import__("tempfile").TemporaryDirectory(
                dir=Path(os.environ.get("TMPDIR", ROOT / "tmp/scratch"))) as td:
            output_path = Path(td) / "out.bin"
            order = list(programs)
            for round_no in range(args.runs):
                rotated = order[round_no % len(order):] + order[:round_no % len(order)]
                for name in rotated:
                    argv = list(programs[name])
                    argv = [a.replace("{pcm}", str(data / "audio.pcm")).replace("{out}", str(output_path))
                            for a in argv] + flags
                    stats_path = args.stats_dir / f"{case_name}-{name}-r{round_no}.json"
                    if stats_path.exists():
                        stats_path.unlink()
                    env = dict(os.environ, MELSPEC_STATS=str(stats_path))
                    log_path = logs / f"library-{case_name}-{name}-r{round_no}.err"
                    rc, elapsed, usage = run_child(argv, env, log_path)
                    if rc != 0:
                        raise SystemExit(f"FAIL rc={rc}: {name} {case_name} r{round_no}; see {log_path}")
                    got = np.fromfile(output_path, dtype="<f4")
                    passed = got.shape == expected.shape and bool(
                        np.allclose(got, expected, rtol=1e-3, atol=1e-4))
                    if not passed:
                        raise SystemExit(f"FAIL correctness: {name} {case_name} r{round_no}")
                    err = float(np.max(np.abs(got - expected))) if got.shape == expected.shape else None
                    record = dict(case=case_name, program=name, round=round_no,
                                  elapsed_s=elapsed, user_s=usage.ru_utime,
                                  sys_s=usage.ru_stime, maxrss_kib=usage.ru_maxrss,
                                  max_abs_err=err, passed=True, stats=None,
                                  command=argv)
                    if stats_path.exists():
                        record["stats"] = json.loads(stats_path.read_text())
                        # wait4's ru_maxrss is a high-water mark across children
                        # on Linux; prefer the implementation's own peak RSS.
                        own = record["stats"].get("peak_rss_kib",
                                                  record["stats"].get("maxrss_kib"))
                        if own:
                            record["maxrss_kib"] = own
                    records.append(record)
                    print(f"{case_name} {name} r{round_no} wall={elapsed:.6f}s "
                          f"rss={record['maxrss_kib']}KiB err={err if err is not None else float('nan'):.8g} PASS", flush=True)
        del expected
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    print("case | program | min_s | median_s | maxrss_kib")
    for case_name in args.cases:
        for name in programs:
            rows = [r for r in records if r["case"] == case_name and r["program"] == name]
            times = sorted(r["elapsed_s"] for r in rows)
            print(f"{case_name} | {name} | {times[0]:.6f} | "
                  f"{np.median(times):.6f} | {max(r['maxrss_kib'] for r in rows)}")


if __name__ == "__main__":
    main()
