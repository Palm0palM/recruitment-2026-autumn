#!/usr/bin/env python3
"""V5 interleaved full-process timing; all readback/validation OUTSIDE timer.

Programs are LABEL=PATH, one label must be cpu (the numerical reference).
Every measured output is checked; official points additionally use expected.
Each row is checkpointed so interrupted jobs retain all completed evidence.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from hybrid_bench import command, official_configs, synthetic_configs


def configs(suite):
    official = official_configs()
    if suite == "official":
        return official
    if suite == "matrix":
        return official + synthetic_configs()
    if suite == "extended":
        return [dict(name=f"extended-{n_fft}-{frames}", synthetic=True,
                     frames=frames, params=dict(sr=96000, n_fft=n_fft,
                     hop_length=128, n_mels=256, f_min=0, f_max=22050))
                for n_fft in (32768, 65536) for frames in (4096, 32768, 65536)]
    # Unseen frame counts, hops, FFTs and output dimensions for holdout.
    holdout = []
    for n_fft, frames, hop, mels in (
        (1024, 33, 73, 19), (4096, 4097, 31, 97),
        (8192, 70001, 257, 128), (16384, 80003, 192, 96),
        (8192, 250003, 64, 64), (16384, 230003, 128, 256),
        (4096, 65539, 1027, 32), (2048, 513, 509, 31),
        (32768, 22003, 257, 97), (65536, 17003, 128, 96),
    ):
        holdout.append(dict(name=f"holdout-{n_fft}-{frames}-{hop}-{mels}",
                            params=dict(sr=96000, n_fft=n_fft, hop_length=hop,
                                        n_mels=mels, f_min=233, f_max=17003),
                            frames=frames, synthetic=True))
    return official + holdout


def invoke(program, pcm, output, params, label):
    env = dict(os.environ)
    env.pop("MELSPEC_BACKEND", None)
    env.pop("MELSPEC_DISPATCH_LOG", None)
    if label.startswith("auto"):
        env["MELSPEC_DISPATCH_LOG"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(command(program, pcm, output, params), env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    stderr = proc.stderr.decode(errors="replace")
    if proc.returncode:
        raise RuntimeError(f"{label} exit={proc.returncode}: {stderr}")
    # Deliberately after elapsed is captured. Never time Python output reads.
    return elapsed, np.fromfile(output, dtype="<f4"), stderr


def check(got, expected, shape, context):
    if got.size != shape or not np.isfinite(got).all():
        raise RuntimeError(f"{context}: size/nonfinite failure")
    if not np.allclose(got, expected, rtol=1e-3, atol=1e-4):
        raise RuntimeError(f"{context}: allclose FAIL max={np.max(np.abs(got-expected))}")
    return float(np.max(np.abs(got - expected)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("programs", nargs="+")
    ap.add_argument("--suite", choices=("official", "matrix", "holdout", "extended"), default="matrix")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.runs < 2:
        ap.error("at least two interleaved runs required")
    programs = dict(item.split("=", 1) for item in args.programs)
    programs = {k: str(Path(v).resolve()) for k, v in programs.items()}
    if "cpu" not in programs:
        ap.error("cpu=PATH reference required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(host=platform.node(), affinity=sorted(os.sched_getaffinity(0)),
                programs=programs, suite=args.suite, runs=args.runs,
                timing="subprocess.run wall only; readback and allclose excluded")
    meta["binary_sha256"] = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                             for k, v in programs.items()}
    meta["environment"] = {k: os.environ.get(k) for k in (
        "ROCFFT_RTC_CACHE_PATH", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TMPDIR")}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    results = []
    names = list(programs)
    for config in configs(args.suite):
        p = config["params"]
        frames = (config["frames"] if config["synthetic"] else
                  1 + (config["pcm"].stat().st_size // 2 - p["n_fft"]) // p["hop_length"])
        shape = frames * p["n_mels"]
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="v5-") as td:
            td = Path(td)
            if config["synthetic"]:
                samples = p["n_fft"] + (frames - 1) * p["hop_length"]
                pcm = td / "input.pcm"
                rng = np.random.default_rng(20261031 + samples + p["n_mels"])
                rng.integers(-32768, 32768, samples, dtype=np.int16).astype("<i2").tofile(pcm)
            else:
                pcm = config["pcm"]
            output = td / "output.bin"
            _, ref, _ = invoke(programs["cpu"], pcm, output, p, "cpu")
            expected = np.fromfile(config["expected"], dtype="<f4") if "expected" in config else ref
            check(ref, expected, shape, config["name"] + "/reference")
            row = dict(name=config["name"], params=p, n_frames=frames,
                       input_bytes=pcm.stat().st_size, output_elements=shape,
                       times={k: [] for k in names}, checks={}, decisions={}, order=[])
            # One explicitly unscored correctness/cache-population run per
            # candidate. Every timed process still pays runtime/context setup.
            # On HIP the first plan may JIT; record it rather than hiding it.
            row["warmup_times"] = {}
            for label in names:
                elapsed, got, _ = invoke(programs[label], pcm, output, p, label)
                check(got, expected, shape, f"{config['name']}/{label}/warmup")
                row["warmup_times"][label] = elapsed
            for repeat in range(args.runs):
                # Latin rotation and reversal gives balanced positions without
                # exhausting factorial permutations for larger comparisons.
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                if (repeat // len(names)) % 2:
                    order = list(reversed(order))
                row["order"].append(order)
                for label in order:
                    elapsed, got, stderr = invoke(programs[label], pcm, output, p, label)
                    error = check(got, expected, shape, f"{config['name']}/{label}/{repeat}")
                    row["times"][label].append(elapsed)
                    row["checks"][label] = dict(passed=True, max_abs=max(
                        error, row["checks"].get(label, {}).get("max_abs", 0)))
                    match = re.search(r"dispatch: backend=(cpu|cuda|hip)", stderr)
                    if match:
                        row["decisions"].setdefault(label, []).append(match[1])
            for label, times in row["times"].items():
                row[label + "_median"] = statistics.median(times)
                row[label + "_min"] = min(times)
                row[label + "_pass"] = True
            results.append(row)
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            print(config["name"], " ".join(f"{k}={row[k+'_median']:.5f}" for k in names),
                  "decisions=" + str(row["decisions"]), "PASS", flush=True)


if __name__ == "__main__":
    main()
