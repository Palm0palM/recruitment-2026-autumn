#!/usr/bin/env python3
"""Serial, interleaved A2/MPI strong-scaling benchmark inside one Slurm job.

Each external wall time includes mpirun, MPI initialization, distribution,
compute, float conversion, gather, root write and finalization. All outputs are
checked against the full official expected arrays.
"""
import argparse
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
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", default=["a2", "1:bcast", "1:scatter", "1:shared",
                    "2:bcast", "2:scatter", "2:shared", "4:bcast", "4:scatter", "4:shared"])
    args = ap.parse_args()
    if args.runs < 1 or "a2" not in args.configs:
        ap.error("--runs must be positive and --configs must include a2")
    mpi_home = Path(os.environ.get("OPENMPI_HOME", "/cache/toolchain/openmpi-4.1.6"))
    mpirun = mpi_home / "bin/mpirun"
    driver = ROOT / "tmp/bin/mpi-melspec"
    reference = ROOT / "tmp/bin/a2-estimate"
    data = find_data_dir()
    cases = {c["name"]: c for c in load_official_cases()}
    if any(name not in cases for name in args.cases):
        ap.error("unknown case")
    results = []
    for case_name in args.cases:
        case = cases[case_name]
        expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
        records = {config: [] for config in args.configs}
        with tempfile.TemporaryDirectory(prefix="mpi-bench-") as td:
            output = Path(td) / "out.bin"
            for repeat in range(args.runs):
                order = args.configs[repeat % len(args.configs):] + args.configs[:repeat % len(args.configs)]
                for config in order:
                    if config == "a2":
                        ranks, mode = 1, "serial-a2"
                        cmd = [str(reference), str(data / "audio.pcm"), str(output), *flags]
                    else:
                        ranks_text, mode = config.split(":", 1)
                        ranks = int(ranks_text)
                        cmd = [str(mpirun), "-np", str(ranks), "--map-by", "ppr:1:node:pe=16",
                               "--bind-to", "none", str(driver), str(data / "audio.pcm"), str(output), *flags]
                    env = dict(os.environ, MELSPEC_MPI_IO=mode)
                    start = time.perf_counter()
                    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
                    wall = time.perf_counter() - start
                    got = np.fromfile(output, dtype="<f4")
                    passed = got.shape == expected.shape and bool(np.allclose(got, expected, rtol=1e-3, atol=1e-4))
                    if not passed:
                        raise SystemExit(f"FAIL {case_name} {config} round={repeat}\n{proc.stdout}\n{proc.stderr}")
                    error = float(np.max(np.abs(got - expected)))
                    timing = [line for line in proc.stdout.splitlines() if line.startswith("MPI_TIMING")]
                    record = dict(case=case_name, config=config, ranks=ranks, io=mode,
                                  round=repeat, wall_s=wall, max_abs_err=error, passed=True,
                                  command=cmd, timing=timing)
                    records[config].append(record)
                    results.append(record)
                    print(f"{case_name} {config} round={repeat} wall_s={wall:.6f} err={error:.8g} PASS", flush=True)
        baseline = min(r["wall_s"] for r in records["a2"])
        for config, rows in records.items():
            times = [r["wall_s"] for r in rows]
            summary = dict(case=case_name, config=config, min_s=min(times),
                           median_s=statistics.median(times),
                           vs_a2=baseline / min(times),
                           efficiency=baseline / (rows[0]["ranks"] * min(times)))
            print("SUMMARY " + json.dumps(summary), flush=True)
        del expected
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
