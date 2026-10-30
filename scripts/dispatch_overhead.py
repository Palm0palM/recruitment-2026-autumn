#!/usr/bin/env python3
"""Separate dispatcher cost from CUDA shared-library load cost.

Three programs are timed on the same points, in the same rotation:

  cpu-only      CPU-only build             (no CUDA DSO at all)
  unified-cpu   unified build, forced CPU  (loads CUDA DSOs, no device init)
  auto          unified build, auto        (model decides)

  unified-cpu - cpu-only  = cost of shipping both backends in one binary
  auto        - unified-cpu = cost of the dispatcher itself
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

from bench import ROOT, find_data_dir
from hybrid_bench import official_configs, synthetic_configs

# Small, mid, large, plus every official case. Names come from the H1 matrix.
SELECTED = [
    "official-default", "official-small_mels", "official-large_fft",
    "official-narrow_band", "frames-64", "frames-4096", "frames-32768",
    "frames-192397", "nfft-8192", "nfft-16384", "mels-64", "large-32768",
    "large-65536",
]


def command(program, pcm, output, params):
    flags = [s for key, value in params.items() for s in (f"--{key}", str(value))]
    return [program, str(pcm), str(output), *flags]


def run(program, pcm, output, params, env=None):
    proc = subprocess.run(command(program, pcm, output, params),
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(command(program, pcm, output, params))}"
                         f"\n{proc.stderr.decode(errors='replace')}")
    return np.fromfile(output, dtype="<f4"), proc.stderr.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-only", required=True)
    ap.add_argument("--unified-cpu", required=True)
    ap.add_argument("--auto", required=True)
    ap.add_argument("--cpu-old", default=None,
                    help="optional older CPU-only build, to expose code-layout drift")
    ap.add_argument("--auto-nogpu", default=None,
                    help="optional auto program run with no visible device, to cost a failed probe")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    configs = [c for c in official_configs() + synthetic_configs()
               if c["name"] in SELECTED]
    if len(configs) != len(SELECTED):
        raise SystemExit("missing benchmark points")
    programs = {
        "cpu-only": (str(Path(args.cpu_only).resolve()), os.environ),
        "unified-cpu": (str(Path(args.unified_cpu).resolve()),
                        dict(os.environ, MELSPEC_BACKEND="cpu")),
        "auto": (str(Path(args.auto).resolve()),
                 dict(os.environ, MELSPEC_DISPATCH_LOG="1")),
    }
    if args.cpu_old:
        programs["cpu-old"] = (str(Path(args.cpu_old).resolve()), os.environ)
    if args.auto_nogpu:
        # Same auto binary, but the device probe fails: the delta against
        # unified-cpu is the cost of that failed probe.
        programs["auto-nogpu"] = (str(Path(args.auto_nogpu).resolve()),
                                  dict(os.environ, MELSPEC_DISPATCH_LOG="1",
                                       CUDA_VISIBLE_DEVICES=""))
    scratch_root = Path(os.environ.get("TMPDIR", str(ROOT / "tmp" / "bench")))
    scratch_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in configs:
        params = config["params"]
        n_fft, hop = params["n_fft"], params["hop_length"]
        if config["synthetic"]:
            n_frames = config["frames"]
        else:
            n_frames = 1 + (find_data_dir().joinpath("audio.pcm").stat().st_size // 2
                            - n_fft) // hop
        with tempfile.TemporaryDirectory(prefix="dispatch-overhead-", dir=scratch_root) as td:
            td = Path(td)
            if config["synthetic"]:
                samples = n_fft + (n_frames - 1) * hop
                rng = np.random.default_rng(1234)
                pcm = td / "input.pcm"
                rng.integers(-32768, 32768, samples, dtype=np.int16).astype("<i2").tofile(pcm)
            else:
                pcm = config["pcm"]
            reference, _ = run(programs["cpu-only"][0], pcm, td / "ref.bin", params)
            oracle = (np.fromfile(config["expected"], dtype="<f4")
                      if "expected" in config else reference)
            times = {name: [] for name in programs}
            auto_backend = "unknown"
            nogpu_backend = "unknown"
            nogpu_output = None
            for repeat in range(args.runs):
                order = list(programs) if repeat % 2 == 0 else list(reversed(list(programs)))
                for name in order:
                    program, env = programs[name]
                    start = time.perf_counter()
                    got, stderr = run(program, pcm, td / f"{name}.bin", params, env)
                    times[name].append(time.perf_counter() - start)
                    if not np.allclose(got, oracle, rtol=1e-3, atol=1e-4):
                        raise SystemExit(f"{config['name']}/{name}: output mismatch")
                    if name == "auto" and "backend=" in stderr:
                        auto_backend = stderr.split("backend=")[1].split()[0]
                    if name == "auto-nogpu":
                        nogpu_output = got
                        if "backend=" in stderr:
                            nogpu_backend = stderr.split("backend=")[1].split()[0]
            row = {"name": config["name"], "n_frames": n_frames,
                   "auto_backend": auto_backend,
                   "times": times,
                   **{f"{k}_min": min(v) for k, v in times.items()},
                   **{f"{k}_median": statistics.median(v) for k, v in times.items()}}
            row["dso_load_ms"] = 1000 * (row["unified-cpu_median"] - row["cpu-only_median"])
            # Only meaningful when the model picked CPU; otherwise auto is on the
            # GPU and the difference is a backend change, not dispatcher cost.
            row["dispatcher_ms"] = (1000 * (row["auto_median"] - row["unified-cpu_median"])
                                    if auto_backend == "cpu" else None)
            if "auto-nogpu" in times:
                row["nogpu_backend"] = nogpu_backend
                row["failed_probe_ms"] = (1000 * (row["auto-nogpu_median"]
                                                 - row["unified-cpu_median"]))
                row["nogpu_equals_cpu"] = bool(np.allclose(nogpu_output, reference,
                                                           rtol=0.0, atol=0.0))
            if "cpu-old" in times:
                row["old_build_delta_pct"] = 100 * (row["cpu-only_median"] /
                                                    row["cpu-old_median"] - 1)
            rows.append(row)
            line = (f"{config['name']:24s} frames={n_frames:7d} "
                    f"cpu-only={row['cpu-only_median']:.4f} "
                    f"unified-cpu={row['unified-cpu_median']:.4f} "
                    f"auto={row['auto_median']:.4f} ({auto_backend}) "
                    f"dso={row['dso_load_ms']:+.1f}ms")
            if row["dispatcher_ms"] is not None:
                line += f" dispatch={row['dispatcher_ms']:+.1f}ms"
            else:
                line += " dispatch=n/a (auto chose GPU)"
            if "cpu-old" in times:
                line += f" old-build={row['cpu-old_median']:.4f} ({row['old_build_delta_pct']:+.1f}%)"
            if "auto-nogpu" in times:
                line += (f" nogpu={row['auto-nogpu_median']:.4f} ({nogpu_backend}, "
                         f"probe={row['failed_probe_ms']:+.1f}ms)")
            print(line, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
