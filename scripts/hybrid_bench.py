#!/usr/bin/env python3
"""CPU/GPU full-process interleaved benchmark for the Hybrid dispatcher study.

Every point is validated (official cases against the shipped expected output,
synthetic cases CPU vs CUDA) before it is timed. Timing is full-process wall
clock, interleaved and rotated between backends so that drift hits all of them.

If --auto is given, auto mode is timed in the same rotation and its dispatch
decision is read from MELSPEC_DISPATCH_LOG output.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases

DISPATCH_RE = re.compile(r"dispatch: backend=(cpu|cuda)")


def official_configs():
    data = find_data_dir()
    result = []
    for case in load_official_cases():
        result.append({
            "name": "official-" + case["name"],
            "params": dict(case["params"]),
            "pcm": data / "audio.pcm",
            "expected": data / Path(case["expected"]).relative_to("data"),
            "synthetic": False,
        })
    return result


def synthetic_configs():
    base = dict(sr=96000, hop_length=128)
    result = []

    def add(name, n_fft, frames, n_mels, f_min=0.0, f_max=22050.0):
        result.append({
            "name": name,
            "params": {**base, "n_fft": n_fft, "n_mels": n_mels,
                       "f_min": f_min, "f_max": f_max},
            "frames": frames,
            "synthetic": True,
        })

    for frames in (1, 8, 64, 512, 4096, 32768, 192397):
        add(f"frames-{frames}", 8192, frames, 256)
    for n_fft in (256, 512, 1024, 2048, 4096, 8192, 16384):
        add(f"nfft-{n_fft}", n_fft, 4096, 256)
    for n_mels in (8, 16, 32, 64, 128, 256):
        add(f"mels-{n_mels}", 8192, 4096, n_mels)
    for n_fft, frames in ((8192, 4096), (16384, 4096)):
        for f_min, f_max in ((0, 22050), (300, 8000), (0, 8000),
                             (300, 22050), (0, 48000)):
            add(f"band-{n_fft}-{int(f_min)}-{int(f_max)}",
                n_fft, frames, 256, f_min, f_max)
    for frames in (32768, 65536, 128000, 192333):
        add(f"large-{frames}", 16384, frames, 256)
    return result


def command(program, pcm, output, params):
    flags = [s for key, value in params.items() for s in (f"--{key}", str(value))]
    return [program, str(pcm), str(output), *flags]


def run(program, pcm, output, params, env=None):
    """Run one full process, returning (float32 output, stderr text)."""
    proc = subprocess.run(command(program, pcm, output, params),
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          env=env)
    stderr = proc.stderr.decode(errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(command(program, pcm, output, params))}\n{stderr}")
    return np.fromfile(output, dtype="<f4"), stderr


def timed(program, pcm, output, params, env=None):
    """Time one full process; also return its stderr for dispatch logging."""
    start = time.perf_counter()
    proc = subprocess.run(command(program, pcm, output, params), env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return elapsed, proc.stderr.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", required=True, help="forced-CPU program")
    ap.add_argument("--cuda", required=True, help="forced-CUDA program")
    ap.add_argument("--auto", default=None, help="auto-dispatch program (optional)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be positive")
    configs = official_configs() + synthetic_configs()
    programs = {
        "cpu": str(Path(args.cpu).resolve()),
        "cuda": str(Path(args.cuda).resolve()),
    }
    if args.auto:
        programs["auto"] = str(Path(args.auto).resolve())
    auto_env = dict(os.environ, MELSPEC_DISPATCH_LOG="1")
    scratch_root = Path(os.environ.get("TMPDIR", str(ROOT / "tmp" / "bench")))
    scratch_root.mkdir(parents=True, exist_ok=True)
    results = []
    for config in configs:
        params = config["params"]
        n_frames = (config.get("frames") if config["synthetic"]
                    else 1 + (find_data_dir().joinpath("audio.pcm").stat().st_size // 2
                              - params["n_fft"]) // params["hop_length"])
        output_elements = params["n_mels"] * n_frames
        with tempfile.TemporaryDirectory(prefix="hybrid-bench-", dir=scratch_root) as td:
            td = Path(td)
            if config["synthetic"]:
                samples = params["n_fft"] + (n_frames - 1) * params["hop_length"]
                seed = (params["n_fft"] * 1000003 + n_frames * 9176 +
                        params["n_mels"] * 65537 + int(params["f_min"]) * 131 +
                        int(params["f_max"])) % (2**32)
                rng = np.random.default_rng(seed)
                pcm = td / "input.pcm"
                rng.integers(-32768, 32768, samples, dtype=np.int16).astype("<i2").tofile(pcm)
            else:
                pcm = config["pcm"]
            ref_cpu, _ = run(programs["cpu"], pcm, td / "cpu.bin", params)
            got_cuda, _ = run(programs["cuda"], pcm, td / "cuda.bin", params)
            if ref_cpu.size != output_elements or got_cuda.size != output_elements:
                raise SystemExit(f"{config['name']}: output size mismatch")
            if "expected" in config:
                expected = np.fromfile(config["expected"], dtype="<f4")
                pass_cpu = bool(np.allclose(ref_cpu, expected, rtol=args.rtol, atol=args.atol))
                pass_cuda = bool(np.allclose(got_cuda, expected, rtol=args.rtol, atol=args.atol))
                max_err = float(max(np.max(np.abs(ref_cpu - expected)),
                                    np.max(np.abs(got_cuda - expected))))
            else:
                pass_cpu = pass_cuda = bool(np.allclose(ref_cpu, got_cuda,
                                                        rtol=args.rtol, atol=args.atol))
                max_err = float(np.max(np.abs(ref_cpu - got_cuda)))
            if not (pass_cpu and pass_cuda):
                raise SystemExit(f"{config['name']}: correctness FAIL")

            auto_backend = None
            pass_auto = None
            if "auto" in programs:
                got_auto, auto_stderr = run(programs["auto"], pcm, td / "auto.bin",
                                            params, auto_env)
                match = DISPATCH_RE.search(auto_stderr)
                auto_backend = match.group(1) if match else "unknown"
                pass_auto = bool(got_auto.size == output_elements and
                                 np.allclose(got_auto, got_cuda if auto_backend == "cuda" else ref_cpu,
                                             rtol=args.rtol, atol=args.atol))
                if "expected" in config:
                    pass_auto = bool(pass_auto and np.allclose(
                        got_auto, expected, rtol=args.rtol, atol=args.atol))
                if not pass_auto:
                    raise SystemExit(f"{config['name']}: auto correctness FAIL")

            names = list(programs)
            times = {name: [] for name in names}
            for repeat in range(args.runs):
                order = names if repeat % 2 == 0 else list(reversed(names))
                for name in order:
                    elapsed, stderr = timed(programs[name], pcm, td / f"{name}-timed.bin",
                                            params, auto_env if name == "auto" else None)
                    times[name].append(elapsed)
                    if name == "auto":
                        match = DISPATCH_RE.search(stderr)
                        if match:
                            auto_backend = match.group(1)

            row = {
                "name": config["name"], "params": params, "n_frames": n_frames,
                "output_elements": output_elements,
                "input_bytes": pcm.stat().st_size,
                "times": times,
                "cpu_min": min(times["cpu"]),
                "cpu_median": statistics.median(times["cpu"]),
                "cuda_min": min(times["cuda"]),
                "cuda_median": statistics.median(times["cuda"]),
                "cpu_pass": pass_cpu, "cuda_pass": pass_cuda,
                "max_abs_err": max_err,
            }
            row["speedup_cpu_over_cuda"] = row["cuda_median"] / row["cpu_median"]
            row["winner"] = "cpu" if row["cpu_median"] < row["cuda_median"] else "cuda"
            if "auto" in times:
                row["auto_min"] = min(times["auto"])
                row["auto_median"] = statistics.median(times["auto"])
                row["auto_backend"] = auto_backend
                row["auto_pass"] = pass_auto
                oracle = min(row["cpu_median"], row["cuda_median"])
                row["auto_regret"] = row["auto_median"] / oracle - 1.0
                row["always_cpu_regret"] = row["cpu_median"] / oracle - 1.0
                row["always_cuda_regret"] = row["cuda_median"] / oracle - 1.0
            results.append(row)
            line = (f"{config['name']}: frames={n_frames} output={output_elements} "
                    f"cpu={row['cpu_median']:.4f}s cuda={row['cuda_median']:.4f}s "
                    f"winner={row['winner']} speedup={row['speedup_cpu_over_cuda']:.3f} "
                    f"PASS cpu/cuda={pass_cpu}/{pass_cuda}")
            if "auto" in times:
                line += (f" auto={row['auto_median']:.4f}s backend={auto_backend} "
                         f"regret={row['auto_regret']:+.1%} PASS={pass_auto}")
            print(line, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
