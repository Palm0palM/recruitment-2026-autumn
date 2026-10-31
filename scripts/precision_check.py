#!/usr/bin/env python3
"""比较候选与 FP64 参考：官方四组误差/tolerance 利用率 + 合成边界输入。"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases


def metrics(got, ref, rtol, atol):
    error = np.abs(got - ref)
    tolerance = atol + rtol * np.abs(ref)
    ratio = error / tolerance
    return {
        "max_abs": float(error.max()),
        "rms": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "max_tolerance_ratio": float(ratio.max()),
        "max_rel_nonzero": float((error[np.abs(ref) > 1e-6] /
                                  np.abs(ref[np.abs(ref) > 1e-6])).max())
        if np.any(np.abs(ref) > 1e-6) else 0.0,
        "allclose": bool(np.all(error <= tolerance)),
    }


def run(program, pcm, output, params):
    flags = [s for key, value in params.items() for s in (f"--{key}", str(value))]
    subprocess.run([program, str(pcm), str(output), *flags], check=True)
    return np.fromfile(output, dtype="<f4")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()
    data = find_data_dir()
    cases = load_official_cases()
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
        td = Path(td)
        for case in cases:
            out_ref, out_got = td / "ref.bin", td / "got.bin"
            got = run(args.candidate, data / "audio.pcm", out_got, case["params"])
            ref = run(args.reference, data / "audio.pcm", out_ref, case["params"])
            print(case["name"], metrics(got, ref, args.rtol, args.atol), flush=True)
        params = cases[0]["params"]
        n = 192000
        sr = params["sr"]
        rng = np.random.default_rng(20261030)
        patterns = {
            "zero": np.zeros(n, dtype=np.int16),
            "impulse": np.eye(1, n, n // 8, dtype=np.int16).ravel() * 32767,
            "alternating": np.where(np.arange(n) % 2, 32767, -32768).astype(np.int16),
            "sine_1k": (np.sin(2 * np.pi * 1000 * np.arange(n) / sr) * 30000).astype(np.int16),
            "sine_near_nyquist": (np.sin(2 * np.pi * (sr / 2 - 1000) * np.arange(n) / sr)
                                  * 30000).astype(np.int16),
            "dc_step": np.where(np.arange(n) > n // 2, 32767, -32768).astype(np.int16),
            "random": rng.integers(-32768, 32768, n, dtype=np.int16),
            "two_frames": np.pad(np.array([32767, -32768], dtype=np.int16), (0, n - 2)),
        }
        for name, pcm in patterns.items():
            path = td / f"{name}.pcm"
            pcm.astype("<i2").tofile(path)
            got = run(args.candidate, path, td / "got.bin", params)
            ref = run(args.reference, path, td / "ref.bin", params)
            print(name, metrics(got, ref, args.rtol, args.atol), flush=True)


if __name__ == "__main__":
    main()
