#!/usr/bin/env python3
"""Deterministic boundary/random checks for library-first candidates.

Reference is the provided melspec.py for n_fft >= 2. n_fft=1 follows the problem
formula (w[0]=0), so zero output is asserted explicitly. Invalid short inputs
must fail safely instead of invoking undefined C++ arithmetic.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from bench import load_official_cases  # noqa: E402

REFERENCE = ROOT / "melspec.py"


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--programs", nargs="+", required=True,
                    help="name=command; {pcm}/{out} placeholders")
    args = ap.parse_args()
    programs = {}
    for item in args.programs:
        name, template = item.split("=", 1)
        programs[name] = template
    rng = np.random.default_rng(20261030)
    params = []
    for n in (1, 2, 8, 32, 2048):
        params.append((f"n{n}", dict(sr=96000, n_fft=n, hop_length=127,
                                     n_mels=16, f_min=0.0, f_max=48000.0),
                       rng.integers(-32768, 32768, 2 * n + 127, dtype=np.int16)))
    for official in load_official_cases():
        params.append((official["name"], official["params"],
                       rng.integers(-32768, 32768, 96000, dtype=np.int16)))
    for kind in ("zero", "max", "impulse"):
        pcm = rng.integers(-32768, 32768, 192000, dtype=np.int16)
        if kind == "zero":
            pcm[:] = 0
        elif kind == "max":
            pcm[:] = 32767
        else:
            pcm[:] = 0
            pcm[[0, 8191, 16383, -1]] = [32767, -32768, 32767, -32768]
        params.append((kind, dict(sr=96000, n_fft=8192, hop_length=128,
                                  n_mels=256, f_min=0.0, f_max=22050.0), pcm))

    scratch = ROOT / "tmp/check"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        td = Path(td)
        reference, out = td / "ref.bin", td / "out.bin"
        counts = dict.fromkeys(programs, 0)
        maxima = dict.fromkeys(programs, 0.0)
        for case_name, p, pcm in params:
            src = td / "input.pcm"
            pcm.astype("<i2").tofile(src)
            flags = [s for k, v in p.items() for s in (f"--{k}", str(v))]
            if p["n_fft"] == 1:
                n_frames = 1 + (len(pcm) - 1) // p["hop_length"]
                ref = np.zeros((p["n_mels"], n_frames), dtype=np.float32)
            else:
                result = run([sys.executable, "-B", str(REFERENCE), str(src), str(reference), *flags])
                if result.returncode != 0:
                    raise SystemExit(f"reference failed {case_name}: {result.stderr.decode()}")
                n_frames = 1 + (len(pcm) - p["n_fft"]) // p["hop_length"]
                ref = np.fromfile(reference, dtype="<f4").reshape(p["n_mels"], n_frames)
            for name, template in programs.items():
                command = template.replace("{pcm}", str(src)).replace("{out}", str(out)).split()
                command += flags
                result = run(command)
                if result.returncode != 0:
                    raise SystemExit(f"FAIL {name} {case_name}: cmd={command} stderr={result.stderr.decode()}")
                got = np.fromfile(out, dtype="<f4")
                if got.size != ref.size:
                    raise SystemExit(f"FAIL shape {name} {case_name}: {got.shape} vs {ref.size}")
                got = got.reshape(ref.shape)
                err = float(np.max(np.abs(got - ref)))
                if not np.allclose(got, ref, rtol=1e-3, atol=1e-4):
                    raise SystemExit(f"FAIL values {name} {case_name}: max_abs_err={err}")
                maxima[name] = max(maxima[name], err)
                counts[name] += 1
        for name, template in programs.items():
            for n in (0, 1, 2047):
                (td / "short.pcm").write_bytes(np.zeros(n, dtype="<i2").tobytes())
                command = template.replace("{pcm}", str(td / "short.pcm")).replace("{out}", str(out)).split()
                result = run(command)
                if result.returncode != 1:
                    raise SystemExit(f"FAIL short-input rejection {name} N={n} rc={result.returncode}")
            print(f"{name}: {counts[name]} valid cases PASS, max_abs_err={maxima[name]:.8g}, short inputs rejected")


if __name__ == "__main__":
    main()
