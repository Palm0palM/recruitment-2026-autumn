#!/usr/bin/env python3
"""Compare FFTW candidates to a locally built V1 on deterministic boundary PCM.

The protected CLI/validator remain unchanged. Short/invalid inputs are expected
to fail safely; they are not passed to V1, whose short-input check is unsafe.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bench import ROOT, load_official_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="tmp/bin/v1")
    ap.add_argument("programs", nargs="+")
    args = ap.parse_args()
    rng = np.random.default_rng(20261030)
    cases = []
    for n in (1, 2, 8, 256, 2048):
        for extra in (0, 126, 127):
            p = dict(sr=96000, n_fft=n, hop_length=127, n_mels=16, f_min=0, f_max=48000)
            cases.append((f"random-n{n}-extra{extra}", rng.integers(-32768, 32768, n + extra, dtype=np.int16), p))
    for kind in ("zero", "max", "min", "nyquist", "impulse", "random"):
        pcm = rng.integers(-32768, 32768, 192000, dtype=np.int16)
        if kind == "zero": pcm[:] = 0
        if kind == "max": pcm[:] = 32767
        if kind == "min": pcm[:] = -32768
        if kind == "nyquist": pcm[::2], pcm[1::2] = 32767, -32768
        if kind == "impulse":
            pcm[:] = 0
            pcm[0], pcm[8191], pcm[16383], pcm[-1] = 32767, -32768, 32767, -32768
        for case in load_official_cases():
            cases.append((f"{kind}-{case['name']}", pcm, case["params"]))
    scratch = ROOT / "tmp/check"
    scratch.mkdir(exist_ok=True)
    counts = dict.fromkeys(args.programs, 0)
    maxima = dict.fromkeys(args.programs, 0.0)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        td = Path(td)
        for name, pcm, p in cases:
            src, dst = td / "input.pcm", td / "out.bin"
            pcm.astype("<i2").tofile(src)
            flags = [s for k, v in p.items() for s in (f"--{k}", str(v))]
            subprocess.run([args.reference, str(src), str(dst), *flags], check=True)
            ref = np.fromfile(dst, dtype="<f4")
            for program in args.programs:
                subprocess.run([program, str(src), str(dst), *flags], check=True)
                got = np.fromfile(dst, dtype="<f4")
                error = float(np.max(np.abs(got - ref)))
                if got.shape != ref.shape or not np.allclose(got, ref, rtol=1e-3, atol=1e-4):
                    raise SystemExit(f"FAIL {program} {name} max_abs_err={error}")
                maxima[program] = max(maxima[program], error)
                counts[program] += 1
        for program in args.programs:
            for length in (0, 1, 2047):
                np.zeros(length, dtype="<i2").tofile(td / "short.pcm")
                result = subprocess.run([program, str(td / "short.pcm"), str(td / "out.bin")], capture_output=True)
                if result.returncode != 1:
                    raise SystemExit(f"FAIL short-input rejection: {program} N={length}")
            print(json.dumps(dict(program=program, passed_valid=counts[program],
                                  max_abs_err_vs_v1=maxima[program], rejected_short=3)), flush=True)


if __name__ == "__main__":
    main()
