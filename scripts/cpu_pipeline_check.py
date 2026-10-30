#!/usr/bin/env python3
"""Independent synthetic CLI checks: arbitrary hops/bands, B16 tails and workers.

Unlike precision_check.py, every failed comparison raises an error.
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

from bench import ROOT


def run(program, pcm, output, params):
    flags = [s for k, v in params.items() for s in (f"--{k}", str(v))]
    subprocess.run([str(Path(program).resolve()), str(pcm), str(output), *flags],
                   check=True, capture_output=True)
    return np.fromfile(output, dtype="<f4")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--exact", action="store_true", help="also require bit-identical frozen outputs")
    args = ap.parse_args()
    rng = np.random.default_rng(3102026)
    specs = []
    # Every tail length, odd hops, n_fft=1/2, and tiny/empty Mel supports.
    for tail in range(16):
        specs.append((512, 37, 32 + tail, 32000, 19, 173.0, 13111.0, "random"))
    for nfft in (1, 2, 4, 8, 32, 128, 1024, 4096, 8192, 16384):
        specs.append((nfft, max(1, nfft // 11), 33, 48000, 23, 0.0, 24000.0, "random"))
    for _ in range(32):
        nfft = 2 ** int(rng.integers(4, 13))
        hop = int(rng.choice([1, 3, 17, 63, nfft // 2, nfft, nfft + 7]))
        sr = int(rng.choice([16000, 22050, 32000, 48000]))
        lo = float(rng.uniform(0, sr / 3))
        hi = float(rng.uniform(lo + 1, sr / 2))
        specs.append((nfft, hop, int(rng.integers(16, 90)), sr,
                      int(rng.integers(1, 100)), lo, hi, "random"))
    for pattern in ("zero", "impulse", "dc", "alternating", "sine", "step", "low"):
        specs.append((2048, 29, 79, 32000, 31, 0.0, 16000.0, pattern))
    # Ensure all 16 partitions are exercised, not just single-thread tiny cases.
    specs.append((256, 11, 4127, 22050, 27, 110.0, 9876.0, "random"))
    specs.append((1024, 5, 4175, 48000, 7, 233.0, 7777.0, "random"))
    # Multi-MiB output to exercise actual huge-page-backed ranges, independent
    # of official parameters. Also crosses many slab boundaries.
    specs.append((512, 13, 17001, 44100, 129, 317.0, 17003.0, "random"))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", ROOT / "tmp")) as td:
        td = Path(td)
        for index, (nfft, hop, frames, sr, mels, lo, hi, pattern) in enumerate(specs):
            n = nfft + hop * (frames - 1)
            pcm = rng.integers(-32768, 32768, n, dtype=np.int16)
            if pattern == "zero":
                pcm.fill(0)
            elif pattern == "impulse":
                pcm.fill(0)
                pcm[n // 3] = 32767
            elif pattern == "dc":
                pcm.fill(32767)
            elif pattern == "alternating":
                pcm[:] = np.where(np.arange(n) % 2, 32767, -32768)
            elif pattern == "sine":
                pcm[:] = (32760 * np.sin(2 * np.pi * 3741 * np.arange(n) / sr)).astype(np.int16)
            elif pattern == "step":
                pcm[:] = np.where(np.arange(n) < n // 2, -32768, 32767)
            elif pattern == "low":
                pcm[:] = rng.integers(-1, 2, n, dtype=np.int16)
            path = td / "input.pcm"
            pcm.astype("<i2").tofile(path)
            params = dict(sr=sr, n_fft=nfft, hop_length=hop, n_mels=mels, f_min=lo, f_max=hi)
            ref = run(args.reference, path, td / "ref.bin", params)
            got = run(args.candidate, path, td / "got.bin", params)
            assert got.size == mels * frames, (index, got.size, mels * frames)
            assert got.shape == ref.shape and np.isfinite(got).all(), index
            np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4, err_msg=str(index))
            if args.exact:
                assert np.array_equal(got.view("u4"), ref.view("u4")), ("not bit exact", index)
            expected = librosa.feature.melspectrogram(
                y=pcm.astype(np.float64) / 32768, sr=sr, n_fft=nfft, hop_length=hop,
                n_mels=mels, fmin=lo, fmax=hi, htk=True, center=False).astype("<f4").ravel()
            if nfft == 1:
                # The problem's explicit periodic Hann formula gives w[0]=0;
                # scipy/librosa use a length-one window special case of 1.
                expected.fill(0)
            np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-4, err_msg=str(index))
            print(json.dumps(dict(case=index, params=params, frames=frames, pattern=pattern,
                                  exact=bool(np.array_equal(got.view("u4"), ref.view("u4"))),
                                  max_abs=float(np.max(np.abs(got - expected))), status="PASS")), flush=True)
    print(f"PASS {len(specs)} independent synthetic cases", flush=True)


if __name__ == "__main__":
    main()
