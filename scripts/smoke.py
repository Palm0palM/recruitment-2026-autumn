#!/usr/bin/env python3
"""本地小规模回归：随机 PCM、官方四组、CLI 默认值和小 FFT 边界。"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

from bench import ROOT, load_official_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", nargs="?", default="./melspec")
    args = parser.parse_args()
    scratch = ROOT / "tmp" / "smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    pcm = np.random.default_rng(2026).integers(-32768, 32768, 192000, dtype=np.int16)
    y = pcm.astype(np.float32) / 32768.0
    cases = [(c["name"], c["params"]) for c in load_official_cases()]
    defaults = dict(sr=44100, n_fft=2048, hop_length=512, n_mels=128, f_min=0, f_max=22050)
    cases.append(("cli_defaults", defaults))
    for n in (1, 2, 8):
        cases.append((f"fft_{n}", dict(sr=96000, n_fft=n, hop_length=128,
                                       n_mels=4, f_min=0, f_max=48000)))
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        src, dst = Path(td) / "random.pcm", Path(td) / "out.bin"
        pcm.astype("<i2").tofile(src)
        for name, p in cases:
            flags = [] if name == "cli_defaults" else [s for k, v in p.items() for s in (f"--{k}", str(v))]
            subprocess.run([args.program, str(src), str(dst), *flags], check=True)
            frames = 1 + (len(pcm) - p["n_fft"]) // p["hop_length"]
            got = np.fromfile(dst, dtype="<f4").reshape(p["n_mels"], frames)
            # scipy 的单点 Hann 特例为 1，而题目公式 w[0]=0；单独检查题意。
            if p["n_fft"] == 1:
                ref = np.zeros_like(got)
            else:
                ref = librosa.feature.melspectrogram(y=y, sr=p["sr"], n_fft=p["n_fft"],
                    hop_length=p["hop_length"], n_mels=p["n_mels"], fmin=p["f_min"],
                    fmax=p["f_max"], center=False, htk=True)
            error = float(np.max(np.abs(got - ref)))
            passed = np.allclose(got, ref, rtol=1e-3, atol=1e-4)
            print(f"{name}: {'PASS' if passed else 'FAIL'} shape={got.shape} max_abs_err={error:.8g}", flush=True)
            if not passed:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
