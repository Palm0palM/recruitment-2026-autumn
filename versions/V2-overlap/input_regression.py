#!/usr/bin/env python3
"""随机/低幅度/1024 刷新边界的 V1 与三个 Overlap 实现对照。"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench import ROOT

PROGRAMS = {
    "v1": "tmp/binaries/melspec-v1",
    "full": "tmp/binaries/melspec-overlap-full",
    "half": "tmp/binaries/melspec-overlap-half",
    "half_delta": "tmp/binaries/melspec-overlap-half-delta",
}


def main():
    rng = np.random.default_rng(20261030)
    records = []
    scratch = ROOT / "tmp/scratch"
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        td = Path(td)
        for n in (8192, 16384):
            frames = 1025  # 覆盖第 0 帧精确建立与第 1024 帧精确刷新
            samples = n + (frames - 1) * 128
            signals = {
                "random": rng.integers(-32768, 32768, samples, dtype=np.int16),
                "low_amplitude": rng.integers(-1, 2, samples, dtype=np.int16),
            }
            for signal_name, pcm in signals.items():
                src = td / f"{n}-{signal_name}.pcm"
                pcm.astype("<i2").tofile(src)
                outputs = {}
                for name, program in PROGRAMS.items():
                    out = td / f"{n}-{signal_name}-{name}.bin"
                    subprocess.run([ROOT / program, src, out, "--sr", "96000", "--n_fft", str(n),
                                    "--hop_length", "128", "--n_mels", "64", "--f_min", "0",
                                    "--f_max", "22050"], check=True)
                    outputs[name] = np.fromfile(out, dtype="<f4")
                ref = outputs["v1"]
                for name in ("full", "half", "half_delta"):
                    got = outputs[name]
                    passed = got.shape == ref.shape and bool(np.allclose(got, ref, rtol=1e-3, atol=1e-4))
                    record = dict(n_fft=n, signal=signal_name, frames=frames, variant=name,
                                  passed=passed,
                                  max_abs_diff=float(np.max(np.abs(got - ref))),
                                  identical=bool(np.array_equal(got, ref)))
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    if not passed:
                        raise SystemExit("regression FAIL")
    out_path = ROOT / "versions/V2-overlap/raw/input-regression.json"
    out_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"PASS: {len(records)} comparisons; wrote {out_path}")


if __name__ == "__main__":
    main()
