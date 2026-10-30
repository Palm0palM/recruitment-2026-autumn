#!/usr/bin/env python3
"""Generalization evidence: independent dimensions and random PCM, not official cases."""
import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    programs = [str(Path(p).resolve()) for p in (args.reference, args.candidate)]
    rng = np.random.default_rng(310202631)
    rows = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", ROOT / "tmp")) as td:
        td = Path(td)
        for nfft, hop, mels, frames in ((4096, 31, 73, 20013), (8192, 257, 91, 12007),
                                        (2048, 509, 43, 17009), (1024, 1027, 29, 12001)):
            params = dict(sr=44100, n_fft=nfft, hop_length=hop, n_mels=mels,
                          f_min=137, f_max=18317)
            flags = [s for k, v in params.items() for s in (f"--{k}", str(v))]
            pcm = td / "synth.pcm"
            rng.integers(-32768, 32768, nfft + (frames - 1) * hop, dtype="<i2").tofile(pcm)
            outputs = [td / "ref.bin", td / "candidate.bin"]
            commands = [[p, str(pcm), str(out), *flags] for p, out in zip(programs, outputs)]
            for command in commands:
                subprocess.run(command, check=True, capture_output=True)
            ref = np.fromfile(outputs[0], dtype="<f4")
            times = [[], []]
            for iteration in range(8):
                for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    start = time.perf_counter()
                    subprocess.run(commands[index], check=True, capture_output=True)
                    times[index].append(time.perf_counter() - start)
                    got = np.fromfile(outputs[index], dtype="<f4")
                    assert np.array_equal(got.view("u4"), ref.view("u4")), params
            row = dict(params=params, frames=frames, times_s=times,
                       medians_s=[statistics.median(t) for t in times], exact=True)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
