#!/usr/bin/env python3
"""Check thread override, affinity limits and partition-dependent output."""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("program", type=Path)
    args = ap.parse_args()
    program = str(args.program.resolve())
    available = len(os.sched_getaffinity(0))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "tmp")) as td:
        td = Path(td)
        pcm, out = td / "input.pcm", td / "output.bin"
        frames = 17003
        np.random.default_rng(64).integers(-32768, 32768, 512 + (frames-1)*13,
                                         dtype=np.int16).astype("<i2").tofile(pcm)
        cmd = [program, str(pcm), str(out), "--sr", "44100", "--n_fft", "512",
               "--hop_length", "13", "--n_mels", "129", "--f_min", "317", "--f_max", "17003"]
        ref = None
        records = []
        for requested in (1, 16, 32, 64):
            env = dict(os.environ, MELSPEC_BACKEND="cpu", MELSPEC_DISPATCH_LOG="1",
                       MELSPEC_CPU_THREADS=str(requested))
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            actual = int(re.search(r"cpu_threads=(\d+)", proc.stderr)[1])
            assert actual == min(requested, available, frames//256), (requested, actual)
            got = np.fromfile(out, dtype="<f4")
            assert got.size == frames*129 and np.isfinite(got).all()
            if ref is None:
                ref = got
            np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)
            records.append(dict(requested=requested, actual=actual,
                                max_abs=float(np.max(np.abs(got-ref))), passed=True))
        for value in ("0", "-1", "1025", "64bad"):
            env = dict(os.environ, MELSPEC_BACKEND="cpu", MELSPEC_CPU_THREADS=value)
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            assert proc.returncode != 0 and "MELSPEC_CPU_THREADS" in proc.stderr
        print(json.dumps(dict(affinity=sorted(os.sched_getaffinity(0)), checks=records,
                              invalid_overrides_rejected=True), indent=2))


if __name__ == "__main__":
    main()
