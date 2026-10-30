#!/usr/bin/env python3
"""Run a CUDA_PROFILE=1 binary on full official inputs and preserve phase logs."""
import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import find_data_dir, load_official_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("program")
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    args = ap.parse_args()
    cases = [c for c in load_official_cases() if c["name"] in args.cases]
    if len(cases) != len(args.cases):
        ap.error("unknown case")
    data = find_data_dir()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.bin"
        for case in cases:
            flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
            print(f"== cuda profile {case['name']} ==", flush=True)
            start = time.perf_counter()
            result = subprocess.run([args.program, str(data / "audio.pcm"), str(out), *flags],
                                    check=True, capture_output=True, text=True)
            print(result.stderr, end="", flush=True)
            print(f"diagnostic_process_wall_s={time.perf_counter() - start:.6f}", flush=True)
            got = np.fromfile(out, dtype="<f4")
            expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
            if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-3, atol=1e-4):
                raise SystemExit(f"FAIL {case['name']}")
            print(f"profile output PASS: {case['name']}", flush=True)


if __name__ == "__main__":
    main()
