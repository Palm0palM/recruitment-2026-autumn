#!/usr/bin/env python3
"""Build a separate FFTW phase-profile executable; never time it as the score.

The protected main.cpp is read only: timed call wrappers are generated in tmp/.
This instruments the SAME production FFTW kernel, not an alternative algorithm.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases
from profile import replace_once


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("c2c", "r2c"), default="r2c")
    ap.add_argument("--planner", choices=("estimate", "measure"), default="estimate")
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    work = ROOT / "tmp/fftw-profile" / f"{args.mode}-{args.planner}"
    work.mkdir(parents=True, exist_ok=True)
    driver = '#include <chrono>\n' + (ROOT / "src/main.cpp").read_text()
    for call, name in [("read_pcm(a.input_pcm, y);", "read"),
                       ("compute_melspectrogram(y, a.sr, a.n_fft, a.hop_length, a.n_mels,\n                               a.f_min, a.f_max, output);", "compute"),
                       ("write_output(a.output_bin, output, a.n_mels, n_frames);", "write")]:
        driver = replace_once(driver, call, "{ const auto t = std::chrono::steady_clock::now(); " + call +
            '\nstd::cerr << "' + name + '_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count() << "\\n"; }')
    (work / "main.cpp").write_text(driver)
    binary = work / "melspec-profile"
    cmd = ["g++", "-std=c++17", "-O3", "-march=native", "-pthread", "-g", "-DMELSPEC_PROFILE=1",
           f"-DMELSPEC_FFTW_R2C={int(args.mode == 'r2c')}", f"-DMELSPEC_FFTW_MEASURE={int(args.planner == 'measure')}",
           "-I" + str(ROOT / "include"), str(work / "main.cpp"), str(ROOT / "src/melspec.cpp"), "-o", str(binary)]
    if os.environ.get("FFTW_PREFIX"):
        prefix = Path(os.environ["FFTW_PREFIX"])
        cmd += ["-I" + str(prefix / "include"), str(prefix / "lib/libfftw3.a"), "-lm"]
    else:
        cmd += ["-lfftw3"]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if args.build_only:
        return
    data = find_data_dir()
    selected = [c for c in load_official_cases() if c["name"] in args.cases]
    if len(selected) != len(args.cases):
        ap.error("unknown case")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", str(work))) as td:
        out = Path(td) / "out.bin"
        for case in selected:
            flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
            print(f"== {args.mode}-{args.planner} {case['name']} ==", flush=True)
            time_tool = shutil.which("time")
            command = ([time_tool, "-v"] if time_tool else []) + [str(binary), str(data / "audio.pcm"), str(out), *flags]
            start = time.perf_counter()
            result = subprocess.run(command,
                                    check=True, capture_output=True, text=True)
            print(f"diagnostic_process_wall_s={time.perf_counter() - start:.6f}")
            if not time_tool:
                print("GNU time unavailable: RSS/user/system statistics NOT collected")
            print(result.stderr, flush=True)
            totals = dict.fromkeys(("pack_s", "fft_s", "power_s", "mel_s"), 0.0)
            for line in result.stderr.splitlines():
                if line.startswith("worker="):
                    row = dict(pair.split("=") for pair in line.split())
                    for k in totals:
                        totals[k] += float(row[k])
            total = sum(totals.values())
            print(json.dumps(dict(case=case["name"], thread_wall_sums=totals,
                                  percentages={k: 100 * v / total for k, v in totals.items()})), flush=True)
            got = np.fromfile(out, dtype="<f4")
            ref = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
            if got.shape != ref.shape or not np.allclose(got, ref, rtol=1e-3, atol=1e-4):
                raise SystemExit("Profile output FAIL")
            print("Profile output PASS", flush=True)


if __name__ == "__main__":
    main()
