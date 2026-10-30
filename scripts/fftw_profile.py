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

PHASE_KEYS = {"read_s", "compute_s", "write_s", "compute_setup_s", "r2c_plan_s", "c2c_plan_s"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("c2c", "r2c"), default="r2c")
    ap.add_argument("--planner", choices=("estimate", "measure"), default="estimate")
    ap.add_argument("--precision", choices=("double", "float"), default="double")
    ap.add_argument("--power", choices=("double", "float"), default="float")
    ap.add_argument("--wisdom", action="store_true")
    ap.add_argument("--cases", nargs="+", default=["default", "small_mels", "large_fft", "narrow_band"])
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat each case; report every run")
    ap.add_argument("--out-json", type=Path, help="Structured per-run phase/worker results")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be positive")
    if args.precision == "float" and args.mode != "r2c":
        ap.error("float requires R2C")
    if args.wisdom and (args.precision != "float" or args.planner != "estimate"):
        ap.error("wisdom requires float ESTIMATE (offline rigor is selected by environment)")
    tag = f"{args.mode}-{args.planner}-{args.precision}-{args.power}-wisdom{int(args.wisdom)}"
    work = ROOT / "tmp/fftw-profile" / tag
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
            f"-DMELSPEC_FFTW_FLOAT={int(args.precision == 'float')}", f"-DMELSPEC_FFTWF_DOUBLE_POWER={int(args.power == 'double')}",
            "-I" + str(ROOT / "include"), str(work / "main.cpp"), str(ROOT / "src/melspec.cpp"), "-o", str(binary)]
    if args.wisdom:
        cmd += ["-DMELSPEC_FFTWF_WISDOM=1", str(ROOT / "src/fftw_float_wisdom.cpp")]
    library = "fftw3f" if args.precision == "float" else "fftw3"
    if os.environ.get("FFTW_PREFIX"):
        prefix = Path(os.environ["FFTW_PREFIX"])
        cmd += ["-I" + str(prefix / "include"), str(prefix / f"lib/lib{library}.a"), "-lm"]
    else:
        cmd += [f"-l{library}"]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if args.build_only:
        return
    data = find_data_dir()
    selected = [c for c in load_official_cases() if c["name"] in args.cases]
    if len(selected) != len(args.cases):
        ap.error("unknown case")
    records = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", str(work))) as td:
        out = Path(td) / "out.bin"
        for repeat in range(args.repeat):
            for case in selected:
                flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
                print(f"== {tag} {case['name']} run={repeat} ==", flush=True)
                time_tool = shutil.which("time")
                command = ([time_tool, "-v"] if time_tool else []) + [str(binary), str(data / "audio.pcm"), str(out), *flags]
                start = time.perf_counter()
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                process_wall = time.perf_counter() - start
                print(f"diagnostic_process_wall_s={process_wall:.6f}")
                if not time_tool:
                    print("GNU time unavailable: RSS/user/system statistics NOT collected")
                print(result.stderr, flush=True)
                if args.wisdom and ("status=loaded" not in result.stderr or "status=fallback" in result.stderr):
                    raise SystemExit("Profile expected loaded wisdom; do not attribute a fallback to the candidate")
                totals = dict.fromkeys(("pack_s", "fft_s", "power_s", "mel_s"), 0.0)
                for line in result.stderr.splitlines():
                    if line.startswith("worker="):
                        row = dict(pair.split("=") for pair in line.split())
                        for k in totals:
                            totals[k] += float(row[k])
                total = sum(totals.values())
                print(json.dumps(dict(case=case["name"], thread_wall_sums=totals,
                                      percentages={k: 100 * v / total for k, v in totals.items()})), flush=True)
                phases = {}
                for line in result.stderr.splitlines():
                    key, sep, value = line.partition("=")
                    if sep and key in PHASE_KEYS:
                        phases[key] = float(value)
                got = np.fromfile(out, dtype="<f4")
                ref = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
                if got.shape != ref.shape or not np.allclose(got, ref, rtol=1e-3, atol=1e-4):
                    raise SystemExit("Profile output FAIL")
                print("Profile output PASS", flush=True)
                records.append(dict(
                    repeat=repeat, case=case["name"], process_wall_s=process_wall,
                    phases=phases, worker_stage_sums=totals,
                    worker_stage_share_pct={k: (100 * v / total if total else 0.0) for k, v in totals.items()},
                    effective_parallelism=(total / phases["compute_s"] if phases.get("compute_s") else None),
                    wisdom_loaded="status=loaded" in result.stderr,
                    output_passed=True))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
