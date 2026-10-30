#!/usr/bin/env python3
"""Local CPU sampling of A2 ESTIMATE without phase instrumentation.

No kernel or PMU assumptions. Save only the SVG/text in versions/, never .data.
The static/system FFTW difference versus cluster must be stated in the report.
"""
import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "tmp/perf-fftw")
    ap.add_argument("--freq", type=int, default=199)
    ap.add_argument("--mmap-pages", type=int, default=256)
    args = ap.parse_args()
    work = args.out.resolve()
    work.mkdir(parents=True, exist_ok=False)
    tools = ROOT / "tmp/tools/FlameGraph"
    os.environ.update(DEBUGINFOD_URLS="", LC_ALL="C")
    binary = work / "melspec-perf"
    cmd = ["g++", "-std=c++17", "-O3", "-march=native", "-pthread", "-g", "-Iinclude",
           "-DMELSPEC_FFTW_R2C=1", "src/main.cpp", "src/melspec.cpp", "-lfftw3", "-o", str(binary)]
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with (work / "environment.txt").open("w") as f:
        for command in (["git", "rev-parse", "HEAD"], ["uname", "-a"], ["lscpu"],
                        ["g++", "--version"], ["perf", "--version"], ["ldd", str(binary)],
                        ["git", "-C", str(tools), "rev-parse", "HEAD"]):
            subprocess.run(command, check=True, stdout=f)
    data = find_data_dir()
    for case in load_official_cases():
        if case["name"] not in ("default", "large_fft"):
            continue
        target = work / case["name"]
        target.mkdir()
        flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", str(work))) as td:
            out = Path(td) / "out.bin"
            cmd = ["perf", "record", "--no-buildid-cache", "-e", "task-clock:u", "-F", str(args.freq),
                   "--call-graph", "dwarf,16384", "-m", str(args.mmap_pages), "-o", str(target / "perf.data"),
                   "--", str(binary), str(data / "audio.pcm"), str(out), *flags]
            with (target / "record.txt").open("w") as f:
                f.write(shlex.join(cmd) + "\n"); f.flush()
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
            got = np.fromfile(out, dtype="<f4")
            ref = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
            assert got.shape == ref.shape and np.allclose(got, ref, rtol=1e-3, atol=1e-4)
            (target / "correctness.json").write_text(json.dumps(dict(passed=True,
                max_abs_err=float(np.max(np.abs(got - ref))))) + "\n")
        for name, flags in (("self", ["--no-children"]), ("inclusive", ["--children", "--inline"])):
            with (target / f"report-{name}.txt").open("w") as f:
                subprocess.run(["perf", "report", "-i", str(target / "perf.data"), "--stdio",
                    "--stdio-color", "never", "--percent-limit", "0.1", "-n", "-g", "none", *flags], check=True, stdout=f)
        if "# Total Lost Samples: 0\n" not in (target / "report-self.txt").read_text():
            raise SystemExit("Lost samples: keep this failed trace, retry at lower --freq or larger --mmap-pages")
        with (target / "stacks.txt").open("w") as f:
            subprocess.run(["perf", "script", "--inline", "-F", "comm,pid,tid,time,event,ip,sym,dso",
                            "-i", str(target / "perf.data")], check=True, stdout=f)
        with (target / "folded.txt").open("w") as f:
            subprocess.run(["perl", str(tools / "stackcollapse-perf.pl"), str(target / "stacks.txt")], check=True, stdout=f)
        with (target / "flamegraph.svg").open("w") as f:
            subprocess.run(["perl", str(tools / "flamegraph.pl"), "--title", f"A2 FFTW R2C ESTIMATE: {case['name']}",
                            "--subtitle", "WSL i7-14650HX / 16 workers / task-clock:u / no phase timers",
                            "--countname", "samples", str(target / "folded.txt")], check=True, stdout=f)
    print(f"Local perf artifacts: {work}", flush=True)


if __name__ == "__main__":
    main()
