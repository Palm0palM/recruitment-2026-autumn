#!/usr/bin/env python3
"""Orchestrator-only, serial Linux post-exec VmHWM checks (no GNU time).

Separate from score timings: one full invocation per candidate/case; retain
child mm high-water mark, not inherited ru_maxrss. No worker delegation.
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from bench import ROOT, find_data_dir, load_official_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--programs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    programs = dict(x.split("=", 1) for x in args.programs)
    data = find_data_dir()
    rows = []
    scratch = Path(os.environ.get("TMPDIR", str(ROOT / "tmp/scratch")))
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        output = Path(td) / "out.bin"
        for case in load_official_cases():
            if case["name"] not in ("default", "large_fft"):
                continue
            flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
            for name, program in programs.items():
                executable = str(Path(program).resolve())
                pid = os.posix_spawn(executable, [executable, str(data / "audio.pcm"), str(output), *flags], dict(os.environ))
                # vfork/posix_spawn can inherit a large Python parent's RSS
                # high-water mark across exec in ru_maxrss. Sample the child's
                # post-exec mm's VmHWM instead; retain wait4 only for CPU times.
                peak = 0
                while True:
                    try:
                        if Path(f"/proc/{pid}/exe").resolve() == Path(executable):
                            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                                if line.startswith("VmHWM:"):
                                    peak = max(peak, int(line.split()[1]))
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    finished, status, usage = os.wait4(pid, os.WNOHANG)
                    if finished:
                        break
                    time.sleep(0.005)
                if os.waitstatus_to_exitcode(status) != 0:
                    raise SystemExit(f"Child failed: {name}")
                got = np.fromfile(output, dtype="<f4")
                ref = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
                if got.shape != ref.shape or not np.isfinite(got).all() or not np.allclose(got, ref, rtol=1e-3, atol=1e-4):
                    raise SystemExit(f"Output FAIL: {name}")
                if not peak:
                    raise SystemExit("No post-exec VmHWM sampled; do not report zero RSS")
                row = dict(case=case["name"], program=name, proc_vmhwm_kib=peak,
                           inherited_wait4_maxrss_kib=usage.ru_maxrss,
                           user_s=usage.ru_utime, system_s=usage.ru_stime, passed=True)
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
