#!/usr/bin/env python3
"""Deployment acceptance for the frozen float+PATIENT candidate.

Lightweight and serial: for every official case, run once with the exact wisdom
environment (must report status=loaded, never fallback), then once with a wrong
library identity (must report fallback) and require identical output. Both runs
are compared with the original rtol/atol against the task's expected arrays.
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bench import find_data_dir, load_official_cases


def run(program, src, dst, flags, env):
    if dst.exists():
        dst.unlink()
    proc = subprocess.run([str(program), str(src), str(dst), *flags],
                          capture_output=True, text=True, env=env)
    return proc, (np.fromfile(dst, dtype="<f4") if dst.exists() else np.array([]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--wisdom-dir", type=Path, required=True)
    ap.add_argument("--library-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    program = args.program.resolve()
    data = find_data_dir()
    base = dict(os.environ)
    for key in ("MELSPEC_FFTWF_WISDOM", "MELSPEC_FFTWF_WISDOM_DIRECTORY", "MELSPEC_FFTWF_LIBRARY_ID"):
        base.pop(key, None)
    good = dict(base, MELSPEC_FFTWF_LIBRARY_ID=args.library_id,
                MELSPEC_FFTWF_WISDOM_DIRECTORY=str(args.wisdom_dir.resolve()))
    bad = dict(base, MELSPEC_FFTWF_LIBRARY_ID="0" * 64,
               MELSPEC_FFTWF_WISDOM_DIRECTORY=str(args.wisdom_dir.resolve()))

    scratch = Path(os.environ.get("TMPDIR", "tmp/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(dir=scratch, prefix="deploy-check-") as td:
        td = Path(td)
        out_loaded, out_fallback = td / "loaded.f32", td / "fallback.f32"
        for case in load_official_cases():
            flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
            expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
            loaded_run, loaded_out = run(program, data / "audio.pcm", out_loaded, flags, good)
            fallback_run, fallback_out = run(program, data / "audio.pcm", out_fallback, flags, bad)
            ok = (loaded_run.returncode == 0 and fallback_run.returncode == 0
                  and "status=loaded" in loaded_run.stderr
                  and "status=fallback" not in loaded_run.stderr
                  and "status=fallback" in fallback_run.stderr
                  and "status=loaded" not in fallback_run.stderr
                  and loaded_out.shape == expected.shape
                  and fallback_out.shape == expected.shape
                  and np.allclose(loaded_out, expected, rtol=1e-3, atol=1e-4)
                  and np.allclose(fallback_out, expected, rtol=1e-3, atol=1e-4)
                  and np.allclose(loaded_out, fallback_out, rtol=1e-3, atol=1e-4))
            rows.append(dict(
                case=case["name"], passed=bool(ok),
                loaded_diagnostic=loaded_run.stderr.strip(),
                fallback_diagnostic=fallback_run.stderr.strip(),
                loaded_exit=loaded_run.returncode, fallback_exit=fallback_run.returncode,
                max_abs_vs_expected=float(np.max(np.abs(loaded_out - expected))) if loaded_out.shape == expected.shape else None,
                loaded_matches_fallback=bool(np.allclose(loaded_out, fallback_out, rtol=1e-3, atol=1e-4)) if loaded_out.shape == fallback_out.shape else False))
            print(json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()
