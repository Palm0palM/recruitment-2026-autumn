#!/usr/bin/env python3
"""Serial envelope/fallback checks; parent-only, no performance measurements."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def checksum(payload):
    value = 14695981039346656037
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--program", required=True, type=Path)
    ap.add_argument("--wisdom-dir", required=True, type=Path)
    ap.add_argument("--library-id", required=True)
    ap.add_argument("--precision", choices=("double", "float"), required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    env_prefix = "MELSPEC_FFTWF_" if args.precision == "float" else "MELSPEC_FFTW_"
    original = (args.wisdom_dir / "r2c-8192.wisdom").read_bytes()
    header, payload = original.split(b"\n\n", 1)
    lines = header.decode().splitlines()
    metadata = dict(line.split("=", 1) for line in lines[1:])

    def changed(key, value, data=payload):
        fields = dict(metadata)
        fields[key] = value
        return (lines[0] + "\n" + "\n".join(f"{k}={v}" for k, v in fields.items()) + "\n\n").encode() + data

    def new_payload(data):
        fields = dict(metadata, payload_bytes=str(len(data)), payload_fnv1a64=checksum(data))
        return (lines[0] + "\n" + "\n".join(f"{k}={v}" for k, v in fields.items()) + "\n\n").encode() + data

    cases = [("loaded", original, args.library_id, "status=loaded")]
    for key, value, diagnostic in (
        ("fftw_version", "wrong", "fftw-version-mismatch"),
        ("fftw_cc", "wrong", "fftw-build-flags-mismatch"),
        ("precision", "wrong", "precision-mismatch"),
        ("cpu", "wrong", "cpu-mismatch"),
        ("size", "16384", "transform-size-mismatch"),
        ("transform", "wrong", "transform-layout-mismatch"),
        ("placement", "wrong", "placement-layout-mismatch"),
        ("planner", "wrong", "generator-planner-invalid"),
        ("input_alignment", str(int(metadata["input_alignment"]) + 1), "alignment-layout-mismatch"),
        ("output_alignment", str(int(metadata["output_alignment"]) + 1), "alignment-layout-mismatch"),
        ("payload_bytes", "1", "payload-length-mismatch"),
        ("payload_fnv1a64", "0", "payload-checksum-mismatch"),
    ):
        cases.append((key, changed(key, value), args.library_id, diagnostic))
    cases.extend([
        ("missing-file", None, args.library_id, "file-missing-or-unreadable"),
        ("missing-library-id", original, None, "library-id-required"),
        ("wrong-library-id", original, "wrong", "library-id-mismatch"),
        ("truncated", original[:40], args.library_id, "envelope-header-missing"),
        ("invalid-fftw-payload", new_payload(b"not-fftw-wisdom\n"), args.library_id, "fftw-import-rejected-payload"),
    ])
    tiny_payload = (args.wisdom_dir / "r2c-1.wisdom").read_bytes().split(b"\n\n", 1)[1]
    cases.append(("import-without-applicable-plan", new_payload(tiny_payload), args.library_id, "no-applicable-imported-plan"))

    scratch = Path(os.environ.get("TMPDIR", "tmp/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(dir=scratch, prefix="wisdom-check-") as td:
        td = Path(td)
        src, output, wisdom = td / "input.pcm", td / "out.f32", td / "test.wisdom"
        np.random.default_rng(20261030).integers(-32768, 32768, 32768, dtype=np.int16).astype("<i2").tofile(src)
        flags = ["--sr", "96000", "--n_fft", "8192", "--hop_length", "128", "--n_mels", "256", "--f_max", "22050"]
        subprocess.run([str(args.reference.resolve()), str(src), str(output), *flags], check=True)
        reference = np.fromfile(output, dtype="<f4")
        for name, data, identity, diagnostic in cases:
            if wisdom.exists():
                wisdom.unlink()
            if data is not None:
                wisdom.write_bytes(data)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("MELSPEC_FFTW_", "MELSPEC_FFTWF_"))}
            env[env_prefix + "WISDOM"] = str(wisdom)
            if identity is not None:
                env[env_prefix + "LIBRARY_ID"] = identity
            if output.exists():
                output.unlink()
            run = subprocess.run([str(args.program.resolve()), str(src), str(output), *flags], env=env, capture_output=True, text=True)
            got = np.fromfile(output, dtype="<f4") if output.exists() else np.array([])
            passed = (run.returncode == 0 and diagnostic in run.stderr and got.shape == reference.shape
                      and np.isfinite(got).all() and np.allclose(got, reference, rtol=1e-3, atol=1e-4))
            rows.append(dict(case=name, passed=bool(passed), expected_diagnostic=diagnostic, stderr=run.stderr.strip(), exit_code=run.returncode))
            print(json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)


if __name__ == "__main__":
    main()
