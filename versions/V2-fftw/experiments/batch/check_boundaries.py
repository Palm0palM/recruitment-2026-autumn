#!/usr/bin/env python3
"""Serial boundary checks for plan-many candidates; does not run benchmarks.

The reference and every candidate are separate executables built with the same
protected main.cpp. Tests cover partial batches and V1 worker-partition edges.
Actual multi-worker coverage requires sufficient std::thread::hardware_concurrency.
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

BATCHES = (1, 4, 8, 16, 32)
RTOL = 1e-3
ATOL = 1e-4


def invoke(program, src, dst, params, expect_success=True):
    if dst.exists():
        dst.unlink()
    flags = [item for key, value in params.items()
             for item in (f"--{key}", str(value))]
    result = subprocess.run([str(program), str(src), str(dst), *flags],
                            capture_output=True)
    if expect_success and result.returncode != 0:
        raise RuntimeError(
            f"{program} failed for {params}: rc={result.returncode}\n"
            f"stdout={result.stdout.decode(errors='replace')}\n"
            f"stderr={result.stderr.decode(errors='replace')}")
    if not expect_success:
        if result.returncode == 0:
            raise RuntimeError(f"{program} accepted invalid parameters: {params}")
        return None
    return np.fromfile(dst, dtype="<f4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--programs", required=True, nargs="+", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(20261030)
    stats = {str(p): {"cases": 0, "max_abs_err": 0.0, "rejected": 0}
             for p in args.programs}

    # tempfile remains inside this experiment, not in shared data or another worktree.
    scratch = Path(os.environ.get("TMPDIR", "tmp/scratch")).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp = Path(tmp)
        src, reference_out, candidate_out = tmp / "in.pcm", tmp / "ref.bin", tmp / "got.bin"

        for batch in BATCHES:
            frame_counts = {1, batch, batch + 1, 2 * batch, 2 * batch + 1}
            if batch > 1:
                frame_counts.update((batch - 1, 2 * batch - 1))

            # With hardware_concurrency >= W, F=256*W selects W workers under
            # the V1 policy. F-1 crosses the worker-count threshold; F+batch
            # gives nontrivial final partitions within workers.
            for workers in (2, 4, 8, 16):
                base = 256 * workers
                frame_counts.update((base - 1, base, base + 1, base + batch))

            for frames in sorted(frame_counts):
                # hop != 1 also checks that frame-to-input indexing is not
                # accidentally expressed in batch or output units.
                n_fft, hop = 8, 3
                sample_count = n_fft + (frames - 1) * hop
                pcm = rng.integers(-32768, 32768, sample_count, dtype=np.int16)
                pcm.astype("<i2").tofile(src)
                params = dict(sr=96000, n_fft=n_fft, hop_length=hop,
                              n_mels=16, f_min=0, f_max=48000)
                reference = invoke(args.reference, src, reference_out, params)
                if reference.size != params["n_mels"] * frames:
                    raise RuntimeError(f"reference shape mismatch for F={frames}")
                for program in args.programs:
                    got = invoke(program, src, candidate_out, params)
                    if got.shape != reference.shape or not np.allclose(
                            got, reference, rtol=RTOL, atol=ATOL):
                        error = (float(np.max(np.abs(got - reference)))
                                 if got.shape == reference.shape else float("inf"))
                        raise RuntimeError(
                            f"FAIL {program} batch-build-context={batch} F={frames} "
                            f"max_abs_err={error}")
                    error = float(np.max(np.abs(got - reference)))
                    entry = stats[str(program)]
                    entry["cases"] += 1
                    entry["max_abs_err"] = max(entry["max_abs_err"], error)

            # n_fft=1 has a zero periodic-Hann coefficient and bypasses FFTW.
            for frames in sorted({batch, batch + 1, 1024 + batch}):
                pcm = rng.integers(-32768, 32768, frames, dtype=np.int16)
                pcm.astype("<i2").tofile(src)
                params = dict(sr=96000, n_fft=1, hop_length=1,
                              n_mels=16, f_min=0, f_max=48000)
                reference = invoke(args.reference, src, reference_out, params)
                for program in args.programs:
                    got = invoke(program, src, candidate_out, params)
                    if got.shape != reference.shape or not np.allclose(
                            got, reference, rtol=RTOL, atol=ATOL):
                        raise RuntimeError(f"FAIL {program} n_fft=1 F={frames}")
                    entry = stats[str(program)]
                    entry["cases"] += 1

        # Invalid calls are candidate-only: V1 is not the safety reference.
        np.zeros(32, dtype="<i2").tofile(src)
        valid = dict(sr=96000, n_fft=8, hop_length=1,
                     n_mels=16, f_min=0, f_max=48000)
        invalid = []
        for samples in (0, 1, 7):
            np.zeros(samples, dtype="<i2").tofile(tmp / f"short-{samples}.pcm")
            invalid.append((tmp / f"short-{samples}.pcm", dict(valid)))
        invalid.extend((src, {**valid, key: value}) for key, value in (
            ("hop_length", 0), ("hop_length", -1), ("n_fft", 0),
            ("n_fft", 3), ("n_mels", 0), ("sr", 0)))
        for program in args.programs:
            for invalid_src, params in invalid:
                invoke(program, invalid_src, candidate_out, params,
                       expect_success=False)
                stats[str(program)]["rejected"] += 1

        for program in args.programs:
            print(json.dumps({"program": str(program), **stats[str(program)]}),
                  flush=True)


if __name__ == "__main__":
    main()
