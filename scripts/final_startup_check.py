#!/usr/bin/env python3
"""Real runtime startup/fallback checks, including CPU-only vs unified cost."""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from hybrid_bench import command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("program", type=Path)
    ap.add_argument("--cpu-only", type=Path, required=True)
    ap.add_argument("--gpu", choices=("cuda", "hip"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    program = str(args.program.resolve())
    cpu_only = str(args.cpu_only.resolve())
    result = {}
    with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="startup-") as td:
        td = Path(td)
        small = td / "small.pcm"
        np.random.default_rng(17).integers(-32768, 32768, 8192+63*128,
                                         dtype=np.int16).astype("<i2").tofile(small)
        p = dict(sr=96000, n_fft=8192, hop_length=128, n_mels=64, f_min=0, f_max=22050)
        output = td / "out.bin"

        def run(binary, pcm, params, backend, hidden=False, trace=None, no_cache=False):
            env = dict(os.environ, MELSPEC_BACKEND=backend, MELSPEC_DISPATCH_LOG="1")
            if hidden:
                env.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="")
            if no_cache:
                env.pop("ROCFFT_RTC_CACHE_PATH", None)
            cmd = command(binary, pcm, output, params)
            if trace is not None:
                cmd = ["strace", "-f", "-e", "trace=open,openat,ioctl", "-o", str(trace), *cmd]
            start = time.perf_counter()
            proc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            elapsed = time.perf_counter() - start
            data = np.fromfile(output, dtype="<f4") if proc.returncode == 0 else None
            return proc, elapsed, data

        reference = run(cpu_only, small, p, "cpu")[2]
        times = {k: [] for k in ("cpu-only", "forced-cpu", "auto")}
        for repeat in range(6):
            labels = list(times)
            order = labels[repeat % 3:] + labels[:repeat % 3]
            for label in order:
                proc, elapsed, got = run(cpu_only if label == "cpu-only" else program,
                                         small, p, "auto" if label == "auto" else "cpu")
                assert proc.returncode == 0 and np.allclose(got, reference, rtol=1e-3, atol=1e-4)
                if label == "auto":
                    assert b"backend=cpu" in proc.stderr
                times[label].append(elapsed)
        result["small_times"] = times
        result["small_medians"] = {k: statistics.median(v) for k,v in times.items()}
        if shutil.which("strace"):
            for mode in ("cpu", "auto", args.gpu):
                trace = td / (mode + ".trace")
                proc, _, got = run(program, small, p, mode, trace=trace)
                assert proc.returncode == 0 and np.allclose(got, reference, rtol=1e-3, atol=1e-4)
                text = trace.read_text()
                shutil.copyfile(trace, args.out.with_name(args.out.stem+"-"+mode+".strace.txt"))
                lines = [s for s in text.splitlines() if any(x in s for x in
                         ("/dev/nvidia", "/dev/kfd", "/dev/dri/"))]
                result["trace_"+mode] = dict(device_access_lines=len(lines),
                                             device_access_examples=lines[:10],
                                             stderr=proc.stderr.decode())
        else:
            result["strace"] = "unavailable; no device-open claim made"

        # A real GPU-favored workload, hidden devices. This is an environment
        # failure at initialization, not a simulated kernel or FFT failure.
        big = td / "big.pcm"
        frames = 32768
        q = dict(p, n_fft=65536, n_mels=256)
        samples = q["n_fft"] + (frames-1)*q["hop_length"]
        np.random.default_rng(19).integers(-32768, 32768, samples,
                                         dtype=np.int16).astype("<i2").tofile(big)
        proc, _, ref = run(program, big, q, "cpu")
        assert proc.returncode == 0
        proc, _, actual = run(program, big, q, "auto")
        assert proc.returncode == 0 and ("backend="+args.gpu).encode() in proc.stderr
        assert np.allclose(actual, ref, rtol=1e-3, atol=1e-4)
        result["gpu_favored_control"] = proc.stderr.decode()
        proc, elapsed, got = run(program, big, q, "auto", hidden=True)
        assert proc.returncode == 0 and b"backend=cpu" in proc.stderr
        assert np.array_equal(got, ref)
        result["auto_hidden"] = dict(passed=True, seconds=elapsed, stderr=proc.stderr.decode(),
                                     bitwise_equal_cpu=True, params=q, frames=frames)
        if args.gpu == "hip":
            proc, elapsed, got = run(program, big, q, "auto", no_cache=True)
            assert proc.returncode == 0 and b"rocfft-cache-unready" in proc.stderr
            assert np.array_equal(got, ref)
            result["auto_unprepared_cache"] = dict(passed=True, seconds=elapsed,
                stderr=proc.stderr.decode(), bitwise_equal_cpu=True)
        proc, _, _ = run(program, small, p, args.gpu, hidden=True)
        assert proc.returncode != 0
        result["forced_hidden"] = dict(exit=proc.returncode, stderr=proc.stderr.decode())
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
