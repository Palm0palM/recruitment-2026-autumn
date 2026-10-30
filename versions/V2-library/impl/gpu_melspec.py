#!/usr/bin/env python3
"""Optional GPU library-first implementation using CuPy (cuFFT + cuBLAS/sparse).

Runs only with the isolated GPU venv; the shared CPU venv is not modified.
Measures H2D input, device compute and D2H output separately.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as gpu_csr
from scipy.sparse import csr_matrix as host_csr

T0 = time.perf_counter()


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel(sr, n_fft, n_mels, f_min, f_max):
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(points)
    w = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_l, f_c, f_r = hz[m:m + 3]
        rise = (freqs - f_l) / (f_c - f_l)
        fall = (f_r - freqs) / (f_r - f_c)
        w[m] = np.maximum(0.0, np.minimum(rise, fall)) * (2.0 / (f_r - f_l))
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_pcm")
    ap.add_argument("output_bin")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--f_min", type=float, default=0.0)
    ap.add_argument("--f_max", type=float, default=22050.0)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--mode", choices=("dense", "sparse"), default="dense")
    args = ap.parse_args()
    if args.n_fft < 1 or args.n_fft & (args.n_fft - 1) or args.hop_length < 1:
        ap.error("n_fft must be a positive power of two and hop_length positive")
    if args.n_mels < 1 or args.block_size < 1:
        ap.error("n_mels and block-size must be positive")

    main_start = time.perf_counter()
    stats = {"impl": "gpu-cupy", "mode": args.mode, "block_size": args.block_size,
             "n_fft": args.n_fft, "n_mels": args.n_mels,
             "import_s": main_start - T0}
    t = time.perf_counter()
    pcm = np.fromfile(args.input_pcm, dtype="<i2")
    y = pcm.astype(np.float32) / 32768.0
    del pcm
    stats["read_s"] = time.perf_counter() - t
    if len(y) < args.n_fft:
        raise SystemExit("Audio too short for given n_fft")
    n_frames = 1 + (len(y) - args.n_fft) // args.hop_length
    n_bins = args.n_fft // 2 + 1
    if args.n_fft == 1:
        window_host = np.zeros(1, dtype=np.float32)
    else:
        n = np.arange(args.n_fft, dtype=np.float64)
        window_host = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / args.n_fft)).astype(np.float32)
    mel_host = build_mel(args.sr, args.n_fft, args.n_mels, args.f_min, args.f_max)
    if args.mode == "sparse":
        mel_device = gpu_csr(host_csr(mel_host))
    else:
        mel_device = cp.asarray(mel_host)
    window = cp.asarray(window_host)
    output = cp.empty((args.n_mels, n_frames), dtype=cp.float32)

    t = time.perf_counter()
    y_device = cp.asarray(y)
    cp.cuda.Stream.null.synchronize()
    stats["h2d_s"] = time.perf_counter() - t
    stats.update({"frame_s": 0.0, "fft_s": 0.0, "power_s": 0.0,
                  "transpose_s": 0.0, "mel_s": 0.0, "store_s": 0.0})
    block = args.block_size
    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        t = time.perf_counter()
        idx = cp.arange(start, stop, dtype=cp.int64) * args.hop_length
        frames = y_device[idx[:, None] + cp.arange(args.n_fft, dtype=cp.int64)]
        windowed = frames * window
        stats["frame_s"] += time.perf_counter() - t
        t = time.perf_counter()
        spectrum = cp.fft.rfft(windowed, axis=1, n=args.n_fft)
        cp.cuda.Stream.null.synchronize()
        stats["fft_s"] += time.perf_counter() - t
        t = time.perf_counter()
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        stats["power_s"] += time.perf_counter() - t
        t = time.perf_counter()
        power_t = cp.ascontiguousarray(power.T)
        stats["transpose_s"] += time.perf_counter() - t
        t = time.perf_counter()
        block_out = mel_device @ power_t
        cp.cuda.Stream.null.synchronize()
        stats["mel_s"] += time.perf_counter() - t
        t = time.perf_counter()
        output[:, start:stop] = block_out
        stats["store_s"] += time.perf_counter() - t
    t = time.perf_counter()
    cp.cuda.Stream.null.synchronize()
    host_output = output.get()
    stats["d2h_s"] = time.perf_counter() - t
    t = time.perf_counter()
    host_output.astype("<f4").tofile(args.output_bin)
    stats["write_s"] = time.perf_counter() - t
    stats["internal_s"] = time.perf_counter() - main_start
    stats["n_frames"] = n_frames
    stats["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    stats["peak_gpu_mib"] = cp.get_default_memory_pool().used_bytes() / 2**20
    stats_path = os.environ.get("MELSPEC_STATS")
    if stats_path:
        Path(stats_path).write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
