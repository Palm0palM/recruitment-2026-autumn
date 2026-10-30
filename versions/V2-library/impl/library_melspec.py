#!/usr/bin/env python3
"""Library-first mel spectrogram CLI (experimental; official semantics unchanged).

Modes:
  a1     Whole-array copy of the provided NumPy/SciPy reference, with stage stats.
  a2     Blocked float32 framing + SciPy R2C + dense BLAS Mel projection.
  a3     Blocked float32 framing + SciPy R2C + SciPy CSR sparse Mel projection.

No audio-specific library or ready-made Mel API is used. The protected main.cpp,
validate.py and data are not touched. Set MELSPEC_STATS=<path> for JSON timings.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

T_MODULE = time.perf_counter()
import numpy as np
from scipy.fft import rfft
from scipy.signal import get_window
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max, dtype):
    if f_max is None:
        f_max = sr / 2.0
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=dtype)
    for m in range(n_mels):
        f_l, f_c, f_r = hz_points[m:m + 3]
        rising = (fft_freqs - f_l) / (f_c - f_l)
        falling = (f_r - fft_freqs) / (f_r - f_c)
        weights[m] = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (f_r - f_l))
    return weights


def periodic_hann(n_fft):
    # scipy's one-point periodic Hann is 1; the problem formula gives w[0]=0.
    # Preserve the problem/V1 edge semantics without changing n_fft > 1.
    if n_fft == 1:
        return np.zeros(1, dtype=np.float64)
    return get_window("hann", n_fft, fftbins=True)


def frame_view(y, n_fft, hop_length):
    n_frames = 1 + (len(y) - n_fft) // hop_length
    return np.lib.stride_tricks.as_strided(
        y, shape=(n_frames, n_fft), strides=(y.strides[0] * hop_length, y.strides[0])
    )


def run_a1(y, args, stats):
    """Exact operation order of the provided reference implementation."""
    t = time.perf_counter()
    window = periodic_hann(args.n_fft)
    frames = frame_view(y, args.n_fft, args.hop_length)
    stats["frame_s"] = time.perf_counter() - t

    t = time.perf_counter()
    windowed = frames * window
    stats["window_s"] = time.perf_counter() - t

    t = time.perf_counter()
    spectrum = rfft(windowed, axis=1, n=args.n_fft)
    stats["fft_s"] = time.perf_counter() - t

    t = time.perf_counter()
    power = np.abs(spectrum) ** 2
    power = power.T
    stats["power_s"] = time.perf_counter() - t

    t = time.perf_counter()
    mel = build_mel_filterbank(args.sr, args.n_fft, args.n_mels,
                               args.f_min, args.f_max, np.float32)
    stats["filterbank_s"] = time.perf_counter() - t

    t = time.perf_counter()
    result = (mel @ power.astype(np.float32)).astype(np.float32)
    stats["mel_s"] = time.perf_counter() - t
    return result


def run_blocked(y, args, stats, sparse_mode):
    dtype = np.float32 if args.dtype == "float32" else np.float64
    n_frames = 1 + (len(y) - args.n_fft) // args.hop_length
    n_bins = args.n_fft // 2 + 1
    t = time.perf_counter()
    window = periodic_hann(args.n_fft).astype(dtype)
    frames = frame_view(y, args.n_fft, args.hop_length)
    mel = build_mel_filterbank(args.sr, args.n_fft, args.n_mels,
                               args.f_min, args.f_max, dtype)
    right_sparse = sparse_mode and args.mel_layout == "right"
    if sparse_mode:
        mel_projection = csr_matrix(mel.T if right_sparse else mel)
    else:
        mel_projection = np.ascontiguousarray(mel)
    result = np.empty((args.n_mels, n_frames), dtype=np.float32)
    stats["setup_s"] = time.perf_counter() - t
    stats["projection_nnz"] = int(mel_projection.nnz) if sparse_mode else 0
    stats["projection_dense_elements"] = 0 if sparse_mode else int(mel.size)

    stats.update({"frame_s": 0.0, "fft_s": 0.0, "power_s": 0.0,
                  "transpose_s": 0.0, "mel_s": 0.0, "store_s": 0.0})
    block = args.block_size
    with threadpool_limits(limits=args.blas_threads, user_api="blas"):
        for start in range(0, n_frames, block):
            stop = min(start + block, n_frames)
            t = time.perf_counter()
            windowed = np.asarray(frames[start:stop] * window, dtype=dtype)
            stats["frame_s"] += time.perf_counter() - t

            t = time.perf_counter()
            spectrum = rfft(windowed, axis=1, n=args.n_fft, workers=args.fft_workers)
            stats["fft_s"] += time.perf_counter() - t

            t = time.perf_counter()
            power = spectrum.real * spectrum.real
            power += spectrum.imag * spectrum.imag
            stats["power_s"] += time.perf_counter() - t

            t = time.perf_counter()
            if right_sparse:
                block_out = np.asarray((power @ mel_projection).T)
                stats["mel_s"] += time.perf_counter() - t
            else:
                power_t = np.ascontiguousarray(power.T)
                stats["transpose_s"] += time.perf_counter() - t
                t = time.perf_counter()
                block_out = mel_projection @ power_t
                stats["mel_s"] += time.perf_counter() - t

            t = time.perf_counter()
            result[:, start:stop] = block_out
            stats["store_s"] += time.perf_counter() - t
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_pcm")
    parser.add_argument("output_bin")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--f_min", type=float, default=0.0)
    parser.add_argument("--f_max", type=float, default=22050.0)
    parser.add_argument("--mode", choices=("a1", "a2", "a3"), default="a3")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--fft-workers", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--mel-layout", choices=("left", "right"), default="left")
    args = parser.parse_args()
    if args.n_fft < 1 or args.n_fft & (args.n_fft - 1) or args.hop_length < 1:
        parser.error("n_fft must be a positive power of two and hop_length must be positive")
    if args.n_mels < 1 or args.sr < 1 or args.block_size < 1:
        parser.error("n_mels, sr and block-size must be positive")
    if args.fft_workers < 1 or args.blas_threads < 1:
        parser.error("fft-workers and blas-threads must be positive")

    main_start = time.perf_counter()
    stats = {"mode": args.mode, "n_fft": args.n_fft, "hop_length": args.hop_length,
             "n_mels": args.n_mels, "block_size": args.block_size,
             "fft_workers": args.fft_workers, "blas_threads": args.blas_threads,
             "dtype": args.dtype, "mel_layout": args.mel_layout,
             "import_s": main_start - T_MODULE}

    t = time.perf_counter()
    pcm = np.fromfile(args.input_pcm, dtype="<i2")
    y = pcm.astype(np.float32) / 32768.0
    del pcm
    stats["read_s"] = time.perf_counter() - t
    if len(y) < args.n_fft:
        raise SystemExit("Audio too short for given n_fft")

    t = time.perf_counter()
    if args.mode == "a1":
        result = run_a1(y, args, stats)
        stats["setup_s"] = 0.0
    else:
        result = run_blocked(y, args, stats, sparse_mode=args.mode == "a3")
    stats["compute_s"] = time.perf_counter() - t

    t = time.perf_counter()
    result.astype("<f4").tofile(args.output_bin)
    stats["write_s"] = time.perf_counter() - t
    stats["internal_s"] = time.perf_counter() - main_start
    stats["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats["n_frames"] = int(result.shape[1])
    stats["bytes_out"] = int(result.size * 4)

    stats_path = os.environ.get("MELSPEC_STATS")
    if stats_path:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
