#!/usr/bin/env python3
"""V2-overlap 数学原型，不参与正式构建。

分解（N=S*B，S=hop_length）：
  B_q[r] = sum_j x[start+q+S*j] exp(-2*pi*i*r*j/B)
  A[r+m*B] = FFT_S(B_q[r] * exp(-2*pi*i*r*q/N))[m]

B_q 是实数交错子序列的 B 点 FFT。帧前进 S 个采样时，每个子序列只移动
一个元素，因此可用 Sliding DFT 更新 B_q。周期性 Hann 窗随后在频域精确施加。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def initial_leaves(y: np.ndarray, start: int, n_fft: int, hop: int) -> np.ndarray:
    """返回 [r][q] 的完整 B 点复数子 FFT；只用于数学对照。"""
    s = hop
    b = n_fft // s
    if n_fft != s * b:
        raise ValueError("n_fft must equal hop * B")
    leaves = np.empty((b, s), dtype=np.complex128)
    for q in range(s):
        leaves[:, q] = np.fft.fft(y[start + q:start + q + n_fft:s])
    return leaves


def assemble(leaves: np.ndarray) -> np.ndarray:
    """由子 FFT 重组未加窗实数帧的 rfft，输出 0..N/2。"""
    b, s = leaves.shape
    n = b * s
    q = np.arange(s)
    out = np.empty(n // 2 + 1, dtype=np.complex128)
    for r in range(b):
        twiddle = np.exp(-2j * np.pi * r * q / n)
        z = np.fft.fft(leaves[r] * twiddle)
        out[r::b] = z[:len(out[r::b])]
    return out


def hann_from_unwindowed(a: np.ndarray) -> np.ndarray:
    """由实输入的未加窗 rfft 精确得到周期 Hann 加窗后的 rfft。"""
    n = 2 * (len(a) - 1)
    if n < 4:
        raise ValueError("prototype boundary helper expects n_fft >= 4")
    w = 0.5 * a.copy()
    w[1:-1] -= 0.25 * (a[:-2] + a[2:])
    w[0] -= 0.25 * (a[1] + np.conj(a[1]))       # A[-1] = conj(A[1])
    w[-1] -= 0.25 * (a[-2] + np.conj(a[-2]))   # A[N/2+1] = conj(A[N/2-1])
    return w


def direct_hann(y: np.ndarray, start: int, n_fft: int) -> np.ndarray:
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    return np.fft.fft(y[start:start + n_fft] * window)[:n_fft // 2 + 1]


def scaled_error(got: np.ndarray, ref: np.ndarray) -> dict:
    scale = max(1.0, float(np.max(np.abs(ref))))
    return {"max_abs": float(np.max(np.abs(got - ref))),
            "max_scaled": float(np.max(np.abs(got - ref)) / scale)}


def correctness_tests() -> list[dict]:
    rng = np.random.default_rng(20261030)
    results = []
    for n in (8192, 16384):
        hop = 128
        signals = {
            "random": rng.integers(-32768, 32768, n + 7 * hop, dtype=np.int16).astype(float) / 32768,
            "low_amplitude": rng.standard_normal(n + 7 * hop) * 2**-20,
            "zeros": np.zeros(n + 7 * hop),
            "dc": np.ones(n + 7 * hop),
            "nyquist": np.tile([1.0, -1.0], (n + 7 * hop + 1) // 2)[:n + 7 * hop],
            "impulse": np.zeros(n + 7 * hop),
        }
        signals["impulse"][0] = 1.0
        signals["impulse"][n - 1] = -0.5
        for name, y in signals.items():
            for frame in (0, 1, 7):
                start = frame * hop
                a = assemble(initial_leaves(y, start, n, hop))
                ref_a = np.fft.fft(y[start:start + n])[:n // 2 + 1]
                got_h = hann_from_unwindowed(a)
                ref_h = direct_hann(y, start, n)
                ea, eh = scaled_error(a, ref_a), scaled_error(got_h, ref_h)
                # DC/Nyquist 也包含在逐元素比较中；此处单独记录边界。
                boundary = max(abs(got_h[0] - ref_h[0]), abs(got_h[-1] - ref_h[-1]))
                passed = ea["max_scaled"] < 1e-10 and eh["max_scaled"] < 1e-10
                results.append(dict(n_fft=n, signal=name, frame=frame,
                                    unwindowed=ea, hann=eh,
                                    dc_nyquist_abs=float(boundary), passed=passed))
                if not passed:
                    raise AssertionError(results[-1])
    # 显式改变输入长度，并检查最后一个完整帧，而非只改变同一数组的起点。
    for n in (8192, 16384):
        hop = 128
        for extra in (0, hop, 7 * hop, 1024 * hop + 17):
            y = rng.integers(-32768, 32768, n + extra, dtype=np.int16).astype(float) / 32768
            frame = extra // hop
            start = frame * hop
            a = assemble(initial_leaves(y, start, n, hop))
            ref_a = np.fft.fft(y[start:start + n])[:n // 2 + 1]
            got_h, ref_h = hann_from_unwindowed(a), direct_hann(y, start, n)
            ea, eh = scaled_error(a, ref_a), scaled_error(got_h, ref_h)
            passed = ea["max_scaled"] < 1e-10 and eh["max_scaled"] < 1e-10
            results.append(dict(n_fft=n, signal=f"length_{len(y)}", frame=frame,
                                input_samples=len(y), unwindowed=ea, hann=eh,
                                dc_nyquist_abs=float(max(abs(got_h[0] - ref_h[0]),
                                                         abs(got_h[-1] - ref_h[-1]))),
                                passed=passed))
            if not passed:
                raise AssertionError(results[-1])
    return results


def drift_test(n_fft: int, hop: int, frames: int, refresh: int, seed: int = 2026) -> dict:
    """同时比较有/无周期精确重算的长时间子 FFT 递推误差。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(-32768, 32768, n_fft + hop * (frames - 1), dtype=np.int16).astype(float) / 32768
    b, s = n_fft // hop, hop
    phase = np.exp(2j * np.pi * np.arange(b) / b)
    plain = initial_leaves(y, 0, n_fft, hop)
    guarded = plain.copy()
    checkpoints = sorted(set([0, 1, 15, 255, 1023, 4095, 8191, 16383, frames - 1]))
    checkpoints = [t for t in checkpoints if 0 <= t < frames]
    records = []
    begin = time.perf_counter()
    for t in range(frames):
        if t:
            old_start = (t - 1) * hop
            delta = y[old_start + n_fft:old_start + n_fft + s] - y[old_start:old_start + s]
            plain = phase[:, None] * (plain + delta[None, :])
            if t % refresh == 0:
                guarded = initial_leaves(y, t * hop, n_fft, hop)
            else:
                guarded = phase[:, None] * (guarded + delta[None, :])
        if t in checkpoints:
            ref = np.fft.fft(y[t * hop:t * hop + n_fft])[:n_fft // 2 + 1]
            row = {"frame": t}
            for name, leaves in (("no_refresh", plain), (f"refresh_{refresh}", guarded)):
                got = assemble(leaves)
                row[name] = scaled_error(got, ref)
            records.append(row)
            print(json.dumps(row), flush=True)
    return dict(n_fft=n_fft, hop=hop, frames=frames, refresh=refresh,
                elapsed_s=time.perf_counter() - begin, records=records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--drift-frames", type=int, default=20000)
    ap.add_argument("--drift-nfft", type=int, default=8192)
    ap.add_argument("--refresh", type=int, default=1024)
    args = ap.parse_args()
    started = time.perf_counter()
    correctness = correctness_tests()
    print(f"correctness: {len(correctness)} cases PASS", flush=True)
    drift = drift_test(args.drift_nfft, 128, args.drift_frames, args.refresh)
    result = dict(correctness=correctness, drift=drift,
                  total_elapsed_s=time.perf_counter() - started)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
