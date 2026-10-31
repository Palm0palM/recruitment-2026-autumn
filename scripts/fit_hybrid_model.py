#!/usr/bin/env python3
"""Fit the CPU/GPU full-process cost model used by the Hybrid dispatcher.

Input is the JSON produced by scripts/hybrid_bench.py. The model is linear in
three workload terms plus an intercept, fitted per backend with NNLS so every
coefficient stays non-negative and the prediction is monotone in workload:

  work    = n_frames * n_fft * log2(n_fft) / 1e8
  input   = samples * 2 / 1e6            (PCM payload in MB)
  output  = n_mels * n_frames * 4 / 1e6  (float32 output in MB)

`--emit-cpp` prints the exact macro block that src/melspec.cpp uses, so the
dispatcher constant and the measured data cannot drift apart silently.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


def features(row):
    p = row["params"]
    frames = row["n_frames"]
    n_fft = p["n_fft"]
    n_mels = p["n_mels"]
    samples = n_fft + (frames - 1) * p["hop_length"]
    fft_work = frames * n_fft * math.log2(n_fft) / 1e8
    input_mb = samples * 2 / 1e6
    output_mb = n_mels * frames * 4 / 1e6
    return [1.0, fft_work, input_mb, output_mb]


def fit(X, y):
    # NNLS keeps the cost model monotone and explainable.
    coef, residual = nnls(X, y)
    return coef, float(residual)


CPP_MACROS = (
    ("CPU_CONST", 0), ("CPU_WORK", 1), ("CPU_INPUT", 2), ("CPU_OUTPUT", 3),
    ("CUDA_CONST", 0), ("CUDA_WORK", 1), ("CUDA_INPUT", 2), ("CUDA_OUTPUT", 3),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--emit-cpp", action="store_true",
                    help="print the MELSPEC_MODEL_* macro block for src/melspec.cpp")
    args = ap.parse_args()
    rows = json.loads(args.input.read_text())
    X = np.array([features(row) for row in rows])
    y_cpu = np.array([row["cpu_median"] for row in rows])
    y_gpu = np.array([row["cuda_median"] for row in rows])
    cpu, r_cpu = fit(X, y_cpu)
    gpu, r_gpu = fit(X, y_gpu)
    pred_cpu = X @ cpu
    pred_gpu = X @ gpu
    pred = np.where(pred_cpu <= pred_gpu, "cpu", "cuda")
    truth = np.where(y_cpu <= y_gpu, "cpu", "cuda")
    correct = pred == truth
    accuracy = float(np.mean(correct))
    print("features: [1, fft_work(frames*nfft*log2(nfft)/1e8), input_MB, output_MB]")
    print("cpu coefficients:", " ".join(f"{v:.9g}" for v in cpu), f"residual={r_cpu:.6g}")
    print("gpu coefficients:", " ".join(f"{v:.9g}" for v in gpu), f"residual={r_gpu:.6g}")
    print(f"winner accuracy={accuracy:.3f} ({int(np.sum(correct))}/{len(rows)})")
    print(f"cpu MAE={np.mean(np.abs(pred_cpu-y_cpu)):.4f}s "
          f"gpu MAE={np.mean(np.abs(pred_gpu-y_gpu)):.4f}s")
    print("misclassified:")
    for i in range(len(rows)):
        if correct[i]:
            continue
        row = rows[i]
        regret = abs(row["cpu_median"] - row["cuda_median"]) / min(
            row["cpu_median"], row["cuda_median"])
        print(f"  {row['name']}: cpu={row['cpu_median']:.4f} cuda={row['cuda_median']:.4f} "
              f"pred_cpu={pred_cpu[i]:.4f} pred_cuda={pred_gpu[i]:.4f} "
              f"chose={pred[i]} regret={regret:.1%}")
    if args.out:
        args.out.write_text(json.dumps({
            "features": ["bias", "fft_work", "input_mb", "output_mb"],
            "cpu": list(map(float, cpu)),
            "gpu": list(map(float, gpu)),
            "accuracy": accuracy,
            "cpu_mae": float(np.mean(np.abs(pred_cpu - y_cpu))),
            "gpu_mae": float(np.mean(np.abs(pred_gpu - y_gpu))),
        }, indent=2) + "\n")
    if args.emit_cpp:
        values = {"CPU": cpu, "CUDA": gpu}
        for name, index in CPP_MACROS:
            backend, term = name.split("_")
            sys.stdout.write(f"#define MELSPEC_MODEL_{name} {values[backend][index]:.9g}\n")


if __name__ == "__main__":
    main()
