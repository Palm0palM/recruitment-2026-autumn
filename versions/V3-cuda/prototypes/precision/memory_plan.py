#!/usr/bin/env python3
"""Conservative capacity arithmetic for a batched CUDA mel pipeline.

This is a planning model, not a measurement.  It deliberately uses separate
out-of-place cuFFT input/output, retained power, native-precision Mel sums and
an additional float32 batch output.  Query actual cuFFT workspace and free GPU
memory before allocating.
"""
from __future__ import annotations

import argparse
import json
import math


def frames(n_samples: int, n_fft: int, hop: int) -> int:
    if n_samples < n_fft:
        raise ValueError("n_samples must be >= n_fft")
    return 1 + (n_samples - n_fft) // hop


def plan(n_samples: int, n_fft: int, hop: int, n_mels: int, precision: str,
         gpu_total_gib: float = 45.0, gpu_budget_fraction: float = 0.8,
         plan_workspace_gib: float = 2.0, batch_cap: int = 4096) -> dict:
    if min(n_samples, n_fft, hop, n_mels, batch_cap) <= 0:
        raise ValueError("sizes and batch cap must be positive")
    if hop > n_fft:
        raise ValueError("this streaming-halo model requires hop <= n_fft")
    if not 0.0 < gpu_budget_fraction <= 1.0:
        raise ValueError("GPU budget fraction must be in (0, 1]")
    p = 4 if precision == "fp32" else 8
    f = frames(n_samples, n_fft, hop)
    k = n_fft // 2 + 1
    gib = 1 << 30
    gpu_budget = int(gpu_total_gib * gpu_budget_fraction * gib)
    workspace = int(plan_workspace_gib * gib)

    # Persistent: Hann, dense Mel weights, and explicit cuFFT workspace reserve.
    window = p * n_fft
    weights = p * n_mels * k
    fixed = window + weights + workspace
    # Per batch: y[n_fft+(B-1)hop], FFT input B*n_fft,
    # complex FFT output B*K*2, retained power B*K,
    # native-precision Mel accumulation B*M, and final f32 B*M.
    per_frame = p * (hop + n_fft + 3 * k + n_mels) + 4 * n_mels
    halo_fixed = p * (n_fft - hop)
    available = gpu_budget - fixed - halo_fixed
    bmax = max(0, available // per_frame)
    recommended = min(f, batch_cap, bmax)

    def batch_bytes(b: int) -> int:
        if b <= 0:
            return 0
        y = p * (n_fft + (b - 1) * hop)
        return fixed + y + b * (p * (n_fft + 3 * k + n_mels) + 4 * n_mels)

    # Full-resident working set uses full y instead of a streaming halo.
    full_resident = (fixed + p * n_samples +
                     f * (p * (n_fft + 3 * k + n_mels) + 4 * n_mels))
    output_values = n_mels * f
    host_output_double = 8 * output_values
    host_output_writer_f32 = 4 * output_values
    # Conservative V1-compatible host ownership: PCM + double y + double output
    # + writer's full f32 conversion + batch y and native output staging.
    b = recommended
    host_batch = (p * (n_fft + (b - 1) * hop) + p * n_mels * b) if b else 0
    host_peak_model = (2 * n_samples + 8 * n_samples + host_output_double +
                       host_output_writer_f32 + host_batch)
    return {
        "precision": precision, "bytes_per_real": p,
        "n_samples": n_samples, "n_fft": n_fft, "hop": hop,
        "n_bins": k, "n_mels": n_mels, "n_frames": f,
        "gpu_total_gib_model": gpu_total_gib,
        "gpu_budget_bytes": gpu_budget,
        "gpu_budget_gib": gpu_budget / gib,
        "assumed_cufft_workspace_bytes": workspace,
        "window_bytes": window, "mel_weights_bytes": weights,
        "streaming_halo_fixed_bytes": halo_fixed,
        "per_frame_bytes": per_frame,
        "batch_max_by_model": int(bmax),
        "recommended_batch": int(recommended),
        "recommended_batch_gpu_bytes": batch_bytes(int(recommended)),
        "recommended_batch_gpu_gib": batch_bytes(int(recommended)) / gib,
        "full_resident_gpu_bytes": full_resident,
        "full_resident_gpu_gib": full_resident / gib,
        "full_y_gpu_bytes": p * n_samples,
        "final_output_f32_bytes": 4 * output_values,
        "host_output_double_bytes": host_output_double,
        "host_output_writer_f32_bytes": host_output_writer_f32,
        "host_peak_model_bytes": host_peak_model,
        "host_peak_model_gib": host_peak_model / gib,
        "model_has_safe_batch": bool(recommended > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-fft", type=int, required=True)
    parser.add_argument("--hop", type=int, required=True)
    parser.add_argument("--n-mels", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    parser.add_argument("--gpu-total-gib", type=float, default=45.0,
                        help="planning total; L40 46068 MiB is about 45 GiB")
    parser.add_argument("--gpu-budget-fraction", type=float, default=0.8)
    parser.add_argument("--plan-workspace-gib", type=float, default=2.0,
                        help="reserve only; replace with actual cuFFT plan query")
    parser.add_argument("--batch-cap", type=int, default=4096)
    args = parser.parse_args()
    result = plan(args.n_samples, args.n_fft, args.hop, args.n_mels,
                  args.precision, args.gpu_total_gib,
                  args.gpu_budget_fraction, args.plan_workspace_gib,
                  args.batch_cap)
    print(json.dumps(result, indent=2))
    if not result["model_has_safe_batch"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
