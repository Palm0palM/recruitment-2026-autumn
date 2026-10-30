#!/usr/bin/env python3
"""Summarise a hybrid benchmark JSON: auto vs always-CPU vs always-CUDA."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--official-only", action="store_true")
    args = ap.parse_args()
    rows = json.loads(args.input.read_text())
    if args.official_only:
        rows = [r for r in rows if r["name"].startswith("official-")]
    has_auto = all("auto_median" in r for r in rows)
    print(f"{'point':28s} {'frames':>7s} {'cpu':>7s} {'cuda':>7s} "
          f"{'auto':>7s} {'chosen':>5s} {'oracle':>6s} {'regret':>7s}")
    for r in rows:
        oracle = "cpu" if r["cpu_median"] <= r["cuda_median"] else "cuda"
        auto = f"{r['auto_median']:7.4f}" if has_auto else "      -"
        chosen = r.get("auto_backend", "-") if has_auto else "-"
        regret = f"{r['auto_regret']:+7.1%}" if has_auto else "      -"
        print(f"{r['name']:28s} {r['n_frames']:7d} {r['cpu_median']:7.4f} "
              f"{r['cuda_median']:7.4f} {auto} {chosen:>5s} {oracle:>6s} {regret}")
    total_cpu = sum(r["cpu_median"] for r in rows)
    total_cuda = sum(r["cuda_median"] for r in rows)
    print(f"\npoints: {len(rows)}")
    print(f"always CPU  total {total_cpu:8.3f}s  x{total_cpu / total_cpu:.3f}")
    print(f"always CUDA total {total_cuda:8.3f}s  x{total_cuda / total_cpu:.3f}")
    if has_auto:
        total_auto = sum(r["auto_median"] for r in rows)
        oracle = sum(min(r["cpu_median"], r["cuda_median"]) for r in rows)
        agree = sum(r["auto_backend"] == ("cpu" if r["cpu_median"] <= r["cuda_median"]
                                          else "cuda") for r in rows)
        picked = sum(r["auto_backend"] == "cuda" for r in rows)
        worst = max(rows, key=lambda r: r["auto_regret"])
        print(f"auto        total {total_auto:8.3f}s  x{total_auto / total_cpu:.3f}  "
              f"(oracle {oracle:.3f}s, gap {total_auto / oracle - 1:+.2%})")
        print(f"auto chose CUDA on {picked}/{len(rows)} points; "
              f"matches per-point oracle {agree}/{len(rows)}")
        print(f"worst per-point regret {worst['auto_regret']:+.1%} on {worst['name']}")


if __name__ == "__main__":
    main()
