#!/usr/bin/env python3
"""Turn fftw_profile.py --out-json records into a Markdown phase table.

Diagnostic only: instrumented build, not a score. 'fixed' is process wall minus
read/compute/write and includes startup, teardown and harness time.
"""
import argparse
import json
from collections import defaultdict


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", type=argparse.FileType("r"))
    ap.add_argument("--out", type=argparse.FileType("w"), default="-")
    args = ap.parse_args()
    rows = json.load(args.records)
    out = args.out
    print("| case | run | wall s | read s | compute s | write s | fixed s | setup s | plan s |", file=out)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|", file=out)
    for r in rows:
        p = r["phases"]
        fixed = r["process_wall_s"] - p.get("read_s", 0) - p.get("compute_s", 0) - p.get("write_s", 0)
        print(f"| {r['case']} | {r['repeat']} | {r['process_wall_s']:.4f} | {p.get('read_s', 0):.4f} | "
              f"{p.get('compute_s', 0):.4f} | {p.get('write_s', 0):.4f} | {fixed:.4f} | "
              f"{p.get('compute_setup_s', 0):.4f} | {p.get('r2c_plan_s', 0):.4f} |", file=out)
    print("\nWorker-stage share (sum over 16 workers, normalized within compute):\n", file=out)
    print("| case | pack % | FFT % | power % | Mel % | effective parallelism |", file=out)
    print("|---|---:|---:|---:|---:|---:|", file=out)
    groups = defaultdict(list)
    for r in rows:
        groups[r["case"]].append(r)
    for case, items in groups.items():
        shares = {k: mean([i["worker_stage_share_pct"][k] for i in items])
                  for k in ("pack_s", "fft_s", "power_s", "mel_s")}
        par = mean([i["effective_parallelism"] for i in items if i["effective_parallelism"]])
        print(f"| {case} | {shares['pack_s']:.1f} | {shares['fft_s']:.1f} | "
              f"{shares['power_s']:.1f} | {shares['mel_s']:.1f} | {par:.1f}× |", file=out)
    print("\nProcess-wall share by phase (mean over runs):\n", file=out)
    print("| case | read % | compute % | write % | fixed % |", file=out)
    print("|---|---:|---:|---:|---:|", file=out)
    for case, items in groups.items():
        w = mean([i["process_wall_s"] for i in items])
        phases = {k: mean([i["phases"].get(k, 0.0) for i in items]) for k in ("read_s", "compute_s", "write_s")}
        fixed = w - sum(phases.values())
        print(f"| {case} | {100*phases['read_s']/w:.1f} | {100*phases['compute_s']/w:.1f} | "
              f"{100*phases['write_s']/w:.1f} | {100*fixed/w:.1f} |", file=out)
    if str(args.out.name) != "<stdout>":
        args.out.close()


if __name__ == "__main__":
    main()
