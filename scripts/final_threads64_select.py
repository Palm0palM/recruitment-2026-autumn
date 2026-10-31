#!/usr/bin/env python3
"""Emit transparent four-configuration CPU64/CUDA rules from measured medians."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--header", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.input.read_text())
    assert len(rows) == 4 and all(r["cpu64_pass"] and r["cuda_pass"] for r in rows)
    lines = ["#pragma once", "// Four measured configurations only; not a general 64-thread cost model.",
             "// Generated from " + args.input.name,
             "struct L40Threads64Rule {",
             "    int sr, n_fft, hop, mels;",
             "    double f_min, f_max;",
             "    bool cuda;", "};",
             "inline constexpr L40Threads64Rule l40_threads64_rules[] = {"]
    for row in rows:
        p = row["params"]
        cuda = row["cuda_median"] < row["cpu64_median"]
        lines.append(f"    // {row['name']}: CPU64 {row['cpu64_median']:.6f}s, CUDA {row['cuda_median']:.6f}s")
        values = [p[k] for k in ("sr", "n_fft", "hop_length", "n_mels", "f_min", "f_max")]
        lines.append("    {" + ", ".join(map(str, values)) + ", " + str(cuda).lower() + "},")
        print(row["name"], "cuda" if cuda else "cpu64", row["cpu64_median"], row["cuda_median"])
    lines.append("};")
    args.header.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
