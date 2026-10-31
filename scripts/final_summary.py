#!/usr/bin/env python3
"""Summarize V5 raw JSON without mixing training, holdout or hardware."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--gpu", choices=("cuda", "hip"), required=True)
    ap.add_argument("--official", action="store_true")
    args = ap.parse_args()
    rows = json.loads(args.input.read_text())
    if args.official:
        rows = [r for r in rows if r["name"].startswith("official-")]
    labels = [k for k in rows[0]["times"]]
    print("| Point | " + " | ".join(k + " min / median (s)" for k in labels) + " | Auto chosen |")
    print("|---|" + "---:|" * len(labels) + "---|")
    agree = 0
    choice_regrets = []
    for r in rows:
        chosen = r["decisions"].get("auto", ["-"])[-1]
        print("| " + r["name"] + " | " + " | ".join(
            f"{r[k+'_min']:.4f} / {r[k+'_median']:.4f}" for k in labels) + f" | {chosen} |")
        if chosen != "-":
            oracle = min(("cpu", args.gpu), key=lambda k: r[k+"_median"])
            agree += chosen == oracle
            choice_regrets.append(r[chosen+"_median"] / r[oracle+"_median"] - 1)
    print("\nCumulative medians (equal weight per listed point, not a production mix):")
    for k in labels:
        print(f"  {k}: {sum(r[k+'_median'] for r in rows):.6f} s")
    oracle = sum(min(r["cpu_median"], r[args.gpu+"_median"]) for r in rows)
    print(f"  oracle: {oracle:.6f} s")
    if choice_regrets:
        actual = sum(r["auto_median"] for r in rows)
        print(f"  auto/oracle measured gap: {actual/oracle-1:+.2%}")
        print(f"  chosen backend matches: {agree}/{len(rows)}")
        print(f"  worst selection-only regret: {max(choice_regrets):.2%}")
    passed = all(all(v["passed"] for v in r["checks"].values()) for r in rows)
    print(f"  all measured outputs checked: {passed}")


if __name__ == "__main__":
    main()
