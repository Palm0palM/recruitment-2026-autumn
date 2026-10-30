#!/usr/bin/env python3
"""本地 V1 perf：保留 -O3，以 DWARF 展开全部线程，生成报告和火焰图。

原始 perf.data、二进制与输出保存在 tmp/，不会改动正式构建或题目源码。
需先准备 FlameGraph：git clone https://github.com/brendangregg/FlameGraph tmp/tools/FlameGraph
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases

V1_SHA256 = "3b00a2d777a20de397c3a2e407b60f6a21258940470851cbeb82c4c21c64f957"


def run(cmd, stdout=None, stderr=None):
    print("+ " + shlex.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=ROOT, stdout=stdout, stderr=stderr)


def summarize_source(path):
    """按栈中最内层的应用源码行归属阶段；每个样本只计一次，不累加 inclusive 行。"""
    counts, tids = Counter(), set()
    for sample in path.read_text().split("\n\n"):
        if "task-clock:u:" not in sample:
            continue
        header = sample.splitlines()[0]
        tid = re.search(r"\s\d+/(\d+)\s", header)
        if tid:
            tids.add(int(tid[1]))
        lines = re.findall(r"^\s+melspec\.cpp:(\d+)", sample, re.MULTILINE)
        line = int(lines[0]) if lines else None
        if line is not None and (52 <= line <= 67 or line == 188):
            phase = "fft_bit_reversal" if line in (54, 55) else "fft_butterflies_and_loop"
        elif line is not None and 184 <= line <= 187:
            phase = "window_pack"
        elif line is not None and 190 <= line <= 199:
            phase = "power_reconstruction"
        elif line is not None and 202 <= line <= 207:
            phase = "mel_projection"
        elif "read_pcm(" in sample:
            phase = "read_pcm"
        elif "write_output(" in sample:
            phase = "write_output"
        elif line is not None and 129 <= line <= 171:
            phase = "compute_setup"
        else:
            phase = "other"
        counts[phase] += 1
    total = sum(counts.values())
    if not total or not counts["fft_butterflies_and_loop"]:
        raise SystemExit("未解析到 V1 FFT 源码栈，请检查调试信息与源码归属规则")
    return dict(total_samples=total, sampled_threads=len(tids),
                phases={k: dict(samples=v, percent=100 * v / total) for k, v in counts.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["default", "large_fft"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--freq", type=int, default=499)
    ap.add_argument("--out", type=Path, default=ROOT / "tmp/perf-v1")
    ap.add_argument("--flamegraph-dir", type=Path, default=ROOT / "tmp/tools/FlameGraph")
    args = ap.parse_args()
    cases = {c["name"]: c for c in load_official_cases()}
    if args.runs < 1 or args.freq < 1 or any(n not in cases for n in args.cases):
        ap.error("用例名无效，或 --runs/--freq 小于 1")
    if hashlib.sha256((ROOT / "src/melspec.cpp").read_bytes()).hexdigest() != V1_SHA256:
        ap.error("阶段归属规则仅适用于 V1 原始迁移源码；请先更新行号映射与 V1_SHA256")
    work, tools = args.out.resolve(), args.flamegraph_dir.resolve()
    if not all((tools / n).is_file() for n in ("stackcollapse-perf.pl", "flamegraph.pl")):
        ap.error("未找到 FlameGraph 脚本，请先准备 --flamegraph-dir")
    # 不覆盖旧的采样证据；复测请另选 --out。
    work.mkdir(parents=True, exist_ok=False)
    binary = work / "melspec-perf"
    os.environ.update(DEBUGINFOD_URLS="", LC_ALL="C")
    compile_cmd = ["g++", "-std=c++17", "-O3", "-march=native", "-pthread", "-g",
                   "-Iinclude", "src/main.cpp", "src/melspec.cpp", "-o", binary]
    with (work / "build.txt").open("w") as f:
        f.write(shlex.join(map(str, compile_cmd)) + "\n")
        f.flush()
        run(compile_cmd, stdout=f, stderr=subprocess.STDOUT)
    with (work / "environment.txt").open("w") as f:
        for cmd in (["git", "rev-parse", "HEAD"], ["uname", "-a"], ["lscpu"],
                    ["free", "-h"], ["g++", "--version"], ["perf", "--version"],
                    ["git", "-C", tools, "rev-parse", "HEAD"]):
            f.write("+ " + shlex.join(map(str, cmd)) + "\n")
            f.flush()
            run(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.write("perf_event_paranoid=" + Path("/proc/sys/kernel/perf_event_paranoid").read_text())
        f.write(next(line + "\n" for line in Path("/proc/self/status").read_text().splitlines()
                     if line.startswith("Cpus_allowed_list:")))
        for p in [ROOT / "src/main.cpp", ROOT / "src/melspec.cpp", binary]:
            f.write(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p) + "\n")
    with (work / "event-probe.txt").open("w") as f:
        run(["perf", "stat", "-e", "cycles:u,instructions:u,cache-references:u,cache-misses:u,task-clock:u", "--", "true"],
            stdout=f, stderr=subprocess.STDOUT)
    data = find_data_dir()
    results = []
    for name in args.cases:
        case = cases[name]
        target = work / name
        target.mkdir()
        output = target / "output.bin"
        cmd = [binary, data / "audio.pcm", output]
        cmd += [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
        # stat 不抓栈；repeat 的数字同样不是集群正式成绩。
        with (target / "stat.txt").open("w") as f:
            run(["perf", "stat", "-r", args.runs, "-e",
                 "task-clock:u,context-switches:u,cpu-migrations:u,page-faults:u", "--", *cmd],
                stdout=f, stderr=subprocess.STDOUT)
        record = ["perf", "record", "--no-buildid-cache", "-e", "task-clock:u", "-F", args.freq,
                  "--call-graph", "dwarf,16384", "-m", "128", "-o", target / "perf.data", "--", *cmd]
        with (target / "record.txt").open("w") as f:
            f.write(shlex.join(map(str, record)) + "\n")
            f.flush()
            run(record, stdout=f, stderr=subprocess.STDOUT)
        got = np.fromfile(output, dtype="<f4")
        ref = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        passed = got.shape == ref.shape and bool(np.allclose(got, ref, rtol=1e-3, atol=1e-4))
        result = dict(case=name, passed=passed, elements=int(got.size),
                      max_abs_err=float(np.max(np.abs(got - ref))) if got.shape == ref.shape else None)
        results.append(result)
        (target / "correctness.json").write_text(json.dumps(result, indent=2) + "\n")
        if not passed:
            raise SystemExit(f"{name}: profile binary output FAIL")
        del got, ref
        for report_name, options in (("report-self.txt", ["--no-children", "--sort", "symbol,dso"]),
                                     ("report-inclusive.txt", ["--children", "--inline", "--sort", "symbol,dso"]),
                                     ("report-lines.txt", ["--no-children", "--sort", "srcline,symbol"])):
            with (target / report_name).open("w") as f:
                run(["perf", "report", "-i", target / "perf.data", "--stdio", "--stdio-color", "never",
                     "--percent-limit", "0.1", "-n", "-g", "none", *options], stdout=f)
        with (target / "stacks.txt").open("w") as f, (target / "script-stderr.txt").open("w") as err:
            # 显式省略 period，避免 stackcollapse 将 ns 周期误当成 sample 数。
            run(["perf", "script", "--inline", "-F", "comm,pid,tid,time,event,ip,sym,dso",
                 "-i", target / "perf.data"], stdout=f, stderr=err)
        with (target / "folded.txt").open("w") as f:
            run(["perl", tools / "stackcollapse-perf.pl", target / "stacks.txt"], stdout=f)
        with (target / "flamegraph.svg").open("w") as f:
            run(["perl", tools / "flamegraph.pl", "--title", f"V1 {name}: local user CPU samples",
                 "--subtitle", "i7-14650HX / WSL2 / task-clock:u / O3 / 16 std::threads / DWARF",
                 "--countname", "samples", target / "folded.txt"], stdout=f)
        with (target / "stacks-source.txt").open("w") as f:
            run(["perf", "script", "--inline", "-F", "comm,pid,tid,time,event,ip,sym,dso,srcline",
                 "-i", target / "perf.data"], stdout=f)
        summary = summarize_source(target / "stacks-source.txt")
        folded_samples = sum(int(line.rsplit(" ", 1)[1])
                             for line in (target / "folded.txt").read_text().splitlines())
        if folded_samples != summary["total_samples"]:
            raise SystemExit("折叠栈与源码归属的样本数不一致，请检查 perf script 的 period 字段")
        (target / "source-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (work / "correctness.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Artifacts: {work}", flush=True)


if __name__ == "__main__":
    main()
