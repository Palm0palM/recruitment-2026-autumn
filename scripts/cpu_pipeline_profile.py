#!/usr/bin/env python3
"""B16 full pipeline diagnostic; original main + I/O hooks, tmp DSP copy only.

Never use this instrumented executable for formal A/B performance claims.
"""
import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases
from batch_profile import replace_once


def instrument(source):
    source = """#include <chrono>
#include <cstdio>
using PipelineClock = std::chrono::steady_clock;
static double pipeline_seconds(PipelineClock::time_point start) {
    return std::chrono::duration<double>(PipelineClock::now() - start).count();
}
""" + source
    source = replace_once(source, "    const int n_samples = static_cast<int>(y.size());",
                          "    const auto setup_start = PipelineClock::now();\n"
                          "    const int n_samples = static_cast<int>(y.size());")
    resize = "    output.resize(static_cast<size_t>(n_mels) * n_frames);"
    source = replace_once(source, resize,
                          '    const double before_resize_s = pipeline_seconds(setup_start);\n'
                          '    const auto resize_start = PipelineClock::now();\n' + resize + '\n'
                          '    std::fprintf(stderr, "setup before_resize_s=%.9f output_init_s=%.9f\\n",\n'
                          '                 before_resize_s, pipeline_seconds(resize_start));')
    marker = "#elif MELSPEC_FP32_BATCH == 16 && defined(__AVX512F__)\n        auto& a = batch_buffers[id];"
    source = replace_once(source, marker, marker.replace("        auto& a", """        const auto worker_start = PipelineClock::now();
        double pack_s = 0, fft_s = 0, recover_s = 0, mel_s = 0;
        auto& a"""))
    anchor = "            const int start = t * hop_length;"
    # Only replace in B16 worker, not other compile-time paths.
    at = source.index(marker.split("\n")[0] + "\n        const auto worker_start")
    end = source.index("#elif defined(__AVX512F__)\n        auto& a = batch_buffers[id];", at)
    prefix, body, suffix = source[:at], source[at:end], source[end:]
    body = replace_once(body, anchor, anchor + "\n            auto phase = PipelineClock::now();")
    body = replace_once(body, "            apply_batch16f(plan32, a);", """            pack_s += pipeline_seconds(phase);
            phase = PipelineClock::now();
            apply_batch16f(plan32, a);
            fft_s += pipeline_seconds(phase);
            phase = PipelineClock::now();""")
    recover = "            recover_power_batch16f(a, recovery_re, recovery_im, support_first, support_last);"
    body = replace_once(body, recover, recover + """
            recover_s += pipeline_seconds(phase);
            phase = PipelineClock::now();""")
    # The final batch-loop brace, not any inner Mel-loop brace.
    body = body.rstrip()[:-1] + "__BATCH_END__\n"
    closing = "        __BATCH_END__\n"
    body = replace_once(body, closing, """            mel_s += pipeline_seconds(phase);
        }
        std::fprintf(stderr, "worker id=%d pack_s=%.9f fft_s=%.9f recover_s=%.9f mel_s=%.9f wall_s=%.9f\\n",
                     id, pack_s, fft_s, recover_s, mel_s, pipeline_seconds(worker_start));
""")
    source = prefix + body + suffix
    source = replace_once(source, "    std::vector<std::thread> threads;", """    const double setup_s = pipeline_seconds(setup_start);
    const auto compute_start = PipelineClock::now();
    std::vector<std::thread> threads;""")
    tail = "    for (int t = tail_begin; t < n_frames; ++t)\n        process_single(t, fft_buffers[0], power_buffers[0]);"
    source = replace_once(source, tail, """    const double workers_s = pipeline_seconds(compute_start);
    const auto tail_start = PipelineClock::now();
""" + tail + """
    std::fprintf(stderr, "pipeline setup_s=%.9f workers_s=%.9f tail_s=%.9f total_s=%.9f\\n",
                 setup_s, workers_s, pipeline_seconds(tail_start), pipeline_seconds(setup_start));
    if (FILE* smaps = std::fopen("/proc/self/smaps_rollup", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), smaps))
            if (std::strstr(line, "AnonHugePages:")) std::fputs(line, stderr);
        std::fclose(smaps);
    }""")
    return source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT / "src/melspec.cpp")
    ap.add_argument("--tag", default="control")
    ap.add_argument("--arch", default="native")
    ap.add_argument("--cppflag", action="append", default=[])
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    work = ROOT / "tmp/pipeline-profile" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    (work / "melspec.cpp").write_text(instrument(args.source.read_text()))
    flags = [os.environ.get("CXX", "g++"), "-std=c++17", "-O3", f"-march={args.arch}",
             "-pthread", "-I" + str(ROOT / "include"), "-DMELSPEC_FP32_BATCH=16", *args.cppflag]
    commands = [flags + ["-finstrument-functions", "-finstrument-functions-exclude-file-list=/usr/include",
                         "-c", str(ROOT / "src/main.cpp"), "-o", str(work / "main.o")],
                flags + [str(work / "main.o"), str(work / "melspec.cpp"),
                         str(ROOT / "scripts/cpu_pipeline_io_hooks.cpp"), "-o", str(work / "profile")]]
    for cmd in commands:
        print("+", shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    if args.build_only:
        return
    rows = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", work)) as td:
        for case in load_official_cases():
            flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
            output = Path(td) / "out.bin"
            for run in range(args.runs):
                start = time.perf_counter()
                result = subprocess.run(["/usr/bin/time", "-v", str(work / "profile"),
                                         str(find_data_dir() / "audio.pcm"), str(output), *flags],
                                        check=True, capture_output=True, text=True)
                wall = time.perf_counter() - start
                print(f"== {case['name']} run={run} diagnostic_wall_s={wall:.9f} ==", flush=True)
                print(result.stderr, flush=True)
                row = {"case": case["name"], "run": run, "wall_s": wall, "workers": []}
                for line in result.stderr.splitlines():
                    if line.startswith(("io ", "setup ", "pipeline ", "worker ")):
                        fields = {k: float(v) for k, v in (p.split("=") for p in line.split()[1:])}
                        if line.startswith("worker "):
                            row["workers"].append(fields)
                        else:
                            row.update(fields)
                if len(row["workers"]) != 16:
                    raise SystemExit("Expected 16 B16 worker diagnostic rows")
                got = np.fromfile(output, dtype="<f4")
                expected = np.fromfile(find_data_dir() / Path(case["expected"]).relative_to("data"), dtype="<f4")
                if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-3, atol=1e-4):
                    raise SystemExit("Diagnostic output FAIL")
                row["pass"] = True
                print("Diagnostic output PASS", flush=True)
                rows.append(row)
    (work / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
