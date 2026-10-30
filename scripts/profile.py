#!/usr/bin/env python3
"""生成 tmp/ 内的插桩副本，不改动正式源码；按线程累计各阶段 wall time。

插桩会增加计时开销，结果只用于定位热点，正式成绩必须使用未插桩二进制。
"""
import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"插桩位置不唯一或已变化，请更新 profile.py: {old!r}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--case", default="default")
    args = parser.parse_args()
    build = ROOT / "tmp" / "profile"
    build.mkdir(parents=True, exist_ok=True)

    # 每个 worker 的累计量留在线程栈上；只在退出时写一行，避免共享计数器干扰。
    source = (ROOT / "src/melspec.cpp").read_text()
    source = '#include <chrono>\n#include <cstdio>\n' + source
    source = replace_once(source, "    auto worker = [&](int id) {", """    auto worker = [&](int id) {
        using Clock = std::chrono::steady_clock;
        double pack_s = 0, fft_s = 0, power_s = 0, mel_s = 0;
        const auto worker_start = Clock::now();""")
    source = replace_once(source, "            if (n_fft == 1) {", """            auto phase = Clock::now();
            if (n_fft == 1) {""")
    source = replace_once(source, "                plan.apply(fft_out);", """                auto now = Clock::now();
                pack_s += std::chrono::duration<double>(now - phase).count();
                phase = now;
                plan.apply(fft_out);
                now = Clock::now();
                fft_s += std::chrono::duration<double>(now - phase).count();
                phase = now;""")
    source = replace_once(source,
        "            // Apply filters immediately, while this frame's power is in cache.",
        """            auto now = Clock::now();
            power_s += std::chrono::duration<double>(now - phase).count();
            phase = now;
            // Apply filters immediately, while this frame's power is in cache.""")
    source = replace_once(source, "        }\n    };", """            mel_s += std::chrono::duration<double>(Clock::now() - phase).count();
        }
        std::fprintf(stderr, "worker=%d pack_s=%.6f fft_s=%.6f power_s=%.6f mel_s=%.6f wall_s=%.6f\\n",
                     id, pack_s, fft_s, power_s, mel_s,
                     std::chrono::duration<double>(Clock::now() - worker_start).count());
    };""")
    (build / "melspec.cpp").write_text(source)

    # 题目 main.cpp 不改；诊断副本只在已有 read/compute/write 调用两侧打点。
    driver = (ROOT / "src/main.cpp").read_text()
    driver = '#include <chrono>\n' + driver
    driver = replace_once(driver, "        read_pcm(a.input_pcm, y);", """        using Clock = std::chrono::steady_clock;
        const auto read_start = Clock::now();
        read_pcm(a.input_pcm, y);
        const auto read_end = Clock::now();""")
    driver = replace_once(driver, "        int n_frames =", """        const auto compute_end = Clock::now();
        int n_frames =""")
    driver = replace_once(driver,
        "        write_output(a.output_bin, output, a.n_mels, n_frames);", """        write_output(a.output_bin, output, a.n_mels, n_frames);
        const auto write_end = Clock::now();
        std::cerr << "process read_s=" << std::chrono::duration<double>(read_end - read_start).count()
                  << " compute_s=" << std::chrono::duration<double>(compute_end - read_end).count()
                  << " write_s=" << std::chrono::duration<double>(write_end - compute_end).count() << "\\n";""")
    (build / "main.cpp").write_text(driver)
    binary = build / "melspec-profile"
    cmd = [os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-march=native",
           "-pthread", "-I" + str(ROOT / "include"), str(build / "main.cpp"),
           str(build / "melspec.cpp"), "-o", str(binary)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if args.build_only:
        return

    from bench import find_data_dir, load_official_cases
    cases = load_official_cases()
    selected = cases if args.case == "all" else [c for c in cases if c["name"] == args.case]
    if not selected:
        parser.error("未知 --case")
    scratch = Path(os.environ.get("TMPDIR", str(build)))
    import tempfile
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        for case in selected:
            params = case["params"]
            flags = [s for k, v in params.items() for s in (f"--{k}", str(v))]
            print(f"== profile {case['name']} ==", flush=True)
            output = Path(td) / "out.bin"
            result = subprocess.run(["/usr/bin/time", "-v", str(binary),
                str(find_data_dir() / "audio.pcm"), str(output), *flags],
                check=True, capture_output=True, text=True)
            print(result.stderr, flush=True)
            totals = dict.fromkeys(("pack_s", "fft_s", "power_s", "mel_s"), 0.0)
            for line in result.stderr.splitlines():
                if line.startswith("worker="):
                    row = dict(pair.split("=") for pair in line.split())
                    for key in totals:
                        totals[key] += float(row[key])
            total = sum(totals.values())
            print("worker phase totals (sum of thread wall time, NOT process wall time):")
            for key, value in totals.items():
                print(f"  {key}: {value:.6f}s ({value / total:.2%})")
            # 诊断副本也要与全量 expected 比较，避免插桩生成错误悄悄漏过。
            import numpy as np
            got = np.fromfile(output, dtype="<f4")
            expected = np.fromfile(find_data_dir() / Path(case["expected"]).relative_to("data"), dtype="<f4")
            if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-3, atol=1e-4):
                raise SystemExit("profile output: FAIL")
            print("profile output: PASS", flush=True)


if __name__ == "__main__":
    main()
