#!/usr/bin/env python3
"""生成 Batch=4/8 阶段计时诊断副本；不修改正式源码。"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bench import ROOT, find_data_dir, load_official_cases


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"插桩锚点不唯一或已变化: {old!r}")
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="default", choices=["default", "large_fft"])
    ap.add_argument("--batch", type=int, default=4, choices=[4, 8, 16])
    ap.add_argument("--precision", type=int, default=64, choices=[64, 32])
    args = ap.parse_args()
    if args.precision == 32 and args.batch not in (8, 16):
        ap.error("--precision 32 requires --batch 8 or 16")
    work = ROOT / f"tmp/batch{args.batch}-fp{args.precision}-profile"
    work.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "src/melspec.cpp").read_text()
    source = '#include <chrono>\n#include <cstdio>\n' + source
    if args.precision == 32:
        if args.batch == 8:
            marker = "#if MELSPEC_FP32_BATCH == 8\n        auto& a = batch_buffers[id];"
            closing = "        }\n#elif MELSPEC_FP32_BATCH == 16 && defined(__AVX512F__)\n        auto& a = batch_buffers[id];"
            suffix = "#elif MELSPEC_FP32_BATCH == 16 && defined(__AVX512F__)\n        auto& a = batch_buffers[id];"
        else:
            marker = "#elif MELSPEC_FP32_BATCH == 16 && defined(__AVX512F__)\n        auto& a = batch_buffers[id];"
            closing = "        }\n#elif defined(__AVX512F__)\n        auto& a = batch_buffers[id];"
            suffix = "#elif defined(__AVX512F__)\n        auto& a = batch_buffers[id];"
        apply_name = f"apply_batch{args.batch}f"
        recover_name = f"recover_power_batch{args.batch}f"
        apply_expr = f"            {apply_name}(plan32, a);"
        recover_expr = (f"            {recover_name}(a, recovery_re, recovery_im, "
                        "support_first, support_last);")
    elif args.batch == 4:
        marker = "#elif defined(__AVX2__) && defined(__FMA__)\n        auto& a = batch_buffers[id];"
        closing = "        }\n#else\n        (void)id;"
        suffix = "#else\n        (void)id;"
        apply_expr = "            apply_batch4(plan, a);"
        recover_expr = "            recover_power_batch4(a, recovery_re, recovery_im, support_first, support_last);"
    else:
        marker = "#if defined(__AVX512F__)\n        auto& a = batch_buffers[id];"
        closing = "        }\n#elif defined(__AVX2__) && defined(__FMA__)\n        auto& a = batch_buffers[id];"
        suffix = "#elif defined(__AVX2__) && defined(__FMA__)\n        auto& a = batch_buffers[id];"
        apply_expr = "            apply_batch8(plan, a);"
        recover_expr = "            recover_power_batch8(a, recovery_re, recovery_im, support_first, support_last);"
    source = replace_once(source, marker, marker.replace("auto& a", """using Clock = std::chrono::steady_clock;
        double pack_s = 0, fft_s = 0, recover_s = 0, mel_s = 0;
        auto& a"""))
    pack_anchor = f"            const int t = batch * {args.batch};\n            const int start = t * hop_length;\n            for (int i = 0; i < half; ++i) {{"
    pack_new = pack_anchor.replace("for (int i", "auto phase = Clock::now();\n            for (int i")
    count = source.count(pack_anchor)
    if count == 1:
        source = source.replace(pack_anchor, pack_new, 1)
    elif count == 2 and args.batch == 8:
        index = source.find(pack_anchor) if args.precision == 32 else source.rfind(pack_anchor)
        source = source[:index] + pack_new + source[index + len(pack_anchor):]
    else:
        raise SystemExit(f"pack 插桩锚点数量异常: {count}")
    source = replace_once(source, apply_expr, """            auto after_pack = Clock::now();
            pack_s += std::chrono::duration<double>(after_pack - phase).count();
""" + apply_expr + """
            auto after_fft = Clock::now();
            fft_s += std::chrono::duration<double>(after_fft - after_pack).count();""")
    source = replace_once(source, recover_expr, recover_expr + """
            auto after_recover = Clock::now();
            recover_s += std::chrono::duration<double>(after_recover - after_fft).count();""")
    source = replace_once(source, closing, """            mel_s += std::chrono::duration<double>(Clock::now() - after_recover).count();
        }
        std::fprintf(stderr, "batch_profile id=%d batches=%d pack_s=%.9f fft_s=%.9f recover_s=%.9f mel_s=%.9f\\n",
                     id, end - begin, pack_s, fft_s, recover_s, mel_s);
""" + suffix)
    (work / "melspec.cpp").write_text(source)
    binary = work / "melspec-profile"
    flags = ["g++", "-std=c++17", "-O3", "-march=native", "-pthread", "-Iinclude"]
    if args.precision == 32:
        flags.append(f"-DMELSPEC_FP32_BATCH={args.batch}")
    flags += ["src/main.cpp", str(work / "melspec.cpp"), "-o", str(binary)]
    subprocess.run(flags, cwd=ROOT, check=True)
    case = next(case for case in load_official_cases() if case["name"] == args.case)
    flags = [item for key, value in case["params"].items() for item in (f"--{key}", str(value))]
    with tempfile.TemporaryDirectory(dir=work) as td:
        output = Path(td) / "out.bin"
        result = subprocess.run([str(binary), str(find_data_dir() / "audio.pcm"), str(output), *flags],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        print(result.stderr, end="")
        rows = [line for line in result.stderr.splitlines() if line.startswith("batch_profile")]
        totals = {key: 0.0 for key in ("pack_s", "fft_s", "recover_s", "mel_s")}
        batches = 0
        for line in rows:
            fields = dict(item.split("=") for item in line.split()[1:])
            batches += int(fields["batches"])
            for key in totals:
                totals[key] += float(fields[key])
        if not batches:
            raise SystemExit("没有采集到 Batch 计时")
        for key, value in totals.items():
            print(f"TOTAL {key}: {value:.9f}s; {value * 1e9 / batches:.3f} ns/batch; "
                  f"{value * 1e9 / (args.batch * batches):.3f} ns/frame")
        got = np.fromfile(output, dtype="<f4")
        expected = np.fromfile(find_data_dir() / Path(case["expected"]).relative_to("data"), dtype="<f4")
        if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-3, atol=1e-4):
            raise SystemExit("诊断输出 FAIL")
        print("诊断输出: PASS")


if __name__ == "__main__":
    main()
