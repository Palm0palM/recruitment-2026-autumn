#!/usr/bin/env python3
"""生成 half-delta 的单线程阶段计时副本，并对官方 expected 校验。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench import ROOT, find_data_dir, load_official_cases


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"插桩位置不唯一或已变化: {old!r}")
    return text.replace(old, new, 1)


def instrument(source: str) -> str:
    source = "#include <chrono>\n#include <cstdio>\n" + source
    source = replace_once(source, "void overlap_power(", """
static double phase_update = 0, phase_twiddle = 0, phase_fft = 0;
static double phase_store = 0, phase_hann_power = 0, phase_mel = 0;
using PhaseClock = std::chrono::steady_clock;

void overlap_power(""")
    source = replace_once(source, "std::thread::hardware_concurrency()", "1u")
    source = replace_once(source, "    const int n = plan.n_fft;", """    const int n = plan.n_fft;
    auto mark = PhaseClock::now();""")
    source = replace_once(source, "    }\n\n    const int stride = s / 2 + 1;", """    }
    phase_update += std::chrono::duration<double>(PhaseClock::now() - mark).count();
    mark = PhaseClock::now();

    const int stride = s / 2 + 1;""")
    source = replace_once(source, "        plan.combine_plan.apply_butterflies(scratch.combined);", """        phase_twiddle += std::chrono::duration<double>(PhaseClock::now() - mark).count();
        mark = PhaseClock::now();
        plan.combine_plan.apply_butterflies(scratch.combined);
        phase_fft += std::chrono::duration<double>(PhaseClock::now() - mark).count();
        mark = PhaseClock::now();""")
    source = replace_once(source, "    }\n\n    // Periodic Hann:", """        phase_store += std::chrono::duration<double>(PhaseClock::now() - mark).count();
        mark = PhaseClock::now();
    }

    // Periodic Hann:""")
    source = replace_once(source, "\n}\n\n// Real FFT: input n real values", """
    phase_hann_power += std::chrono::duration<double>(PhaseClock::now() - mark).count();
}

// Real FFT: input n real values""")
    source = replace_once(source, "            // Apply filters immediately, while this frame's power is in cache.", """            auto mel_mark = PhaseClock::now();
            // Apply filters immediately, while this frame's power is in cache.""")
    source = replace_once(source, "                output[static_cast<size_t>(m) * n_frames + t] = acc;\n            }\n        }\n    };", """                output[static_cast<size_t>(m) * n_frames + t] = acc;
            }
            phase_mel += std::chrono::duration<double>(PhaseClock::now() - mel_mark).count();
        }
        std::fprintf(stderr, "PHASE update=%.9f twiddle=%.9f fft=%.9f store=%.9f hann_power=%.9f mel=%.9f\\n",
                     phase_update, phase_twiddle, phase_fft, phase_store, phase_hann_power, phase_mel);
    };""")
    return source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="default")
    ap.add_argument("--source", type=Path,
                    default=ROOT / "versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    case = next(c for c in load_official_cases() if c["name"] == args.case)
    build = ROOT / "tmp/phase-profile"
    build.mkdir(parents=True, exist_ok=True)
    source = instrument(args.source.read_text())
    source_path = build / f"{args.case}.cpp"
    binary = build / args.case
    source_path.write_text(source)
    subprocess.run(["g++", "-std=c++17", "-O3", "-march=native", "-pthread", "-Iinclude",
                    "src/main.cpp", str(source_path), "-o", str(binary)], cwd=ROOT, check=True)
    data = find_data_dir()
    flags = [s for k, v in case["params"].items() for s in (f"--{k}", str(v))]
    scratch = ROOT / "tmp/scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        output = Path(td) / "out.bin"
        result = subprocess.run(["taskset", "-c", "0", str(binary), str(data / "audio.pcm"),
                                 str(output), *flags], check=True, capture_output=True, text=True)
        lines = [line for line in result.stderr.splitlines() if line.startswith("PHASE ")]
        if len(lines) != 1:
            raise SystemExit(f"预期一行 PHASE，实际为 {lines}")
        phases = {k: float(v) for k, v in (pair.split("=") for pair in lines[0][6:].split())}
        got = np.fromfile(output, dtype="<f4")
        expected = np.fromfile(data / Path(case["expected"]).relative_to("data"), dtype="<f4")
        passed = got.shape == expected.shape and bool(np.allclose(got, expected, rtol=1e-3, atol=1e-4))
    total = sum(phases.values())
    report = dict(case=args.case, source=str(args.source), single_thread=True, phases_seconds=phases,
                  total_timed_seconds=total,
                  phases_percent={k: 100 * v / total for k, v in phases.items()},
                  passed=passed,
                  note="诊断插桩有额外开销；Hann 与模平方为融合阶段，不是未插桩成绩。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("stage output FAIL")


if __name__ == "__main__":
    main()
