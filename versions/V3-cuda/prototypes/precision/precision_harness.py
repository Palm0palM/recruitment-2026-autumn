#!/usr/bin/env python3
"""Correctness-only harness for V3 CUDA FP32/FP64 candidates.

The candidate must implement the unchanged melspec CLI and write little-endian
float32 [n_mels, n_frames] output.  No timing or benchmarking is performed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any, Iterator

import numpy as np

RTOL = 1e-3
ATOL = 1e-4
TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[2]
OFFICIAL = [
    {
        "name": "default",
        "params": {"sr": 96000, "n_fft": 8192, "hop_length": 128,
                   "n_mels": 256, "f_min": 0.0, "f_max": 22050.0},
        "shape": (256, 192397),
    },
    {
        "name": "small_mels",
        "params": {"sr": 96000, "n_fft": 8192, "hop_length": 128,
                   "n_mels": 64, "f_min": 0.0, "f_max": 22050.0},
        "shape": (64, 192397),
    },
    {
        "name": "large_fft",
        "params": {"sr": 96000, "n_fft": 16384, "hop_length": 128,
                   "n_mels": 256, "f_min": 0.0, "f_max": 22050.0},
        "shape": (256, 192333),
    },
    {
        "name": "narrow_band",
        "params": {"sr": 96000, "n_fft": 16384, "hop_length": 128,
                   "n_mels": 256, "f_min": 300.0, "f_max": 8000.0},
        "shape": (256, 192333),
    },
]


def n_frames(n_samples: int, n_fft: int, hop: int) -> int:
    if n_samples < n_fft:
        raise ValueError("audio too short")
    return 1 + (n_samples - n_fft) // hop


def cli_args(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key in ("sr", "n_fft", "hop_length", "n_mels", "f_min", "f_max"):
        args += [f"--{key}", str(params[key])]
    return args


def program_command(program: Path, pcm: Path, output: Path,
                    params: dict[str, Any]) -> list[str]:
    prefix = [sys.executable, str(program)] if program.suffix == ".py" else [str(program)]
    return prefix + [str(pcm), str(output)] + cli_args(params)


def run_program(program: Path, pcm: Path, output: Path,
                params: dict[str, Any], timeout: int) -> dict[str, Any]:
    cmd = program_command(program, pcm, output, params)
    record: dict[str, Any] = {"command": cmd}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        record.update(returncode=proc.returncode, stdout=proc.stdout,
                      stderr=proc.stderr)
        record["ok"] = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        record.update(ok=False, timeout_seconds=timeout,
                      stdout=exc.stdout, stderr=exc.stderr)
    return record


def random_pcm(length: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        -32768, 32768, size=length, dtype=np.int16)


def signal_pcm(kind: str, length: int, n_fft: int, hop: int,
               seed: int) -> np.ndarray:
    if kind == "random":
        return random_pcm(length, seed)
    if kind == "zero":
        return np.zeros(length, dtype=np.int16)
    if kind == "low_amplitude":
        # PCM quantization makes 1 LSB the smallest nonzero input: 1/32768.
        pattern = np.array([0, 1, -1, 1, -2, 2, -1, 0], dtype=np.int16)
        return np.resize(pattern, length)
    if kind == "full_scale_positive":
        return np.full(length, 32767, dtype=np.int16)
    if kind == "full_scale_negative":
        return np.full(length, -32768, dtype=np.int16)
    if kind == "dc":
        return np.full(length, 16384, dtype=np.int16)
    if kind == "nyquist":
        index = np.arange(length)
        return np.where(index % 2 == 0, 32767, -32768).astype(np.int16)
    if kind == "impulses":
        out = np.zeros(length, dtype=np.int16)
        positions = {0, 1, length - 1, n_fft // 2, n_fft - 1,
                     hop, n_fft + hop - 1}
        for j, pos in enumerate(sorted(p for p in positions if 0 <= p < length)):
            out[pos] = -32768 if j % 2 else 32767
        return out
    raise ValueError(f"unknown signal kind: {kind}")


def synthetic_cases(quick: bool = False) -> Iterator[dict[str, Any]]:
    core_kinds = ["random", "zero", "low_amplitude", "full_scale_positive",
                  "full_scale_negative", "dc", "nyquist", "impulses"]
    if quick:
        core_kinds = ["random", "zero", "low_amplitude", "impulses"]
    for n_fft, hop in ((1, 1), (2, 1), (8, 3)):
        params = {"sr": 96000, "n_fft": n_fft, "hop_length": hop,
                  "n_mels": 8, "f_min": 0.0, "f_max": 22050.0}
        length = n_fft + 2 * hop + 1
        for kind in core_kinds:
            yield {"name": f"core/nfft{n_fft}/{kind}", "params": params,
                   "pcm": signal_pcm(kind, length, n_fft, hop, 1000 + n_fft)}
        boundaries = (n_fft, n_fft + hop - 1, n_fft + hop, n_fft + 2 * hop)
        for length in dict.fromkeys(boundaries):
            yield {"name": f"core/nfft{n_fft}/boundary_N{length}",
                   "params": params,
                   "pcm": random_pcm(length, 2000 + n_fft + length)}

    for spec_index, spec in enumerate(OFFICIAL):
        params = spec["params"]
        n_fft, hop = params["n_fft"], params["hop_length"]
        official_kinds = core_kinds if not quick else ["random", "impulses"]
        for kind in official_kinds:
            yield {"name": f"official-params/{spec['name']}/{kind}",
                   "params": params,
                   "pcm": signal_pcm(kind, n_fft + 2 * hop, n_fft, hop,
                                     3000 + spec_index)}
        if not quick:
            for length in (n_fft, n_fft + hop):
                yield {"name": f"official-params/{spec['name']}/boundary_N{length}",
                       "params": params, "pcm": random_pcm(length, 4000 + length)}


def librosa_reference(pcm: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    import librosa
    y = pcm.astype(np.float32) / np.float32(32768.0)
    result = librosa.feature.melspectrogram(
        y=y, sr=params["sr"], n_fft=params["n_fft"],
        hop_length=params["hop_length"], n_mels=params["n_mels"],
        fmin=params["f_min"], fmax=params["f_max"], center=False,
        htk=True, power=2.0)
    return np.asarray(result, dtype=np.float32)


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape_mismatch",
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape)}
    # This call is the acceptance decision.  Float64 below is diagnostics only.
    passed = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL,
                              equal_nan=False))
    a64 = actual.astype(np.float64)
    e64 = expected.astype(np.float64)
    diff = np.abs(a64 - e64)
    tolerance = ATOL + RTOL * np.abs(e64)
    close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    nonzero = e64 != 0.0
    result: dict[str, Any] = {
        "passed": passed,
        "rtol": RTOL,
        "atol": ATOL,
        "all_elements_finite": bool(np.all(np.isfinite(actual))),
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "max_error_over_tolerance": float(np.max(diff / tolerance)) if diff.size else 0.0,
        "max_relative_error_nonzero_reference": (
            float(np.max(diff[nonzero] / np.abs(e64[nonzero]))) if np.any(nonzero) else None),
        "failing_elements": int(np.count_nonzero(~close)),
    }
    if not passed and np.any(~close):
        first = int(np.argmax(~close.ravel()))
        index = np.unravel_index(first, actual.shape)
        result["first_failure"] = {
            "index": list(map(int, index)), "actual": float(actual.ravel()[first]),
            "expected": float(expected.ravel()[first]),
            "abs_error": float(diff.ravel()[first]),
            "tolerance": float(tolerance.ravel()[first]),
        }
    return result


def output_size_error(path: Path, shape: tuple[int, int]) -> str | None:
    expected_bytes = 4 * math.prod(shape)
    actual_bytes = path.stat().st_size if path.exists() else -1
    if actual_bytes != expected_bytes:
        return f"output byte count {actual_bytes}, expected {expected_bytes}"
    return None


def load_f32_output(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    error = output_size_error(path, shape)
    if error:
        return None, error
    try:
        array = np.fromfile(path, dtype="<f4")
    except (OSError, ValueError) as exc:
        return None, str(exc)
    return array.reshape(shape), None


def inside_task_dir(path: Path) -> bool:
    try:
        path.resolve().relative_to(TASK_DIR)
        return True
    except ValueError:
        return False


def run_synthetic(candidate: Path, v1: Path | None, cases: list[dict[str, Any]],
                  run_dir: Path, timeout: int, fail_fast: bool) -> tuple[list[dict[str, Any]], bool]:
    reports: list[dict[str, Any]] = []
    all_passed = True
    for index, case in enumerate(cases):
        name = case["name"]
        params = case["params"]
        pcm = case["pcm"]
        safe_name = name.replace("/", "__")
        case_dir = run_dir / f"{index:03d}__{safe_name}"
        case_dir.mkdir()
        pcm_path = case_dir / "input.pcm"
        pcm.astype("<i2", copy=False).tofile(pcm_path)
        shape = (params["n_mels"], n_frames(len(pcm), params["n_fft"],
                                            params["hop_length"]))
        report: dict[str, Any] = {"name": name, "params": params,
                                  "n_samples": len(pcm), "shape": list(shape)}
        try:
            librosa = librosa_reference(pcm, params)
            librosa.astype("<f4", copy=False).tofile(case_dir / "librosa.f32")
            v1_array = None
            if v1 is not None:
                v1_output = case_dir / "v1.f32"
                report["v1_run"] = run_program(v1, pcm_path, v1_output, params, timeout)
                if report["v1_run"]["ok"]:
                    v1_array, error = load_f32_output(v1_output, shape)
                    if error:
                        report["v1_load_error"] = error
                if v1_array is not None:
                    report["v1_vs_librosa"] = comparison(v1_array, librosa)

            candidate_output = case_dir / "candidate.f32"
            report["candidate_run"] = run_program(candidate, pcm_path, candidate_output,
                                                  params, timeout)
            if report["candidate_run"]["ok"]:
                actual, error = load_f32_output(candidate_output, shape)
                if error:
                    report["candidate_load_error"] = error
                    case_passed = False
                else:
                    report["candidate_vs_librosa"] = comparison(actual, librosa)
                    case_passed = report["candidate_vs_librosa"]["passed"]
                    if v1 is not None and v1_array is None:
                        # A requested reference execution/load failure is not a pass.
                        case_passed = False
                    if v1_array is not None:
                        report["candidate_vs_v1"] = comparison(actual, v1_array)
                        # A dual match is impossible when the two strict oracles differ.
                        oracle_compatible = report["v1_vs_librosa"]["passed"]
                        if oracle_compatible and not report["candidate_vs_v1"]["passed"]:
                            case_passed = False
            else:
                case_passed = False
            report["passed"] = bool(case_passed)
        except Exception as exc:  # retain a machine-readable failure and continue
            report.update(passed=False, harness_exception=str(exc),
                          traceback=traceback.format_exc())
        reports.append(report)
        all_passed &= report["passed"]
        suffix = ""
        if report.get("v1_vs_librosa") and not report["v1_vs_librosa"]["passed"]:
            suffix = " [V1/librosa ORACLE-DIFF]"
        print(f"[{index + 1}/{len(cases)}] {'PASS' if report['passed'] else 'FAIL'} "
              f"{name}{suffix}", flush=True)
        if fail_fast and not report["passed"]:
            break
    return reports, all_passed


def compare_official_files(actual_path: Path, expected_path: Path,
                           shape: tuple[int, int], chunk_values: int = 1 << 20) -> dict[str, Any]:
    error = output_size_error(actual_path, shape)
    if error:
        return {"passed": False, "reason": error}
    # Both large files remain memmapped; only one chunk is materialized at a time.
    actual_map = np.memmap(actual_path, dtype="<f4", mode="r", shape=shape)
    expected_map = np.memmap(expected_path, dtype="<f4", mode="r", shape=shape)
    flat_a = actual_map.reshape(-1)
    flat_e = expected_map.reshape(-1)
    passed = True
    max_abs = 0.0
    max_ratio = 0.0
    failing = 0
    first_failure = None
    for begin in range(0, len(flat_a), chunk_values):
        end = min(begin + chunk_values, len(flat_a))
        a = np.asarray(flat_a[begin:end])
        e = np.asarray(flat_e[begin:end])
        close = np.isclose(a, e, rtol=RTOL, atol=ATOL, equal_nan=False)
        chunk_passed = bool(np.allclose(a, e, rtol=RTOL, atol=ATOL, equal_nan=False))
        passed &= chunk_passed
        diff = np.abs(a.astype(np.float64) - e.astype(np.float64))
        tolerance = ATOL + RTOL * np.abs(e.astype(np.float64))
        max_abs = max(max_abs, float(np.max(diff)))
        max_ratio = max(max_ratio, float(np.max(diff / tolerance)))
        failing += int(np.count_nonzero(~close))
        if first_failure is None and np.any(~close):
            local = int(np.argmax(~close))
            global_index = begin + local
            first_failure = {
                "flat_index": global_index,
                "index": list(map(int, np.unravel_index(global_index, shape))),
                "actual": float(a[local]), "expected": float(e[local]),
                "abs_error": float(diff[local]), "tolerance": float(tolerance[local]),
            }
    return {"passed": passed, "rtol": RTOL, "atol": ATOL,
            "max_abs_error": max_abs, "max_error_over_tolerance": max_ratio,
            "failing_elements": failing, "first_failure": first_failure,
            "chunk_values": chunk_values}


def run_official(candidate: Path, run_dir: Path, timeout: int,
                 fail_fast: bool, case_re: re.Pattern[str] | None) -> tuple[list[dict[str, Any]], bool]:
    data_dir = Path(os.environ.get("MELSPEC_DATA", str(ROOT / "tmp" / "data")))
    pcm = data_dir / "audio.pcm"
    reports = []
    all_passed = True
    selected = [s for s in OFFICIAL if not case_re or case_re.search(s["name"])]
    for index, spec in enumerate(selected):
        name = f"official-audio/{spec['name']}"
        output = run_dir / f"{spec['name']}.candidate.f32"
        expected = data_dir / "expected" / f"{spec['name']}.bin"
        report: dict[str, Any] = {"name": name, "params": spec["params"],
                                  "shape": list(spec["shape"])}
        report["candidate_run"] = run_program(candidate, pcm, output,
                                              spec["params"], timeout)
        if report["candidate_run"]["ok"]:
            report["candidate_vs_expected"] = compare_official_files(
                output, expected, spec["shape"])
            report["passed"] = report["candidate_vs_expected"]["passed"]
        else:
            report["passed"] = False
        reports.append(report)
        all_passed &= report["passed"]
        print(f"[{index + 1}/{len(selected)}] {'PASS' if report['passed'] else 'FAIL'} {name}",
              flush=True)
        if fail_fast and not report["passed"]:
            break
    return reports, all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path,
                        help="candidate executable or Python CLI")
    parser.add_argument("--suite", choices=("synthetic", "official"), default="synthetic")
    parser.add_argument("--quick", action="store_true",
                        help="reduced synthetic smoke suite; not a substitute for full suite")
    parser.add_argument("--case", dest="case_pattern", help="regular expression selecting cases")
    parser.add_argument("--v1", type=Path, default=ROOT / "melspec",
                        help="fixed V1 executable for diagnostic comparison")
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--timeout", type=int, default=300,
                        help="per-process timeout in seconds (default: 300)")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--run-root", type=Path, default=TASK_DIR / "runs")
    parser.add_argument("--list", action="store_true", help="list selected synthetic cases and exit")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if not inside_task_dir(run_root):
        parser.error(f"--run-root must stay under {TASK_DIR}")
    case_re = re.compile(args.case_pattern) if args.case_pattern else None
    if args.suite == "synthetic":
        cases = [c for c in synthetic_cases(args.quick)
                 if not case_re or case_re.search(c["name"])]
    else:
        cases = []
    if args.list:
        if args.suite != "synthetic":
            for spec in OFFICIAL:
                print(f"official-audio/{spec['name']}")
        else:
            for case in cases:
                print(case["name"])
        return 0
    if not args.candidate:
        parser.error("candidate is required unless --list is used")
    if args.suite == "synthetic" and not cases:
        parser.error("case filter selected no cases")
    candidate = args.candidate.resolve()
    if not candidate.exists():
        parser.error(f"candidate not found: {candidate}")
    v1 = None if args.skip_v1 or args.suite == "official" else args.v1.resolve()
    if v1 is not None and not v1.exists():
        parser.error(f"V1 executable not found: {v1}; use --v1 or --skip-v1")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / f"{stamp}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Correctness artifacts: {run_dir}")
    print(f"Acceptance: np.allclose(rtol={RTOL}, atol={ATOL}, equal_nan=False); no timing")
    if args.suite == "synthetic":
        reports, passed = run_synthetic(candidate, v1, cases, run_dir,
                                        args.timeout, args.fail_fast)
    else:
        reports, passed = run_official(candidate, run_dir, args.timeout,
                                       args.fail_fast, case_re)
    summary = {
        "suite": args.suite, "quick": args.quick, "candidate": str(candidate),
        "v1": str(v1) if v1 else None, "rtol": RTOL, "atol": ATOL,
        "passed": bool(passed), "cases_run": len(reports), "results": reports,
        "python": sys.version, "numpy": np.__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Result: {'PASS' if passed else 'FAIL'}; summary: {run_dir / 'summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
