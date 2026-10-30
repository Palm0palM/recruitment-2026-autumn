#!/usr/bin/env python3
"""Serial, small-output numerical stress search for the fftwf candidates.

This is a falsification harness, not a proof of numerical correctness.  It uses
the protected CLI and the original float32 acceptance tolerance.  PCM and raw
output files live only in a TemporaryDirectory below TMPDIR; --out receives the
complete JSON report.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


RTOL = 1.0e-3
ATOL = 1.0e-4
N_FFT_VALUES = (256, 2048, 8192, 16384, 32768, 65536)

CONFIGS = {
    "wide256": {
        "sr": 96000,
        "n_mels": 256,
        "f_min": 0.0,
        "f_max": 48000.0,
    },
    "wide64": {
        "sr": 48000,
        "n_mels": 64,
        "f_min": 0.0,
        "f_max": 24000.0,
    },
    "narrow64": {
        "sr": 96000,
        "n_mels": 64,
        "f_min": 100.0,
        "f_max": 4000.0,
    },
}


def case_descriptor(n_fft, signal, config_name, frames, **signal_args):
    hop = n_fft // 4
    params = dict(CONFIGS[config_name])
    params.update(n_fft=n_fft, hop_length=hop)
    # Keep the protected CLI's expected parameter order explicit at run time.
    descriptor = {
        "id": f"N{n_fft}-{signal}-{config_name}-f{frames}",
        "n_fft": n_fft,
        "signal": signal,
        "signal_args": signal_args,
        "config": config_name,
        "frames": frames,
        "n_samples": n_fft + (frames - 1) * hop,
        "params": params,
    }
    return descriptor


def build_cases():
    cases = []
    for n in N_FFT_VALUES:
        # Exact integer-period, full-scale signals.  Every supported n_fft is
        # divisible by both periods, including the final partial hop region.
        cases.append(case_descriptor(n, "period4", "wide256", 1))
        cases.append(case_descriptor(n, "period8", "wide64", 4))

        # 6000/96000 and 3000/48000 are both exactly 1/16, so k=N/16
        # is integral for every N in this matrix.
        k_wide256 = n // 16
        f_bin = k_wide256 * CONFIGS["wide256"]["sr"] / n
        cases.append(case_descriptor(
            n, "quantized-bin-tone", "wide256", 1,
            frequency_hz=f_bin, amplitude=30000, bin=k_wide256))

        k_wide64 = n // 16
        f_offbin = (k_wide64 + 0.5) * CONFIGS["wide64"]["sr"] / n
        cases.append(case_descriptor(
            n, "quantized-offbin-tone", "wide64", 4,
            frequency_hz=f_offbin, amplitude=30000, bin=k_wide64 + 0.5))

        k_nyquist = n / 2.0 - 0.5
        f_nyquist = k_nyquist * CONFIGS["wide256"]["sr"] / n
        cases.append(case_descriptor(
            n, "near-nyquist", "wide256", 1,
            frequency_hz=f_nyquist, amplitude=32767, bin=k_nyquist))

        # Quiet and multitone cases land inside the narrow band.  Rounding the
        # bin index is intentional; the exact realized frequency is recorded.
        sr_narrow = CONFIGS["narrow64"]["sr"]
        k_quiet = max(1, round(2000.0 * n / sr_narrow))
        f_quiet = k_quiet * sr_narrow / n
        cases.append(case_descriptor(
            n, "quiet-tone", "narrow64", 4,
            frequency_hz=f_quiet, amplitude=4, bin=k_quiet))

        # A negative full-scale DC signal with exactly one sample raised by
        # one int16 LSB avoids clipping and makes the perturbation unambiguous.
        cases.append(case_descriptor(
            n, "fullscale-dc-plus-1lsb", "wide256", 1,
            base=-32768, changed_value=-32767, index=n // 3))

        k_strong = max(1, round(1000.0 * n / sr_narrow))
        k_weak = max(k_strong + 2, round(2500.0 * n / sr_narrow))
        f_strong = k_strong * sr_narrow / n
        f_weak = (k_weak + 0.5) * sr_narrow / n
        cases.append(case_descriptor(
            n, "multitone-plus-quiet", "narrow64", 4,
            strong_frequency_hz=f_strong, weak_frequency_hz=f_weak,
            strong_amplitude=24000, weak_amplitude=4,
            strong_bin=k_strong, weak_bin=k_weak + 0.5))

        cases.append(case_descriptor(n, "sparse-impulse", "wide64", 4))
    assert len(cases) <= 100
    return cases


def quantized_tone(n_samples, sr, frequency, amplitude, phase=0.0):
    index = np.arange(n_samples, dtype=np.float64)
    values = amplitude * np.sin(
        (2.0 * np.pi * frequency / sr) * index + phase)
    return np.clip(np.rint(values), -32768, 32767).astype(np.int16)


def make_pcm(case):
    n = case["n_samples"]
    n_fft = case["n_fft"]
    sr = case["params"]["sr"]
    kind = case["signal"]
    args = case["signal_args"]

    if kind == "period4":
        pattern = np.array([32767, 0, -32768, 0], dtype=np.int16)
        return np.resize(pattern, n).copy()
    if kind == "period8":
        pattern = np.array(
            [32767, 32767, 0, -32768, -32768, -32768, 0, 32767],
            dtype=np.int16)
        return np.resize(pattern, n).copy()
    if kind in ("quantized-bin-tone", "quantized-offbin-tone",
                "near-nyquist", "quiet-tone"):
        return quantized_tone(
            n, sr, args["frequency_hz"], args["amplitude"])
    if kind == "fullscale-dc-plus-1lsb":
        pcm = np.full(n, args["base"], dtype=np.int16)
        pcm[args["index"]] = args["changed_value"]
        return pcm
    if kind == "multitone-plus-quiet":
        index = np.arange(n, dtype=np.float64)
        strong = args["strong_amplitude"] * np.sin(
            (2.0 * np.pi * args["strong_frequency_hz"] / sr) * index)
        weak = args["weak_amplitude"] * np.sin(
            (2.0 * np.pi * args["weak_frequency_hz"] / sr) * index)
        # Quantize only after summing so the weak component genuinely has to
        # survive the strong waveform and final int16 rounding.
        return np.clip(np.rint(strong + weak),
                       -32768, 32767).astype(np.int16)
    if kind == "sparse-impulse":
        pcm = np.zeros(n, dtype=np.int16)
        locations = [0, 1, n_fft // 2, n_fft - 1]
        if n > n_fft:
            locations.extend([
                n_fft,
                n_fft + case["params"]["hop_length"] // 2,
                n - 1,
            ])
        for j, location in enumerate(locations):
            if 0 <= location < n:
                pcm[location] = 32767 if j % 2 == 0 else -32768
        return pcm
    raise ValueError(f"unknown signal kind: {kind}")


def cli_command(executable, pcm_path, output_path, params):
    command = [str(executable), str(pcm_path), str(output_path)]
    for key in ("sr", "n_fft", "hop_length", "n_mels", "f_min", "f_max"):
        command.extend((f"--{key}", str(params[key])))
    return command


def decode_captured(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_one(executable, pcm_path, output_path, params, timeout):
    command = cli_command(executable, pcm_path, output_path, params)
    record = {"executable": str(executable), "command": command}
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False)
        record.update(
            returncode=result.returncode,
            stdout=decode_captured(result.stdout),
            stderr=decode_captured(result.stderr),
            timed_out=False)
    except subprocess.TimeoutExpired as exc:
        record.update(
            returncode=None,
            stdout=decode_captured(exc.stdout),
            stderr=decode_captured(exc.stderr),
            timed_out=True)
    except OSError as exc:
        record.update(
            returncode=None, stdout="", stderr=str(exc), timed_out=False,
            launch_error=True)
    return record


def read_f32(path):
    if not path.is_file():
        raise FileNotFoundError(f"output file was not created: {path}")
    return np.fromfile(path, dtype="<f4")


def json_scalar(value):
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "+Inf" if value > 0 else "-Inf"


def invalid_comparison(reason, expected_size, got=None, ref=None):
    result = {
        "passed": False,
        "reason": reason,
        "expected_size": int(expected_size),
        "got_size": None if got is None else int(got.size),
        "ref_size": None if ref is None else int(ref.size),
        "normalized_error": None,
        "abs": None,
        "rel": None,
        "maxindex": None,
        "ref": None,
        "got": None,
    }
    arrays = [] if ref is None else [ref]
    if got is not None:
        arrays.append(got)
    for array in arrays:
        bad = np.flatnonzero(~np.isfinite(array))
        if bad.size:
            index = int(bad[0])
            result["maxindex"] = index
            if ref is not None and index < ref.size:
                result["ref"] = json_scalar(ref[index])
            if got is not None and index < got.size:
                result["got"] = json_scalar(got[index])
            break
    return result


def compare_output(ref, got, expected_size):
    if ref.size != expected_size or got.size != expected_size:
        return invalid_comparison("shape", expected_size, got=got, ref=ref)
    if not np.all(np.isfinite(ref)):
        return invalid_comparison("reference-nonfinite", expected_size,
                                  got=got, ref=ref)
    if not np.all(np.isfinite(got)):
        return invalid_comparison("got-nonfinite", expected_size,
                                  got=got, ref=ref)

    ref64 = ref.astype(np.float64)
    got64 = got.astype(np.float64)
    abs_error = np.abs(got64 - ref64)
    budget = ATOL + RTOL * np.abs(ref64)
    normalized = abs_error / budget
    maxindex = int(np.argmax(normalized))
    abs_at_worst = float(abs_error[maxindex])
    ref_at_worst = float(ref64[maxindex])
    if ref_at_worst != 0.0:
        rel_at_worst = abs_at_worst / abs(ref_at_worst)
    elif abs_at_worst == 0.0:
        rel_at_worst = 0.0
    else:
        # JSON null denotes an undefined 0/0-free infinite relative error;
        # abs/ref/got and normalized_error still fully identify the failure.
        rel_at_worst = None
    failed = normalized > 1.0
    max_abs_index = int(np.argmax(abs_error))
    return {
        "passed": not bool(np.any(failed)),
        "reason": None if not np.any(failed) else "tolerance",
        "expected_size": int(expected_size),
        "got_size": int(got.size),
        "ref_size": int(ref.size),
        "num_failed": int(np.count_nonzero(failed)),
        "normalized_error": float(normalized[maxindex]),
        "abs": abs_at_worst,
        "rel": rel_at_worst,
        "maxindex": maxindex,
        "ref": ref_at_worst,
        "got": float(got64[maxindex]),
        "max_abs": float(abs_error[max_abs_index]),
        "max_abs_index": max_abs_index,
        "max_abs_ref": float(ref64[max_abs_index]),
        "max_abs_got": float(got64[max_abs_index]),
    }


def parse_program(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("program must have the form name=exe")
    name, executable = value.split("=", 1)
    if not name or not executable:
        raise argparse.ArgumentTypeError("program must have nonempty name and exe")
    return name, executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True,
                        help="independently built double-precision A2 executable")
    parser.add_argument("--programs", required=True, nargs="+",
                        type=parse_program, metavar="name=exe",
                        help="candidate executables; all are run serially")
    parser.add_argument("--out", required=True, type=Path,
                        help="orchestrator-private directory for summary.json")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="per-process timeout in seconds (default: 120)")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")

    names = [name for name, _ in args.programs]
    if len(set(names)) != len(names):
        parser.error("candidate names must be unique")

    args.out.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    report_cases = []
    program_summaries = {
        name: {
            "passed": 0, "failed": 0, "not_comparable": 0,
            "worst_normalized_error": None, "worst_case": None,
            "worst_detail": None,
        }
        for name in names
    }
    reference_problem_count = 0

    # TMPDIR is selected explicitly.  No PCM or raw output is written below
    # --out, and context-manager cleanup also runs after ordinary failures.
    scratch_parent = os.environ.get("TMPDIR") or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(
            prefix="fftwf-stress-", dir=scratch_parent) as scratch:
        scratch = Path(scratch)
        pcm_path = scratch / "input.pcm"
        reference_path = scratch / "reference.f32"
        candidate_path = scratch / "candidate.f32"

        for case_number, case in enumerate(cases, 1):
            pcm = make_pcm(case)
            pcm.astype("<i2").tofile(pcm_path)
            expected_size = case["params"]["n_mels"] * case["frames"]

            for path in (reference_path, candidate_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            reference_run = run_one(
                args.reference, pcm_path, reference_path,
                case["params"], args.timeout)
            reference = None
            reference_read_error = None
            reference_valid = reference_run.get("returncode") == 0
            if reference_valid:
                try:
                    reference = read_f32(reference_path)
                    if (reference.size != expected_size
                            or not np.all(np.isfinite(reference))):
                        reference_valid = False
                except (OSError, ValueError) as exc:
                    reference_read_error = str(exc)
                    reference_valid = False
            if not reference_valid:
                reference_problem_count += 1

            case_record = dict(case)
            case_record["pcm_summary"] = {
                "dtype": "int16",
                "samples": int(pcm.size),
                "min": int(pcm.min()),
                "max": int(pcm.max()),
                "nonzero": int(np.count_nonzero(pcm)),
            }
            case_record["reference"] = {
                "run": reference_run,
                "valid": reference_valid,
                "read_error": reference_read_error,
                "output_size": None if reference is None else int(reference.size),
            }
            case_record["programs"] = {}

            for name, executable in args.programs:
                try:
                    candidate_path.unlink()
                except FileNotFoundError:
                    pass
                candidate_run = run_one(
                    executable, pcm_path, candidate_path,
                    case["params"], args.timeout)
                candidate = None
                read_error = None
                if candidate_run.get("returncode") == 0:
                    try:
                        candidate = read_f32(candidate_path)
                    except (OSError, ValueError) as exc:
                        read_error = str(exc)

                if not reference_valid:
                    comparison = invalid_comparison(
                        "reference-unavailable-or-invalid", expected_size,
                        got=candidate, ref=reference)
                    program_summaries[name]["not_comparable"] += 1
                elif candidate_run.get("returncode") != 0:
                    comparison = invalid_comparison(
                        "candidate-process-failed", expected_size)
                    program_summaries[name]["failed"] += 1
                elif read_error is not None:
                    comparison = invalid_comparison(
                        "candidate-output-unreadable", expected_size,
                        ref=reference)
                    comparison["read_error"] = read_error
                    program_summaries[name]["failed"] += 1
                else:
                    comparison = compare_output(
                        reference, candidate, expected_size)
                    if comparison["passed"]:
                        program_summaries[name]["passed"] += 1
                    else:
                        program_summaries[name]["failed"] += 1
                    normalized = comparison["normalized_error"]
                    old = program_summaries[name]["worst_normalized_error"]
                    if normalized is not None and (old is None or normalized > old):
                        program_summaries[name]["worst_normalized_error"] = normalized
                        program_summaries[name]["worst_case"] = case["id"]
                        program_summaries[name]["worst_detail"] = {
                            key: comparison[key] for key in
                            ("abs", "rel", "maxindex", "ref", "got")
                        }

                case_record["programs"][name] = {
                    "run": candidate_run,
                    "comparison": comparison,
                }

            report_cases.append(case_record)
            statuses = " ".join(
                f"{name}={case_record['programs'][name]['comparison']['reason'] or 'ok'}"
                for name in names)
            print(f"[{case_number}/{len(cases)}] {case['id']} "
                  f"reference={'ok' if reference_valid else 'ERROR'} {statuses}",
                  flush=True)

    all_ok = reference_problem_count == 0
    for summary in program_summaries.values():
        if summary["failed"] or summary["not_comparable"]:
            all_ok = False
    report = {
        "kind": "fftwf-precision-stress-v1",
        "proof_status": "counterexample-search-only-not-correctness-proof",
        "reference": str(args.reference),
        "programs": {name: executable for name, executable in args.programs},
        "tolerance": {"rtol": RTOL, "atol": ATOL,
                      "normalized_error_definition":
                          "abs(got-ref)/(atol+rtol*abs(ref))"},
        "output_dtype": "little-endian float32",
        "serial_execution": True,
        "case_count": len(cases),
        "reference_problem_count": reference_problem_count,
        "program_summaries": program_summaries,
        "all_ok": all_ok,
        "cases": report_cases,
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "all_ok": all_ok,
        "reference_problem_count": reference_problem_count,
        "programs": program_summaries,
    }, indent=2, allow_nan=False), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
