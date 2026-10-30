#!/usr/bin/env python3
"""统一的 benchmark 入口（本地开发期 + 集群正式测量）。

参数组直接复用官方 validate.py 的 TEST_CASES，避免两边参数写不一致。

用法：
    python scripts/bench.py                          # 全量四组，3 次取最小
    python scripts/bench.py --case default --slice 5 # 截断到 5 秒，快速迭代
    python scripts/bench.py --case all --runs 5 --out tmp/bench/v3.md
    python scripts/bench.py --slice 5 --no-librosa   # 跳过 librosa 对比

注意：
    * --slice 只用于开发期快速迭代，成绩必须以全量为准（题目要求在集群上跑）。
    * 全量模式下 librosa 需要 ~25 GB 内存，内存不足会自动跳过。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_official_cases():
    """从 validate.py（不可修改）载入官方参数组。"""
    spec = importlib.util.spec_from_file_location("_validate", ROOT / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TEST_CASES


def find_data_dir() -> Path:
    candidates = [ROOT / "data", ROOT / "tmp" / "data"]
    if os.environ.get("MELSPEC_DATA"):
        candidates.insert(0, Path(os.environ["MELSPEC_DATA"]))
    for cand in candidates:
        if (cand / "audio.pcm").exists():
            return cand
    raise SystemExit(
        "找不到可用的 data/audio.pcm。请先运行 scripts/setup.sh，"
        "或用 MELSPEC_DATA=<集群数据目录> 指定。"
    )


def make_input(data_dir: Path, seconds: float | None) -> Path:
    """返回本次使用的 PCM 路径；seconds 非空时生成 tmp/ 下的截断副本。"""
    if seconds is None:
        return data_dir / "audio.pcm"
    src = data_dir / "audio.pcm"
    dst = ROOT / "tmp" / f"slice_{seconds:g}s.pcm"
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = int(seconds * 96000) * 2  # 单声道 int16
    size = want if want <= src.stat().st_size else src.stat().st_size
    if not dst.exists() or dst.stat().st_size != size:
        # 注意：shutil.copyfileobj 的第三个参数是块大小，不是字节上限，这里必须自己截断
        with open(src, "rb") as f, open(dst, "wb") as g:
            remaining = size
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                g.write(chunk)
                remaining -= len(chunk)
    return dst


def run_once(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True)
    dt = time.perf_counter() - t0
    return dt


def timed(prog: str, pcm: Path, params: dict, out_path: Path, runs: int) -> float:
    args = []
    for k, v in params.items():
        args += [f"--{k}", str(v)]
    cmd = [prog, str(pcm), str(out_path), *args]
    times = [run_once(cmd) for _ in range(runs)]
    return min(times)


def ref_librosa(pcm: Path, params: dict, runs: int):
    try:
        import librosa
    except ImportError:
        return None, None
    y = np.fromfile(pcm, dtype="<i2").astype(np.float32) / 32768.0
    kw = dict(
        sr=params["sr"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"],
        fmin=params["f_min"],
        fmax=params["f_max"],
    )
    times = []
    S = None
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            S = librosa.feature.melspectrogram(y=y, center=False, htk=True, **kw)
        except MemoryError:
            return None, None
        times.append(time.perf_counter() - t0)
    return min(times), S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program", nargs="?", default="./melspec",
                    help="被测程序（默认 ./melspec）")
    ap.add_argument("--case", default="all",
                    help="all 或某个用例名：default/small_mels/large_fft/narrow_band")
    ap.add_argument("--slice", type=float, default=None, metavar="SECONDS",
                    help="只取输入前 N 秒（开发期快速迭代，不可作为最终成绩）")
    ap.add_argument("--runs", type=int, default=3, help="每次取 N 次运行的最小值")
    ap.add_argument("--no-librosa", action="store_true", help="跳过 librosa 对比")
    ap.add_argument("--force-librosa", action="store_true",
                    help="全量模式下也跑 librosa（内存需求约 25 GB，慎用）")
    ap.add_argument("--no-check", action="store_true", help="跳过正确性校验")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=None, help="把测量摘要另存到文件")
    args = ap.parse_args()
    if args.runs < 1 or (args.slice is not None and args.slice <= 0):
        ap.error("--runs 必须 >= 1，--slice 必须 > 0")

    data_dir = find_data_dir()
    cases = load_official_cases()
    if args.case != "all":
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            raise SystemExit(f"未知用例: {args.case}")

    prog = shutil.which(args.program) or args.program
    pcm = make_input(data_dir, args.slice)
    n_samples = pcm.stat().st_size // 2
    if any(n_samples < tc["params"]["n_fft"] for tc in cases):
        ap.error("输入长度必须至少覆盖一帧 n_fft")
    mode = f"截断 {args.slice:g}s" if args.slice else "全量"
    print(f"# 程序: {prog}\n# 输入: {pcm} ({n_samples} samples, {n_samples/96000:.1f}s, {mode})\n")

    rows = []
    for tc in cases:
        p = dict(tc["params"])
        n_frames = 1 + (n_samples - p["n_fft"]) // p["hop_length"]
        shape = (p["n_mels"], n_frames)

        scratch = ROOT / "tmp" / "bench"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as td:
            out = Path(td) / "out.bin"
            t_prog = timed(prog, pcm, p, out, args.runs)
            got = np.fromfile(out, dtype="<f4")

        # librosa 只算一次：既做参考真值，也做时间基准
        use_librosa = not args.no_librosa and (args.slice is not None or args.force_librosa)
        t_lib, S = (None, None) if not use_librosa else ref_librosa(pcm, p, args.runs)

        err = None
        passed = None
        if not args.no_check:
            view = got.reshape(shape) if got.size == shape[0] * shape[1] else None
            if view is None:
                print(f"!! {tc['name']}: 输出元素数 {got.size} != 期望 {shape[0] * shape[1]}"
                      f"（shape {shape}）")
                passed = False
            elif args.slice is None:
                expected = Path(tc["expected"]).relative_to("data")
                exp = np.fromfile(data_dir / expected, dtype="<f4").reshape(shape)
                err = float(np.max(np.abs(view - exp)))
                passed = bool(np.allclose(view, exp, rtol=args.rtol, atol=args.atol))
                del exp
            elif S is not None:
                S = S[:, : shape[1]]
                err = float(np.max(np.abs(view - S)))
                passed = bool(np.allclose(view, S, rtol=args.rtol, atol=args.atol))

        speedup = (t_lib / t_prog) if t_lib else None
        rows.append(dict(name=tc["name"], p=p, t=t_prog, err=err, ok=passed,
                         t_lib=t_lib, speedup=speedup))
        if S is not None:
            del S

    def fmt_row(r) -> str:
        p = r["p"]
        err = "-" if r["err"] is None else f"{r['err']:.2e}"
        ok = "-" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
        lib = "-" if r["t_lib"] is None else f"{r['t_lib']:.3f}"
        spd = "-" if not r["speedup"] else f"{r['speedup']:.2f}x"
        return (f"{r['name']:<12}| {p['n_fft']:>6} | {p['hop_length']:>4} | {p['n_mels']:>4} | "
                f"{p['f_min']:>5.0f} | {p['f_max']:>6.0f} | {err:>10} | {ok:>4} | "
                f"{r['t']:>8.3f} | {lib:>8} | {spd:>7}")

    hdr = (f"{'case':<12}| {'n_fft':>6} | {'hop':>4} | {'mels':>4} | {'f_min':>5} | "
           f"{'f_max':>6} | {'max|err|':>10} | {'pass':>4} | {'time(s)':>8} | "
           f"{'librosa':>8} | {'speedup':>7}")
    sep = "-" * len(hdr)
    print(sep); print(hdr); print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)
    if args.slice is not None:
        print("提示：截断模式的结果仅用于开发期比较，最终成绩必须在集群上按全量测算。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tag = f" ({mode})" if args.slice else ""
        lines = [f"# bench {time.strftime('%Y-%m-%d %H:%M')} {prog}{tag}", "", hdr, sep]
        for r in rows:
            p = r["p"]
            lines.append(f"{r['name']} | n_fft={p['n_fft']} hop={p['hop_length']} "
                         f"mels={p['n_mels']} | {r['t']:.3f}s | "
                         f"err={r['err']} | ok={r['ok']} | "
                         f"librosa={r['t_lib']}")
        lines += [sep, ""]
        args.out.write_text("\n".join(lines))
        print(f"\n已写入 {args.out}")

    if any(r["ok"] is False for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
