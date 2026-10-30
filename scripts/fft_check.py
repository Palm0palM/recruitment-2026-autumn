#!/usr/bin/env python3
"""直接检查公开 compute_rfft：全部二次幂尺寸、随机/冲激/正弦输入。

NumPy 只用于测试参考，不参与被测 C++ 实现。构建产物仅写 tmp/fft-check/。
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np

from bench import ROOT


def main():
    work = ROOT / "tmp/fft-check"
    work.mkdir(parents=True, exist_ok=True)
    wrapper = work / "wrapper.cpp"
    wrapper.write_text(r'''#include "melspec.h"
#include <vector>
extern "C" void checked_rfft(int n, const double* input, double* output) {
    std::vector<double> frame(input, input + n);
    std::vector<std::complex<double>> spectrum;
    compute_rfft(frame, spectrum);
    for (int k = 0; k <= n / 2; ++k) {
        output[2 * k] = spectrum[k].real();
        output[2 * k + 1] = spectrum[k].imag();
    }
}
''')
    library = work / "librfft.so"
    subprocess.run(["g++", "-std=c++17", "-O3", "-march=native", "-pthread", "-shared", "-fPIC",
                    "-I" + str(ROOT / "include"), str(wrapper), str(ROOT / "src/melspec.cpp"),
                    "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.checked_rfft.argtypes = [ctypes.c_int, pointer, pointer]
    rng = np.random.default_rng(20261030)
    for exponent in range(15):
        n = 1 << exponent
        signals = {
            "random": rng.standard_normal(n),
            "impulse": np.eye(1, n, n // 3).ravel(),
            "sine": np.sin(np.arange(n) * (2 * np.pi * 5.25 / n)),
        }
        for kind, frame in signals.items():
            frame = np.ascontiguousarray(frame, dtype=np.float64)
            got = np.empty(2 * (n // 2 + 1), dtype=np.float64)
            lib.checked_rfft(n, frame.ctypes.data_as(pointer), got.ctypes.data_as(pointer))
            got = got.view(np.complex128)
            expected = np.fft.rfft(frame)
            error = float(np.max(np.abs(got - expected)))
            scale = max(1.0, float(np.max(np.abs(expected))))
            if error > 1e-9 * scale:
                raise SystemExit(f"FAIL n={n} {kind}: max_abs_err={error:.3e}, scale={scale:.3e}")
        print(f"n={n:5d}: random/impulse/sine PASS", flush=True)
    print("All public rfft checks passed.")


if __name__ == "__main__":
    main()
