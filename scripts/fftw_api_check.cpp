// Small direct-DFT checks for the public helper (not exercised by the hot CLI).
#include "melspec.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    for (int n : {1, 2, 4, 8, 32, 128}) {
        std::vector<double> input(n);
        for (int i = 0; i < n; ++i)
            input[i] = 0.5 * std::sin(i * 0.173) + 0.25 * std::cos(i * 0.59);
        std::vector<std::complex<double>> got;
        compute_rfft(input, got);
        if (got.size() != static_cast<size_t>(n / 2 + 1)) return 1;
        for (int k = 0; k <= n / 2; ++k) {
            std::complex<double> ref{};
            for (int i = 0; i < n; ++i) {
                double angle = -2.0 * std::acos(-1.0) * k * i / n;
                ref += input[i] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            if (std::abs(got[k] - ref) > 1e-10 * (1.0 + std::abs(ref))) return 1;
        }
        std::cout << "compute_rfft n=" << n << " PASS\n";
    }
    try {
        std::vector<std::complex<double>> got;
        compute_rfft({}, got);
        return 1;
    } catch (const std::runtime_error&) {
        std::cout << "compute_rfft empty input rejected PASS\n";
    }
}
