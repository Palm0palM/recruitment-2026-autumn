#include "melspec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
    const int specs[][3] = {{1, 1, 33}, {2, 3, 31}, {128, 7, 47},
                           {1024, 13, 159}, {4096, 17, 97},
                           {512, 513, 49}, {256, 3, 4127}};
    for (const auto& spec : specs) {
        const int n = spec[0], hop = spec[1], frames = spec[2], mels = 21;
        std::vector<double> y(n + hop * (frames - 1));
        for (size_t i = 0; i < y.size(); ++i)
            y[i] = std::sin(i * 0.13) + 0.00123456789 * std::cos(i * 0.037);
        std::vector<double> expected;
        compute_melspectrogram(y, 32000, n, hop, mels, 73, 15137, expected);
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<double> output(mode == 0 ? 1 : expected.size() + 31, -713.25);
            if (mode == 2) output.reserve(output.size() + 100000);
            compute_melspectrogram(y, 32000, n, hop, mels, 73, 15137, output);
            if (output != expected) throw std::runtime_error("reused output mismatch");
            if (!std::all_of(output.begin(), output.end(), [](double x) { return std::isfinite(x); }))
                throw std::runtime_error("nonfinite output");
        }
    }
    std::puts("PASS 21 reused-vector / arbitrary-double interface cases");
}
