#include "fft128_codelet.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5000;
    Codelet128Plan plan;
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    alignas(64) double re[128], im[128];
    std::vector<std::complex<double>> reference(128);
    for (int q = 0; q < 128; ++q) {
        re[q] = dist(rng); im[q] = dist(rng);
        reference[q] = {re[q], im[q]};
    }
    auto original_re = std::vector<double>(re, re + 128);
    auto original_im = std::vector<double>(im, im + 128);
    plan.apply(re, im);
    double max_error = 0;
    for (int k = 0; k < 128; ++k) {
        std::complex<double> expected = 0;
        for (int q = 0; q < 128; ++q) {
            double ang = -2.0 * M_PI * k * q / 128.0;
            expected += reference[q] * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        max_error = std::max(max_error, std::abs(expected -
            std::complex<double>(re[plan.perm[k]], im[plan.perm[k]])));
    }
    std::printf("codelet direct-DFT max_abs_err=%.9g\n", max_error);
    if (max_error > 1e-9) return 2;

    for (int groups : {33, 65}) {
        std::vector<double> input_re(groups * 128), input_im(groups * 128);
        for (int i = 0; i < groups * 128; ++i) {
            input_re[i] = dist(rng); input_im[i] = dist(rng);
        }
        auto re = input_re, im = input_im;
        volatile double checksum = 0;
        double ms = 0;
        for (int it = 0; it < iterations; ++it) {
            auto begin = std::chrono::steady_clock::now();
            for (int r = 0; r < groups; ++r) plan.apply(re.data() + r * 128, im.data() + r * 128);
            ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            checksum += re[0];
            re = input_re; im = input_im;  // 不计时，防止重复 FFT 数值增长。
        }
        std::printf("groups=%d codelet: %.3f ns/FFT, checksum=%.6f\n",
                    groups, ms * 1e6 / (iterations * groups),
                    static_cast<double>(checksum));
    }
    return 0;
}
