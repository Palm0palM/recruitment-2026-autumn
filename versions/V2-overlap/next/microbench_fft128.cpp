#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#ifdef HAVE_FFTW
#include <fftw3.h>
#endif

struct FFTPlan {
    std::vector<int> reversed;
    std::vector<std::complex<double>> twiddles;
    explicit FFTPlan(int n) : reversed(n), twiddles(n) {
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            reversed[i] = j;
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * M_PI / len;
            std::complex<double> wlen(std::cos(ang), std::sin(ang)), w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                twiddles[len / 2 + j] = w;
                w *= wlen;
            }
        }
    }
    void apply(std::vector<std::complex<double>>& a) const {
        const int n = a.size();
        for (int len = 2; len <= n; len <<= 1)
            for (int i = 0; i < n; i += len)
                for (int j = 0; j < len / 2; ++j) {
                    auto u = a[i + j];
                    auto v = a[i + j + len / 2] * twiddles[len / 2 + j];
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                }
    }
};

template <class F, class R>
double time_ms(F&& fn, R&& reset, int iterations) {
    double total = 0;
    for (int i = 0; i < iterations; ++i) {
        auto begin = std::chrono::steady_clock::now();
        fn();
        total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        reset();  // 不计时，避免重复 FFT 的数值增长和输入副本污染测量。
    }
    return total;
}

int main(int argc, char** argv) {
    constexpr int S = 128;
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 2000;
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    FFTPlan plan(S);
    for (int groups : {33, 65}) {
        std::vector<std::vector<std::complex<double>>> input(
            groups, std::vector<std::complex<double>>(S));
        for (auto& row : input)
            for (auto& z : row) z = {dist(rng), dist(rng)};
        // apply() expects bit-reversed input, matching overlap_power's fused fill.
        for (auto& row : input) {
            std::vector<std::complex<double>> natural = row;
            for (int q = 0; q < S; ++q) row[plan.reversed[q]] = natural[q];
        }
        auto work = input;
        volatile double checksum = 0;
        double scalar_ms = time_ms([&] {
            for (int r = 0; r < groups; ++r) plan.apply(work[r]);
            checksum += work[0][0].real();
        }, [&] { work = input; }, iterations);
        std::printf("groups=%d scalar radix2: %.3f ns/FFT, %.3f ms total\n",
                    groups, scalar_ms * 1e6 / (iterations * groups), scalar_ms);
#ifdef HAVE_FFTW
        std::vector<std::complex<double>> batch(groups * S);
        for (int r = 0; r < groups; ++r)
            for (int q = 0; q < S; ++q) batch[r * S + q] = input[r][q];
        const auto batch_input = batch;
        auto* ptr = reinterpret_cast<fftw_complex*>(batch.data());
        int n[1] = {S};
        auto t0 = std::chrono::steady_clock::now();
        fftw_plan many = fftw_plan_many_dft(1, n, groups, ptr, n, 1, S,
                                             ptr, n, 1, S, FFTW_FORWARD,
                                             FFTW_ESTIMATE | FFTW_UNALIGNED);
        double plan_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (!many) return 2;
        double fftw_ms = time_ms([&] {
            fftw_execute(many);
            checksum += batch[0].real();
        }, [&] { batch = batch_input; }, iterations);
        std::printf("groups=%d FFTW batched: %.3f ns/FFT, %.3f ns/batch, %.3f ms total, plan %.3f ms\n",
                    groups, fftw_ms * 1e6 / (iterations * groups),
                    fftw_ms * 1e6 / iterations, fftw_ms, plan_ms);
        std::printf("groups=%d FFTW/scalar per-FFT ratio: %.4f\n", groups, fftw_ms / scalar_ms);
        fftw_destroy_plan(many);
#endif
        std::printf("groups=%d checksum=%.6f\n", groups, static_cast<double>(checksum));
    }
    std::printf("official 128-point calls: default/small=%lld, large/narrow=%lld\n",
                192397LL * 33, 192333LL * 65);
    return 0;
}
