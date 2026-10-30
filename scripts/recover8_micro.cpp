// 八帧 AVX512F recovery microbenchmark；不作为正式 Melspec 程序链接。
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>

struct alignas(64) Lane8 {
    std::array<double, 8> values;
};

struct RecoverBuffer {
    std::vector<Lane8> re, im, power;
    explicit RecoverBuffer(int n) : re(n), im(n), power(n + 1) {}
};

static inline __m512d load8(const Lane8& value) {
    return _mm512_load_pd(value.values.data());
}

static inline void store8(Lane8& destination, __m512d value) {
    _mm512_store_pd(destination.values.data(), value);
}

// A：与 src/melspec.cpp 的 recover_power_batch8 逐 k 公式一致。
static void recover_power_batch8_current(RecoverBuffer& a,
                                         const std::vector<double>& recovery_re,
                                         const std::vector<double>& recovery_im,
                                         int support_first, int support_last) {
    const int half = static_cast<int>(a.re.size());
    auto dc = _mm512_add_pd(load8(a.re[0]), load8(a.im[0]));
    auto nyquist = _mm512_sub_pd(load8(a.re[0]), load8(a.im[0]));
    store8(a.power[0], _mm512_mul_pd(dc, dc));
    store8(a.power[half], _mm512_mul_pd(nyquist, nyquist));
    const auto scale = _mm512_set1_pd(0.5);
    const int begin = std::max(1, support_first);
    const int end = std::min(half - 1, support_last - 1);
    for (int k = begin; k <= end; ++k) {
        auto ar = load8(a.re[k]), ai = load8(a.im[k]);
        auto br = load8(a.re[half - k]);
        auto bi = _mm512_sub_pd(_mm512_setzero_pd(), load8(a.im[half - k]));
        auto rr = _mm512_set1_pd(recovery_re[k]);
        auto ri = _mm512_set1_pd(recovery_im[k]);
        auto dr = _mm512_sub_pd(ar, br);
        auto di = _mm512_sub_pd(ai, bi);
        auto xr = _mm512_fmadd_pd(rr, dr, _mm512_mul_pd(scale, _mm512_add_pd(ar, br)));
        xr = _mm512_fnmadd_pd(ri, di, xr);
        auto xi = _mm512_fmadd_pd(ri, dr, _mm512_mul_pd(scale, _mm512_add_pd(ai, bi)));
        xi = _mm512_fmadd_pd(rr, di, xi);
        store8(a.power[k], _mm512_fmadd_pd(xr, xr, _mm512_mul_pd(xi, xi)));
    }
}

struct SupportRange {
    std::string name;
    int first;
    int last;
};

// k 与 half-k 使用同一组复数输入，但 recovery 系数仍分别读取。
static void recover_power_batch8_pair(RecoverBuffer& a,
                                      const std::vector<double>& recovery_re,
                                      const std::vector<double>& recovery_im,
                                      int support_first, int support_last) {
    const int half = static_cast<int>(a.re.size());
    const auto re0 = load8(a.re[0]);
    const auto im0 = load8(a.im[0]);
    const auto dc = _mm512_add_pd(re0, im0);
    const auto nyquist = _mm512_sub_pd(re0, im0);
    store8(a.power[0], _mm512_mul_pd(dc, dc));
    store8(a.power[half], _mm512_mul_pd(nyquist, nyquist));
    const auto scale = _mm512_set1_pd(0.5);
    const int begin = std::max(1, support_first);
    const int end = std::min(half - 1, support_last - 1);
    for (int k = begin; k <= end; ++k) {
        const int mirror = half - k;
        if (k > mirror && mirror >= begin)
            continue;  // 已在处理 mirror 时成对写出。

        const auto ar = load8(a.re[k]);
        const auto ai = load8(a.im[k]);
        const auto br = load8(a.re[mirror]);
        const auto bi = _mm512_sub_pd(_mm512_setzero_pd(), load8(a.im[mirror]));
        const auto rr = _mm512_set1_pd(recovery_re[k]);
        const auto ri = _mm512_set1_pd(recovery_im[k]);
        const auto dr = _mm512_sub_pd(ar, br);
        const auto di = _mm512_sub_pd(ai, bi);
        const auto base_r = _mm512_mul_pd(scale, _mm512_add_pd(ar, br));
        const auto base_i = _mm512_mul_pd(scale, _mm512_add_pd(ai, bi));
        auto xr = _mm512_fmadd_pd(rr, dr, base_r);
        xr = _mm512_fnmadd_pd(ri, di, xr);
        auto xi = _mm512_fmadd_pd(ri, dr, base_i);
        xi = _mm512_fmadd_pd(rr, di, xi);
        store8(a.power[k], _mm512_fmadd_pd(xr, xr, _mm512_mul_pd(xi, xi)));

        if (k < mirror && mirror <= end) {
            const auto rr2 = _mm512_set1_pd(recovery_re[mirror]);
            const auto ri2 = _mm512_set1_pd(recovery_im[mirror]);
            // mirror 的 dr 为 -dr，di 仍为 di，虚部公共项为 -base_i。
            auto xr2 = _mm512_fnmadd_pd(rr2, dr, base_r);
            xr2 = _mm512_fnmadd_pd(ri2, di, xr2);
            auto xi2 = _mm512_fnmadd_pd(ri2, dr, _mm512_sub_pd(_mm512_setzero_pd(), base_i));
            xi2 = _mm512_fmadd_pd(rr2, di, xi2);
            store8(a.power[mirror], _mm512_fmadd_pd(xr2, xr2, _mm512_mul_pd(xi2, xi2)));
        }
    }
}

using Recover = void (*)(RecoverBuffer&, const std::vector<double>&,
                         const std::vector<double>&, int, int);

static void scalar_reference(const RecoverBuffer& input,
                             const std::vector<double>& recovery_re,
                             const std::vector<double>& recovery_im,
                             int support_first, int support_last,
                             std::vector<Lane8>& expected) {
    const int half = static_cast<int>(input.re.size());
    for (auto& lane : expected)
        lane.values.fill(0.0);
    const int begin = std::max(1, support_first);
    const int end = std::min(half - 1, support_last - 1);
    for (int lane = 0; lane < 8; ++lane) {
        const double dc = input.re[0].values[lane] + input.im[0].values[lane];
        const double nyquist = input.re[0].values[lane] - input.im[0].values[lane];
        expected[0].values[lane] = dc * dc;
        expected[half].values[lane] = nyquist * nyquist;
        for (int k = begin; k <= end; ++k) {
            const int mirror = half - k;
            const double ar = input.re[k].values[lane];
            const double ai = input.im[k].values[lane];
            const double br = input.re[mirror].values[lane];
            const double bi = -input.im[mirror].values[lane];
            const double dr = ar - br;
            const double di = ai - bi;
            const double xr = 0.5 * (ar + br) + recovery_re[k] * dr
                - recovery_im[k] * di;
            const double xi = 0.5 * (ai + bi) + recovery_im[k] * dr
                + recovery_re[k] * di;
            expected[k].values[lane] = xr * xr + xi * xi;
        }
    }
}

static double validate(Recover recover, RecoverBuffer& a,
                       const std::vector<double>& recovery_re,
                       const std::vector<double>& recovery_im,
                       const SupportRange& range,
                       const std::vector<Lane8>& expected) {
    for (auto& lane : a.power)
        lane.values.fill(0.0);
    recover(a, recovery_re, recovery_im, range.first, range.last);
    double max_error = 0.0;
    double scale = 1.0;
    for (size_t k = 0; k < expected.size(); ++k) {
        for (int lane = 0; lane < 8; ++lane) {
            const double reference = expected[k].values[lane];
            const double error = std::abs(reference - a.power[k].values[lane]);
            if (!std::isfinite(error))
                return std::numeric_limits<double>::infinity();
            max_error = std::max(max_error, error);
            scale = std::max(scale, std::abs(reference));
        }
    }
    return max_error / scale;
}

static std::pair<long long, long long> pair_and_single_counts(int half,
                                                               const SupportRange& range) {
    const int begin = std::max(1, range.first);
    const int end = std::min(half - 1, range.last - 1);
    long long pairs = 0;
    long long singles = 0;
    for (int k = begin; k <= end; ++k) {
        const int mirror = half - k;
        if (k < mirror && mirror <= end)
            ++pairs;
        else if (!(k > mirror && mirror >= begin))
            ++singles;
    }
    return {pairs, singles};
}

static long long read_bytes_per_batch(int half, const SupportRange& range, bool paired) {
    const auto counts = pair_and_single_counts(half, range);
    const long long pairs = paired ? counts.first : 0;
    const long long singles = paired ? counts.second : 2 * counts.first + counts.second;
    // re/im[0] 各读一次；普通 bin 为四个 64B 向量加两个 8B 系数。
    return 2 * 64 + pairs * (4 * 64 + 4 * 8) + singles * (4 * 64 + 2 * 8);
}

using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

#endif

int main(int argc, char** argv) {
#if !defined(__AVX512F__)
    (void)argc;
    (void)argv;
    std::cout << "AVX512F is required; recover8 benchmark was not built.\n";
    return 0;
#else
    int n_fft = 8192;
    int frames = 20000;
    int repeats = 5;
    int custom_first = -1;
    int custom_last = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc)
            return 2;
        const int value = std::atoi(argv[++i]);
        if (key == "--nfft") n_fft = value;
        else if (key == "--frames") frames = value;
        else if (key == "--repeats") repeats = value;
        else if (key == "--first") custom_first = value;
        else if (key == "--last") custom_last = value;
        else return 2;
    }
    if (n_fft < 4 || (n_fft & (n_fft - 1)) != 0 || frames < 8 || frames % 8 != 0
        || repeats < 1 || (custom_first < 0) != (custom_last < 0)) {
        std::cerr << "usage: recover8_micro [--nfft POW2] [--frames MULTIPLE_OF_8]"
                     " [--repeats N] [--first K --last K]\n";
        return 2;
    }

    const int half = n_fft / 2;
    std::vector<SupportRange> ranges;
    if (custom_first >= 0) {
        ranges.push_back({"custom", custom_first, custom_last});
    } else {
        // 后两组近似官方 default 与 narrow_band 的并集支持区间。
        const int last46 = std::min(half, std::max(2, (46 * half + 50) / 100));
        const int first17 = std::max(1, half / 160);
        const int last17 = std::min(half, std::max(first17 + 1, (half + 5) / 6));
        ranges = {{"full", 1, half}, {"support46", 1, last46},
                  {"support17", first17, last17}};
    }
    for (const auto& range : ranges) {
        if (range.first < 0 || range.first > range.last || range.last > half + 1) {
            std::cerr << "support range must satisfy 0 <= first <= last <= nfft/2+1\n";
            return 2;
        }
    }

    RecoverBuffer a(half);
    std::vector<double> recovery_re(half + 1), recovery_im(half + 1);
    for (int k = 1; k < half; ++k) {
        const double angle = -2.0 * M_PI * k / static_cast<double>(n_fft);
        recovery_re[k] = 0.5 * std::sin(angle);
        recovery_im[k] = -0.5 * std::cos(angle);
    }
    for (int k = 0; k < half; ++k) {
        for (int lane = 0; lane < 8; ++lane) {
            a.re[k].values[lane] = 0.25 + 0.5 * std::sin(0.017 * k + 0.13 * lane)
                + 0.1 * std::cos(0.071 * k - 0.07 * lane);
            a.im[k].values[lane] = 0.125 * std::cos(0.023 * k + 0.11 * lane)
                - 0.0625 * std::sin(0.047 * k + 0.03 * lane);
        }
    }

    std::vector<Lane8> expected(half + 1);
    for (const auto& range : ranges) {
        scalar_reference(a, recovery_re, recovery_im, range.first, range.last, expected);
        const double error_a = validate(recover_power_batch8_current, a, recovery_re, recovery_im,
                                        range, expected);
        const double error_b = validate(recover_power_batch8_pair, a, recovery_re,
                                        recovery_im, range, expected);
        const auto counts = pair_and_single_counts(half, range);
        std::cout << "validation range=" << range.name << " first=" << range.first
                  << " last=" << range.last << " pairs=" << counts.first
                  << " singles=" << counts.second << std::setprecision(12)
                  << " A_scaled_error=" << error_a << " B_scaled_error=" << error_b << '\n';
        if (!(error_a <= 1e-12) || !(error_b <= 1e-12)) {
            std::cerr << "recovery numerical validation failed\n";
            return 1;
        }
    }

    std::cout << "mode,range,first,last,repeat,recover_ns_per_frame,"
                 "read_bytes_per_frame,estimated_total_read_bytes,checksum\n";
    for (const auto& range : ranges) {
        const Recover modes[] = {recover_power_batch8_current, recover_power_batch8_pair};
        const char* names[] = {"A_current", "B_paired"};
        for (int mode = 0; mode < 2; ++mode) {
            const long long bytes_per_batch = read_bytes_per_batch(half, range, mode == 1);
            for (int rep = 0; rep < repeats; ++rep) {
                double elapsed = 0.0;
                volatile double checksum = 0.0;
                for (int frame = 0; frame < frames; frame += 8) {
                    const auto begin = Clock::now();
                    modes[mode](a, recovery_re, recovery_im, range.first, range.last);
                    const auto end = Clock::now();
                    elapsed += seconds(begin, end);
                    const int k = (frame / 8) % (half + 1);
                    for (int lane = 0; lane < 8; ++lane)
                        checksum += a.power[k].values[lane];
                }
                const double bytes_per_frame = static_cast<double>(bytes_per_batch) / 8.0;
                std::cout << names[mode] << ',' << range.name << ',' << range.first << ','
                          << range.last << ',' << rep << ',' << std::setprecision(12)
                          << elapsed * 1e9 / frames << ',' << bytes_per_frame << ','
                          << static_cast<long long>(bytes_per_frame * frames) << ','
                          << checksum << '\n';
            }
        }
    }
    return 0;
#endif
}
