// Batch=8 bit-reversal/window packing microbenchmark.
// Build only for an AVX-512 target, for example:
//   g++ -std=c++17 -O3 -march=icelake-server scripts/pack8_micro.cpp -o pack8_micro
// Run (on a suitable machine):
//   ./pack8_micro --n 4096 --batches 2048 --repeats 5
// n is the complex FFT size; each input frame and the Hann window contain 2*n doubles.

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBatch = 8;
constexpr int kHop = 128;

struct FFTPlan {
    std::vector<int> reversed;

    explicit FFTPlan(int n) : reversed(static_cast<std::size_t>(n)) {
        for (int i = 0, j = 0; i < n; ++i) {
            reversed[static_cast<std::size_t>(i)] = j;
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
        }
    }
};

struct alignas(64) Lane8 {
    std::array<double, kBatch> values;
};

struct Batch8Buffer {
    std::vector<Lane8> re;
    std::vector<Lane8> im;

    explicit Batch8Buffer(int n)
        : re(static_cast<std::size_t>(n)), im(static_cast<std::size_t>(n)) {}
};

inline void store8(Lane8& destination, __m512d value) {
    _mm512_store_pd(destination.values.data(), value);
}

// A: the current data flow. For each complex-bin index i, gather the even and
// odd samples from eight independent time-frame streams with _mm512_setr_pd.
__attribute__((noinline)) void pack_a_setr(
    const FFTPlan& plan, const double* y, const double* window,
    std::size_t start, Batch8Buffer& out) {
    const int n = static_cast<int>(plan.reversed.size());
    for (int i = 0; i < n; ++i) {
        const int p = plan.reversed[static_cast<std::size_t>(i)];
        const double we = window[2 * i];
        const double wo = window[2 * i + 1];
        const std::size_t even = start + static_cast<std::size_t>(2 * i);
        store8(out.re[static_cast<std::size_t>(p)], _mm512_setr_pd(
            y[even] * we,
            y[even + kHop] * we,
            y[even + 2 * kHop] * we,
            y[even + 3 * kHop] * we,
            y[even + 4 * kHop] * we,
            y[even + 5 * kHop] * we,
            y[even + 6 * kHop] * we,
            y[even + 7 * kHop] * we));
        const std::size_t odd = even + 1;
        store8(out.im[static_cast<std::size_t>(p)], _mm512_setr_pd(
            y[odd] * wo,
            y[odd + kHop] * wo,
            y[odd + 2 * kHop] * wo,
            y[odd + 3 * kHop] * wo,
            y[odd + 4 * kHop] * wo,
            y[odd + 5 * kHop] * wo,
            y[odd + 6 * kHop] * wo,
            y[odd + 7 * kHop] * wo));
    }
}

// Transpose eight rows of eight doubles. Each output column j contains row 0
// through row 7 at sample j. The three stages are:
//   1. unpacklo/unpackhi: interleave pairs of frames within 128-bit lanes;
//   2. shuffle_f64x2(0x88/0xdd): join frame pairs 0/1 with 2/3 and 4/5 with 6/7;
//   3. shuffle_f64x2(0x88/0xdd): join the lower and upper four-frame groups.
// This preserves lane order: output[j] = {row0[j], row1[j], ..., row7[j]}.
inline void transpose8x8(const __m512d rows[8], __m512d columns[8]) {
    __m512d pair03[4];
    __m512d pair47[4];

    // Keep the pair arrays indexed by the two sample columns they contain.
    const __m512d t01lo = _mm512_unpacklo_pd(rows[0], rows[1]);
    const __m512d t01hi = _mm512_unpackhi_pd(rows[0], rows[1]);
    const __m512d t23lo = _mm512_unpacklo_pd(rows[2], rows[3]);
    const __m512d t23hi = _mm512_unpackhi_pd(rows[2], rows[3]);
    const __m512d t45lo = _mm512_unpacklo_pd(rows[4], rows[5]);
    const __m512d t45hi = _mm512_unpackhi_pd(rows[4], rows[5]);
    const __m512d t67lo = _mm512_unpacklo_pd(rows[6], rows[7]);
    const __m512d t67hi = _mm512_unpackhi_pd(rows[6], rows[7]);

    // Low/high sample-column pairs for frames 0..3 and 4..7.
    pair03[0] = _mm512_shuffle_f64x2(t01lo, t23lo, 0x88); // columns 0,4
    pair03[1] = _mm512_shuffle_f64x2(t01hi, t23hi, 0x88); // columns 1,5
    pair03[2] = _mm512_shuffle_f64x2(t01lo, t23lo, 0xdd); // columns 2,6
    pair03[3] = _mm512_shuffle_f64x2(t01hi, t23hi, 0xdd); // columns 3,7
    pair47[0] = _mm512_shuffle_f64x2(t45lo, t67lo, 0x88);
    pair47[1] = _mm512_shuffle_f64x2(t45hi, t67hi, 0x88);
    pair47[2] = _mm512_shuffle_f64x2(t45lo, t67lo, 0xdd);
    pair47[3] = _mm512_shuffle_f64x2(t45hi, t67hi, 0xdd);

    columns[0] = _mm512_shuffle_f64x2(pair03[0], pair47[0], 0x88);
    columns[4] = _mm512_shuffle_f64x2(pair03[0], pair47[0], 0xdd);
    columns[1] = _mm512_shuffle_f64x2(pair03[1], pair47[1], 0x88);
    columns[5] = _mm512_shuffle_f64x2(pair03[1], pair47[1], 0xdd);
    columns[2] = _mm512_shuffle_f64x2(pair03[2], pair47[2], 0x88);
    columns[6] = _mm512_shuffle_f64x2(pair03[2], pair47[2], 0xdd);
    columns[3] = _mm512_shuffle_f64x2(pair03[3], pair47[3], 0x88);
    columns[7] = _mm512_shuffle_f64x2(pair03[3], pair47[3], 0xdd);
}

// B: process eight consecutive complex-bin indices at a time. Each frame uses
// two contiguous 8-double loads for 16 audio samples and two for Hann weights.
// After packed multiplication, permutex2var deinterleaves even/odd samples into
// one row per frame. transpose8x8 then turns eight frame rows into eight Lane8
// vectors, which are scattered only at the final bit-reversed stores.
__attribute__((noinline)) void pack_b_contiguous(
    const FFTPlan& plan, const double* y, const double* window,
    std::size_t start, Batch8Buffer& out) {
    const int n = static_cast<int>(plan.reversed.size());
    const __m512i even_indices = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i odd_indices = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

    for (int base = 0; base < n; base += 8) {
        __m512d even_rows[8];
        __m512d odd_rows[8];
        for (int lane = 0; lane < 8; ++lane) {
            const double* samples = y + start + static_cast<std::size_t>(lane) * kHop
                + static_cast<std::size_t>(2 * base);
            const double* weights = window + 2 * base;
            const __m512d x0 = _mm512_loadu_pd(samples);
            const __m512d x1 = _mm512_loadu_pd(samples + 8);
            const __m512d w0 = _mm512_loadu_pd(weights);
            const __m512d w1 = _mm512_loadu_pd(weights + 8);
            const __m512d weighted0 = _mm512_mul_pd(x0, w0);
            const __m512d weighted1 = _mm512_mul_pd(x1, w1);
            even_rows[static_cast<std::size_t>(lane)] =
                _mm512_permutex2var_pd(weighted0, even_indices, weighted1);
            odd_rows[static_cast<std::size_t>(lane)] =
                _mm512_permutex2var_pd(weighted0, odd_indices, weighted1);
        }

        __m512d columns[8];
        transpose8x8(even_rows, columns);
        for (int j = 0; j < 8; ++j) {
            const int p = plan.reversed[static_cast<std::size_t>(base + j)];
            store8(out.re[static_cast<std::size_t>(p)], columns[static_cast<std::size_t>(j)]);
        }
        transpose8x8(odd_rows, columns);
        for (int j = 0; j < 8; ++j) {
            const int p = plan.reversed[static_cast<std::size_t>(base + j)];
            store8(out.im[static_cast<std::size_t>(p)], columns[static_cast<std::size_t>(j)]);
        }
    }
}

struct Options {
    int n = 4096;
    int batches = 2048;
    int repeats = 5;
};

int positive_int(const char* text, const std::string& option) {
    try {
        std::size_t used = 0;
        const long long value = std::stoll(text, &used);
        if (text[used] != '\0' || value < 1 || value > std::numeric_limits<int>::max())
            throw std::invalid_argument(option);
        return static_cast<int>(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + option + ": " + text);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const int value = positive_int(argv[++i], key);
        if (key == "--n") options.n = value;
        else if (key == "--batches") options.batches = value;
        else if (key == "--repeats") options.repeats = value;
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (options.n != 4096 && options.n != 8192)
        throw std::invalid_argument("--n must be 4096 or 8192");
    return options;
}

std::size_t input_size(int n, int batches) {
    // Last batch starts at (batches-1)*8*hop; its last frame starts 7*hop later
    // and reads exactly 2*n samples. All load endpoints are therefore in range.
    return (static_cast<std::size_t>(batches - 1) * 8 + 7) * kHop
        + static_cast<std::size_t>(2 * n);
}

struct ValidationResult {
    double max_abs_re = 0.0;
    double max_abs_im = 0.0;
    bool pass = true;
};

ValidationResult validate_batch(const FFTPlan& plan, const std::vector<double>& y,
                                const std::vector<double>& window, int batch,
                                Batch8Buffer& a, Batch8Buffer& b) {
    const std::size_t start = static_cast<std::size_t>(batch) * 8 * kHop;
    pack_a_setr(plan, y.data(), window.data(), start, a);
    pack_b_contiguous(plan, y.data(), window.data(), start, b);

    ValidationResult result;
    const std::size_t n = plan.reversed.size();
    for (std::size_t p = 0; p < n; ++p) {
        for (int lane = 0; lane < 8; ++lane) {
            const double ar = a.re[p].values[static_cast<std::size_t>(lane)];
            const double br = b.re[p].values[static_cast<std::size_t>(lane)];
            const double ai = a.im[p].values[static_cast<std::size_t>(lane)];
            const double bi = b.im[p].values[static_cast<std::size_t>(lane)];
            const double error_re = std::abs(ar - br);
            const double error_im = std::abs(ai - bi);
            result.max_abs_re = std::max(result.max_abs_re, error_re);
            result.max_abs_im = std::max(result.max_abs_im, error_im);
            const double tolerance_re = 1e-13 * std::max({1.0, std::abs(ar), std::abs(br)});
            const double tolerance_im = 1e-13 * std::max({1.0, std::abs(ai), std::abs(bi)});
            if (!std::isfinite(error_re) || !std::isfinite(error_im)
                || error_re > tolerance_re || error_im > tolerance_im)
                result.pass = false;
        }
    }
    return result;
}

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

template <typename Pack>
double measure(Pack pack, const FFTPlan& plan, const std::vector<double>& y,
               const std::vector<double>& window, int batches, Batch8Buffer& out,
               volatile double& checksum) {
    const auto begin = Clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const std::size_t start = static_cast<std::size_t>(batch) * 8 * kHop;
        pack(plan, y.data(), window.data(), start, out);
    }
    const auto end = Clock::now();

    // Observe every output element of the final batch, outside the timed region.
    double sum = 0.0;
    for (std::size_t p = 0; p < plan.reversed.size(); ++p) {
        for (int lane = 0; lane < 8; ++lane) {
            sum += out.re[p].values[static_cast<std::size_t>(lane)];
            sum += out.im[p].values[static_cast<std::size_t>(lane)];
        }
    }
    checksum += sum;
    return seconds(begin, end);
}

} // namespace

int main(int argc, char** argv) {
#if !defined(__AVX512F__)
    (void)argc;
    (void)argv;
    std::cerr << "pack8_micro requires AVX512F; compile with -march=icelake-server\n";
    return 2;
#else
    try {
        const Options options = parse_options(argc, argv);
        const int n = options.n;
        FFTPlan plan(n);
        std::vector<double> window(static_cast<std::size_t>(2 * n));
        constexpr double pi = 3.141592653589793238462643383279502884;
        for (int i = 0; i < 2 * n; ++i) {
            window[static_cast<std::size_t>(i)] =
                0.5 * (1.0 - std::cos(2.0 * pi * i / static_cast<double>(2 * n)));
        }

        std::vector<double> y(input_size(n, options.batches));
        for (std::size_t i = 0; i < y.size(); ++i) {
            // Deterministic, non-periodic-enough values expose lane/order errors.
            const std::uint64_t x = static_cast<std::uint64_t>(i) * 1099511628211ULL
                + 1469598103934665603ULL;
            y[i] = static_cast<double>((x >> 11) & 0xfffffULL) / 1048576.0 - 0.5
                + 0.125 * std::sin(0.0013 * static_cast<double>(i));
        }

        Batch8Buffer a(n);
        Batch8Buffer b(n);
        double max_error_re = 0.0;
        double max_error_im = 0.0;
        const std::array<int, 3> validation_batches = {
            0, options.batches / 2, options.batches - 1
        };
        for (const int batch : validation_batches) {
            const ValidationResult result = validate_batch(plan, y, window, batch, a, b);
            max_error_re = std::max(max_error_re, result.max_abs_re);
            max_error_im = std::max(max_error_im, result.max_abs_im);
            if (!result.pass) {
                std::cerr << "validation failed at batch " << batch << '\n';
                return 1;
            }
        }
        std::cout << "validation=PASS n=" << n << " batches=" << options.batches
                  << " hop=" << kHop << " max_abs_re=" << std::setprecision(12)
                  << max_error_re << " max_abs_im=" << max_error_im << '\n';
        std::cout << "mode,n,repeat,batches,pack_ns_per_frame,checksum\n";

        volatile double checksum_a = 0.0;
        volatile double checksum_b = 0.0;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            double elapsed_a = 0.0;
            double elapsed_b = 0.0;
            // Alternate measurement order to reduce systematic drift.
            if ((repeat & 1) == 0) {
                elapsed_a = measure(pack_a_setr, plan, y, window, options.batches, a, checksum_a);
                elapsed_b = measure(pack_b_contiguous, plan, y, window, options.batches, b, checksum_b);
            } else {
                elapsed_b = measure(pack_b_contiguous, plan, y, window, options.batches, b, checksum_b);
                elapsed_a = measure(pack_a_setr, plan, y, window, options.batches, a, checksum_a);
            }
            const double frames = static_cast<double>(options.batches) * 8.0;
            std::cout << "A_setr," << n << ',' << repeat << ',' << options.batches << ','
                      << std::setprecision(12) << elapsed_a * 1e9 / frames << ','
                      << checksum_a << '\n';
            std::cout << "B_contiguous_transpose," << n << ',' << repeat << ','
                      << options.batches << ',' << std::setprecision(12)
                      << elapsed_b * 1e9 / frames << ',' << checksum_b << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n'
                  << "usage: pack8_micro [--n 4096|8192] [--batches N] [--repeats N]\n";
        return 2;
    }
#endif
}
