#include "melspec.h"
#include "melspec_backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#ifndef MELSPEC_FP32_BATCH
#define MELSPEC_FP32_BATCH 0
#endif
#if MELSPEC_FP32_BATCH != 0 && MELSPEC_FP32_BATCH != 8 && MELSPEC_FP32_BATCH != 16
#error "MELSPEC_FP32_BATCH must be 0, 8, or 16"
#endif
#if MELSPEC_FP32_BATCH == 8 && !(defined(__AVX2__) && defined(__FMA__))
#error "MELSPEC_FP32_BATCH=8 requires AVX2 and FMA"
#endif
#if MELSPEC_FP32_BATCH == 16 && !defined(__AVX512F__)
#error "MELSPEC_FP32_BATCH=16 requires AVX-512F"
#endif

#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__))
#include <immintrin.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
static inline void transpose4(__m256d& a, __m256d& b, __m256d& c, __m256d& d) {
    auto ab_lo = _mm256_unpacklo_pd(a, b);
    auto ab_hi = _mm256_unpackhi_pd(a, b);
    auto cd_lo = _mm256_unpacklo_pd(c, d);
    auto cd_hi = _mm256_unpackhi_pd(c, d);
    a = _mm256_permute2f128_pd(ab_lo, cd_lo, 0x20);
    b = _mm256_permute2f128_pd(ab_hi, cd_hi, 0x20);
    c = _mm256_permute2f128_pd(ab_lo, cd_lo, 0x31);
    d = _mm256_permute2f128_pd(ab_hi, cd_hi, 0x31);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// HTK mel scale
// ─────────────────────────────────────────────────────────────────────────────

double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cooley-Tukey iterative FFT (in-place, complex input/output)
// ─────────────────────────────────────────────────────────────────────────────

struct ComplexBuffer {
    std::vector<double> re, im;
    explicit ComplexBuffer(int n) : re(n), im(n) {}
};

struct Radix4Stage {
    int len;
    std::vector<double> w1re, w1im, w2re, w2im, w3re, w3im;
    explicit Radix4Stage(int n)
        : len(n), w1re(n / 4), w1im(n / 4), w2re(n / 4), w2im(n / 4),
          w3re(n / 4), w3im(n / 4) {}
};

struct FFTPlan {
    std::vector<int> reversed;
    std::vector<Radix4Stage> stages;
    bool initial_radix2;

    explicit FFTPlan(int n) : reversed(n), initial_radix2((n & 0xaaaaaaaa) != 0) {
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            reversed[i] = j;
        }
        std::vector<std::complex<double>> twiddles(n);
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * M_PI / static_cast<double>(len);
            std::complex<double> wlen(std::cos(ang), std::sin(ang));
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                twiddles[len / 2 + j] = w;
                w *= wlen;
            }
        }
        int len = initial_radix2 ? 8 : 4;
        for (; len <= n; len <<= 2) {
            Radix4Stage stage(len);
            for (int j = 0; j < len / 4; ++j) {
                auto w1 = twiddles[len / 2 + j];
                auto w2 = twiddles[len / 2 + 2 * j];
                auto w3 = w1 * w2;
                stage.w1re[j] = w1.real();
                stage.w1im[j] = w1.imag();
                stage.w2re[j] = w2.real();
                stage.w2im[j] = w2.imag();
                stage.w3re[j] = w3.real();
                stage.w3im[j] = w3.imag();
            }
            stages.push_back(std::move(stage));
        }
    }

    void apply_permuted(ComplexBuffer& a) const {
        const int n = static_cast<int>(a.im.size());
        double* re = a.re.data();
        double* im = a.im.data();
        if (initial_radix2) {
            for (int i = 0; i < n; i += 2) {
                double rr = re[i], ri = im[i];
                re[i] = rr + re[i + 1];
                im[i] = ri + im[i + 1];
                re[i + 1] = rr - re[i + 1];
                im[i + 1] = ri - im[i + 1];
            }
        }
        for (const auto& stage : stages) {
            const int len = stage.len;
            const int quarter = len / 4;
#if defined(__AVX2__) && defined(__FMA__)
            if (quarter == 1 && n >= 16) {
                for (int i = 0; i < n; i += 16) {
                    auto ar = _mm256_loadu_pd(re + i);
                    auto br = _mm256_loadu_pd(re + i + 4);
                    auto cr = _mm256_loadu_pd(re + i + 8);
                    auto dr = _mm256_loadu_pd(re + i + 12);
                    auto ai = _mm256_loadu_pd(im + i);
                    auto bi = _mm256_loadu_pd(im + i + 4);
                    auto ci = _mm256_loadu_pd(im + i + 8);
                    auto di = _mm256_loadu_pd(im + i + 12);
                    transpose4(ar, br, cr, dr);
                    transpose4(ai, bi, ci, di);
                    auto sum01r = _mm256_add_pd(ar, br);
                    auto sum01i = _mm256_add_pd(ai, bi);
                    auto dif01r = _mm256_sub_pd(ar, br);
                    auto dif01i = _mm256_sub_pd(ai, bi);
                    auto sum23r = _mm256_add_pd(cr, dr);
                    auto sum23i = _mm256_add_pd(ci, di);
                    auto dif23r = _mm256_sub_pd(cr, dr);
                    auto dif23i = _mm256_sub_pd(ci, di);
                    ar = _mm256_add_pd(sum01r, sum23r);
                    ai = _mm256_add_pd(sum01i, sum23i);
                    br = _mm256_add_pd(dif01r, dif23i);
                    bi = _mm256_sub_pd(dif01i, dif23r);
                    cr = _mm256_sub_pd(sum01r, sum23r);
                    ci = _mm256_sub_pd(sum01i, sum23i);
                    dr = _mm256_sub_pd(dif01r, dif23i);
                    di = _mm256_add_pd(dif01i, dif23r);
                    transpose4(ar, br, cr, dr);
                    transpose4(ai, bi, ci, di);
                    _mm256_storeu_pd(re + i, ar);
                    _mm256_storeu_pd(re + i + 4, br);
                    _mm256_storeu_pd(re + i + 8, cr);
                    _mm256_storeu_pd(re + i + 12, dr);
                    _mm256_storeu_pd(im + i, ai);
                    _mm256_storeu_pd(im + i + 4, bi);
                    _mm256_storeu_pd(im + i + 8, ci);
                    _mm256_storeu_pd(im + i + 12, di);
                }
                continue;
            }
            if (quarter == 2 && n >= 32) {
                for (int i = 0; i < n; i += 32) {
                    alignas(32) double out_re[2][4][4], out_im[2][4][4];
                    for (int j = 0; j < 2; ++j) {
                        auto ar = _mm256_setr_pd(re[i + j], re[i + 8 + j],
                                                 re[i + 16 + j], re[i + 24 + j]);
                        auto ai = _mm256_setr_pd(im[i + j], im[i + 8 + j],
                                                 im[i + 16 + j], im[i + 24 + j]);
                        auto br = _mm256_setr_pd(re[i + 2 + j], re[i + 10 + j],
                                                 re[i + 18 + j], re[i + 26 + j]);
                        auto bi = _mm256_setr_pd(im[i + 2 + j], im[i + 10 + j],
                                                 im[i + 18 + j], im[i + 26 + j]);
                        auto cr = _mm256_setr_pd(re[i + 4 + j], re[i + 12 + j],
                                                 re[i + 20 + j], re[i + 28 + j]);
                        auto ci = _mm256_setr_pd(im[i + 4 + j], im[i + 12 + j],
                                                 im[i + 20 + j], im[i + 28 + j]);
                        auto dr = _mm256_setr_pd(re[i + 6 + j], re[i + 14 + j],
                                                 re[i + 22 + j], re[i + 30 + j]);
                        auto di = _mm256_setr_pd(im[i + 6 + j], im[i + 14 + j],
                                                 im[i + 22 + j], im[i + 30 + j]);
                        auto w1r = _mm256_set1_pd(stage.w1re[j]);
                        auto w1i = _mm256_set1_pd(stage.w1im[j]);
                        auto w2r = _mm256_set1_pd(stage.w2re[j]);
                        auto w2i = _mm256_set1_pd(stage.w2im[j]);
                        auto w3r = _mm256_set1_pd(stage.w3re[j]);
                        auto w3i = _mm256_set1_pd(stage.w3im[j]);
                        auto br2 = _mm256_fmsub_pd(br, w2r, _mm256_mul_pd(bi, w2i));
                        auto bi2 = _mm256_fmadd_pd(br, w2i, _mm256_mul_pd(bi, w2r));
                        auto cr1 = _mm256_fmsub_pd(cr, w1r, _mm256_mul_pd(ci, w1i));
                        auto ci1 = _mm256_fmadd_pd(cr, w1i, _mm256_mul_pd(ci, w1r));
                        auto dr3 = _mm256_fmsub_pd(dr, w3r, _mm256_mul_pd(di, w3i));
                        auto di3 = _mm256_fmadd_pd(dr, w3i, _mm256_mul_pd(di, w3r));
                        auto sum01r = _mm256_add_pd(ar, br2);
                        auto sum01i = _mm256_add_pd(ai, bi2);
                        auto dif01r = _mm256_sub_pd(ar, br2);
                        auto dif01i = _mm256_sub_pd(ai, bi2);
                        auto sum23r = _mm256_add_pd(cr1, dr3);
                        auto sum23i = _mm256_add_pd(ci1, di3);
                        auto dif23r = _mm256_sub_pd(cr1, dr3);
                        auto dif23i = _mm256_sub_pd(ci1, di3);
                        _mm256_store_pd(out_re[j][0], _mm256_add_pd(sum01r, sum23r));
                        _mm256_store_pd(out_im[j][0], _mm256_add_pd(sum01i, sum23i));
                        _mm256_store_pd(out_re[j][1], _mm256_add_pd(dif01r, dif23i));
                        _mm256_store_pd(out_im[j][1], _mm256_sub_pd(dif01i, dif23r));
                        _mm256_store_pd(out_re[j][2], _mm256_sub_pd(sum01r, sum23r));
                        _mm256_store_pd(out_im[j][2], _mm256_sub_pd(sum01i, sum23i));
                        _mm256_store_pd(out_re[j][3], _mm256_sub_pd(dif01r, dif23i));
                        _mm256_store_pd(out_im[j][3], _mm256_add_pd(dif01i, dif23r));
                    }
                    for (int block = 0; block < 4; ++block) {
                        for (int j = 0; j < 2; ++j) {
                            for (int part = 0; part < 4; ++part) {
                                re[i + 8 * block + j + 2 * part] = out_re[j][part][block];
                                im[i + 8 * block + j + 2 * part] = out_im[j][part][block];
                            }
                        }
                    }
                }
                continue;
            }
#endif
            for (int i = 0; i < n; i += len) {
                int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
                for (; j + 3 < quarter; j += 4) {
                    const int p = i + j;
                    auto ar = _mm256_loadu_pd(re + p);
                    auto ai = _mm256_loadu_pd(im + p);
                    auto br = _mm256_loadu_pd(re + p + quarter);
                    auto bi = _mm256_loadu_pd(im + p + quarter);
                    auto cr = _mm256_loadu_pd(re + p + 2 * quarter);
                    auto ci = _mm256_loadu_pd(im + p + 2 * quarter);
                    auto dr = _mm256_loadu_pd(re + p + 3 * quarter);
                    auto di = _mm256_loadu_pd(im + p + 3 * quarter);
                    auto w1r = _mm256_loadu_pd(stage.w1re.data() + j);
                    auto w1i = _mm256_loadu_pd(stage.w1im.data() + j);
                    auto w2r = _mm256_loadu_pd(stage.w2re.data() + j);
                    auto w2i = _mm256_loadu_pd(stage.w2im.data() + j);
                    auto w3r = _mm256_loadu_pd(stage.w3re.data() + j);
                    auto w3i = _mm256_loadu_pd(stage.w3im.data() + j);
                    auto br2 = _mm256_fmsub_pd(br, w2r, _mm256_mul_pd(bi, w2i));
                    auto bi2 = _mm256_fmadd_pd(br, w2i, _mm256_mul_pd(bi, w2r));
                    auto cr1 = _mm256_fmsub_pd(cr, w1r, _mm256_mul_pd(ci, w1i));
                    auto ci1 = _mm256_fmadd_pd(cr, w1i, _mm256_mul_pd(ci, w1r));
                    auto dr3 = _mm256_fmsub_pd(dr, w3r, _mm256_mul_pd(di, w3i));
                    auto di3 = _mm256_fmadd_pd(dr, w3i, _mm256_mul_pd(di, w3r));
                    auto sum01r = _mm256_add_pd(ar, br2);
                    auto sum01i = _mm256_add_pd(ai, bi2);
                    auto dif01r = _mm256_sub_pd(ar, br2);
                    auto dif01i = _mm256_sub_pd(ai, bi2);
                    auto sum23r = _mm256_add_pd(cr1, dr3);
                    auto sum23i = _mm256_add_pd(ci1, di3);
                    auto dif23r = _mm256_sub_pd(cr1, dr3);
                    auto dif23i = _mm256_sub_pd(ci1, di3);
                    _mm256_storeu_pd(re + p, _mm256_add_pd(sum01r, sum23r));
                    _mm256_storeu_pd(im + p, _mm256_add_pd(sum01i, sum23i));
                    _mm256_storeu_pd(re + p + quarter, _mm256_add_pd(dif01r, dif23i));
                    _mm256_storeu_pd(im + p + quarter, _mm256_sub_pd(dif01i, dif23r));
                    _mm256_storeu_pd(re + p + 2 * quarter, _mm256_sub_pd(sum01r, sum23r));
                    _mm256_storeu_pd(im + p + 2 * quarter, _mm256_sub_pd(sum01i, sum23i));
                    _mm256_storeu_pd(re + p + 3 * quarter, _mm256_sub_pd(dif01r, dif23i));
                    _mm256_storeu_pd(im + p + 3 * quarter, _mm256_add_pd(dif01i, dif23r));
                }
#endif
                for (; j < quarter; ++j) {
                    const int p = i + j;
                    double ar = re[p], ai = im[p];
                    double br = re[p + quarter], bi = im[p + quarter];
                    double cr = re[p + 2 * quarter], ci = im[p + 2 * quarter];
                    double dr = re[p + 3 * quarter], di = im[p + 3 * quarter];
                    double br2 = br * stage.w2re[j] - bi * stage.w2im[j];
                    double bi2 = br * stage.w2im[j] + bi * stage.w2re[j];
                    double cr1 = cr * stage.w1re[j] - ci * stage.w1im[j];
                    double ci1 = cr * stage.w1im[j] + ci * stage.w1re[j];
                    double dr3 = dr * stage.w3re[j] - di * stage.w3im[j];
                    double di3 = dr * stage.w3im[j] + di * stage.w3re[j];
                    double sum01r = ar + br2, sum01i = ai + bi2;
                    double dif01r = ar - br2, dif01i = ai - bi2;
                    double sum23r = cr1 + dr3, sum23i = ci1 + di3;
                    double dif23r = cr1 - dr3, dif23i = ci1 - di3;
                    re[p] = sum01r + sum23r;
                    im[p] = sum01i + sum23i;
                    re[p + quarter] = dif01r + dif23i;
                    im[p + quarter] = dif01i - dif23r;
                    re[p + 2 * quarter] = sum01r - sum23r;
                    im[p + 2 * quarter] = sum01i - sum23i;
                    re[p + 3 * quarter] = dif01r - dif23i;
                    im[p + 3 * quarter] = dif01i + dif23r;
                }
            }
        }
    }
};

struct Radix4Stage32 {
    int len;
    std::vector<float> w1re, w1im, w2re, w2im, w3re, w3im;
    explicit Radix4Stage32(const Radix4Stage& source)
        : len(source.len), w1re(source.w1re.begin(), source.w1re.end()),
          w1im(source.w1im.begin(), source.w1im.end()),
          w2re(source.w2re.begin(), source.w2re.end()),
          w2im(source.w2im.begin(), source.w2im.end()),
          w3re(source.w3re.begin(), source.w3re.end()),
          w3im(source.w3im.begin(), source.w3im.end()) {}
};

struct FFTPlan32 {
    std::vector<int> reversed;
    std::vector<Radix4Stage32> stages;
    bool initial_radix2;
    explicit FFTPlan32(const FFTPlan& source)
        : reversed(source.reversed), initial_radix2(source.initial_radix2) {
        stages.reserve(source.stages.size());
        for (const auto& stage : source.stages) stages.emplace_back(stage);
    }
};

#if defined(__AVX2__) && defined(__FMA__)
struct alignas(32) Lane4 {
    std::array<double, 4> values;
};

static inline __m256d load4(const Lane4& value) {
    return _mm256_load_pd(value.values.data());
}

static inline void store4(Lane4& destination, __m256d value) {
    _mm256_store_pd(destination.values.data(), value);
}

struct Batch4Buffer {
    std::vector<Lane4> re, im;
    explicit Batch4Buffer(int n) : re(n + 1), im(n) {}
};

void apply_batch4(const FFTPlan& plan, Batch4Buffer& a) {
    const int n = static_cast<int>(a.im.size());
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            auto rr = load4(a.re[i]), ri = load4(a.im[i]);
            store4(a.re[i], _mm256_add_pd(rr, load4(a.re[i + 1])));
            store4(a.im[i], _mm256_add_pd(ri, load4(a.im[i + 1])));
            store4(a.re[i + 1], _mm256_sub_pd(rr, load4(a.re[i + 1])));
            store4(a.im[i + 1], _mm256_sub_pd(ri, load4(a.im[i + 1])));
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            if (quarter == 1) {
                auto ar = load4(a.re[base]), ai = load4(a.im[base]);
                auto br = load4(a.re[base + 1]), bi = load4(a.im[base + 1]);
                auto cr = load4(a.re[base + 2]), ci = load4(a.im[base + 2]);
                auto dr = load4(a.re[base + 3]), di = load4(a.im[base + 3]);
                auto sum01r = _mm256_add_pd(ar, br);
                auto sum01i = _mm256_add_pd(ai, bi);
                auto dif01r = _mm256_sub_pd(ar, br);
                auto dif01i = _mm256_sub_pd(ai, bi);
                auto sum23r = _mm256_add_pd(cr, dr);
                auto sum23i = _mm256_add_pd(ci, di);
                auto dif23r = _mm256_sub_pd(cr, dr);
                auto dif23i = _mm256_sub_pd(ci, di);
                store4(a.re[base], _mm256_add_pd(sum01r, sum23r));
                store4(a.im[base], _mm256_add_pd(sum01i, sum23i));
                store4(a.re[base + 1], _mm256_add_pd(dif01r, dif23i));
                store4(a.im[base + 1], _mm256_sub_pd(dif01i, dif23r));
                store4(a.re[base + 2], _mm256_sub_pd(sum01r, sum23r));
                store4(a.im[base + 2], _mm256_sub_pd(sum01i, sum23i));
                store4(a.re[base + 3], _mm256_sub_pd(dif01r, dif23i));
                store4(a.im[base + 3], _mm256_add_pd(dif01i, dif23r));
                continue;
            }
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                auto ar = load4(a.re[p]), ai = load4(a.im[p]);
                auto br = load4(a.re[p + quarter]), bi = load4(a.im[p + quarter]);
                auto cr = load4(a.re[p + 2 * quarter]), ci = load4(a.im[p + 2 * quarter]);
                auto dr = load4(a.re[p + 3 * quarter]), di = load4(a.im[p + 3 * quarter]);
                auto w1r = _mm256_set1_pd(stage.w1re[j]);
                auto w1i = _mm256_set1_pd(stage.w1im[j]);
                auto w2r = _mm256_set1_pd(stage.w2re[j]);
                auto w2i = _mm256_set1_pd(stage.w2im[j]);
                auto w3r = _mm256_set1_pd(stage.w3re[j]);
                auto w3i = _mm256_set1_pd(stage.w3im[j]);
                auto br2 = _mm256_fmsub_pd(br, w2r, _mm256_mul_pd(bi, w2i));
                auto bi2 = _mm256_fmadd_pd(br, w2i, _mm256_mul_pd(bi, w2r));
                auto cr1 = _mm256_fmsub_pd(cr, w1r, _mm256_mul_pd(ci, w1i));
                auto ci1 = _mm256_fmadd_pd(cr, w1i, _mm256_mul_pd(ci, w1r));
                auto dr3 = _mm256_fmsub_pd(dr, w3r, _mm256_mul_pd(di, w3i));
                auto di3 = _mm256_fmadd_pd(dr, w3i, _mm256_mul_pd(di, w3r));
                auto sum01r = _mm256_add_pd(ar, br2);
                auto sum01i = _mm256_add_pd(ai, bi2);
                auto dif01r = _mm256_sub_pd(ar, br2);
                auto dif01i = _mm256_sub_pd(ai, bi2);
                auto sum23r = _mm256_add_pd(cr1, dr3);
                auto sum23i = _mm256_add_pd(ci1, di3);
                auto dif23r = _mm256_sub_pd(cr1, dr3);
                auto dif23i = _mm256_sub_pd(ci1, di3);
                store4(a.re[p], _mm256_add_pd(sum01r, sum23r));
                store4(a.im[p], _mm256_add_pd(sum01i, sum23i));
                store4(a.re[p + quarter], _mm256_add_pd(dif01r, dif23i));
                store4(a.im[p + quarter], _mm256_sub_pd(dif01i, dif23r));
                store4(a.re[p + 2 * quarter], _mm256_sub_pd(sum01r, sum23r));
                store4(a.im[p + 2 * quarter], _mm256_sub_pd(sum01i, sum23i));
                store4(a.re[p + 3 * quarter], _mm256_sub_pd(dif01r, dif23i));
                store4(a.im[p + 3 * quarter], _mm256_add_pd(dif01i, dif23r));
            }
        }
    }
}

void recover_power_batch4(Batch4Buffer& a, const std::vector<double>& recovery_re,
                          const std::vector<double>& recovery_im,
                          int support_first, int support_last) {
    const int half = static_cast<int>(a.im.size());
    auto dc = _mm256_add_pd(load4(a.re[0]), load4(a.im[0]));
    auto nyquist = _mm256_sub_pd(load4(a.re[0]), load4(a.im[0]));
    store4(a.re[0], _mm256_mul_pd(dc, dc));
    store4(a.re[half], _mm256_mul_pd(nyquist, nyquist));
    const auto scale = _mm256_set1_pd(0.5);
    if (2 * (support_last - 1) >= half) {
        for (int k = 1; k < half - k; ++k) {
            auto ar = load4(a.re[k]), ai = load4(a.im[k]);
            auto br = load4(a.re[half - k]);
            auto bi = _mm256_sub_pd(_mm256_setzero_pd(), load4(a.im[half - k]));
            auto rr = _mm256_set1_pd(recovery_re[k]);
            auto ri = _mm256_set1_pd(recovery_im[k]);
            auto dr = _mm256_sub_pd(ar, br);
            auto di = _mm256_sub_pd(ai, bi);
            auto sumr = _mm256_mul_pd(scale, _mm256_add_pd(ar, br));
            auto xr = _mm256_fnmadd_pd(ri, di, _mm256_fmadd_pd(rr, dr, sumr));
            auto xi = _mm256_fmadd_pd(ri, dr, _mm256_mul_pd(scale, _mm256_add_pd(ai, bi)));
            xi = _mm256_fmadd_pd(rr, di, xi);
            int j = half - k;
            auto rr2 = _mm256_set1_pd(recovery_re[j]);
            auto ri2 = _mm256_set1_pd(recovery_im[j]);
            auto dr2 = _mm256_sub_pd(_mm256_setzero_pd(), dr);
            auto di2 = di;
            auto zr = _mm256_fnmadd_pd(ri2, di2, _mm256_fmadd_pd(rr2, dr2, sumr));
            auto zi = _mm256_fmadd_pd(ri2, dr2,
                _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), scale),
                              _mm256_add_pd(ai, bi)));
            zi = _mm256_fmadd_pd(rr2, di2, zi);
            store4(a.re[k], _mm256_fmadd_pd(xr, xr, _mm256_mul_pd(xi, xi)));
            store4(a.re[j], _mm256_fmadd_pd(zr, zr, _mm256_mul_pd(zi, zi)));
        }
        if (half % 2 == 0) {
            const int k = half / 2;
            auto ar = load4(a.re[k]), ai = load4(a.im[k]);
            auto bi = _mm256_sub_pd(_mm256_setzero_pd(), ai);
            auto rr = _mm256_set1_pd(recovery_re[k]);
            auto ri = _mm256_set1_pd(recovery_im[k]);
            auto di = _mm256_sub_pd(ai, bi);
            auto xr = _mm256_fnmadd_pd(ri, di, ar);
            auto xi = _mm256_fmadd_pd(rr, di, _mm256_mul_pd(scale, _mm256_add_pd(ai, bi)));
            store4(a.re[k], _mm256_fmadd_pd(xr, xr, _mm256_mul_pd(xi, xi)));
        }
        return;
    }
    const int begin = std::max(1, support_first);
    const int end = std::min(half - 1, support_last - 1);
    for (int k = begin; k <= end; ++k) {
        auto ar = load4(a.re[k]), ai = load4(a.im[k]);
        auto br = load4(a.re[half - k]);
        auto bi = _mm256_sub_pd(_mm256_setzero_pd(), load4(a.im[half - k]));
        auto rr = _mm256_set1_pd(recovery_re[k]);
        auto ri = _mm256_set1_pd(recovery_im[k]);
        auto dr = _mm256_sub_pd(ar, br);
        auto di = _mm256_sub_pd(ai, bi);
        auto xr = _mm256_fmadd_pd(rr, dr, _mm256_mul_pd(scale, _mm256_add_pd(ar, br)));
        xr = _mm256_fnmadd_pd(ri, di, xr);
        auto xi = _mm256_fmadd_pd(ri, dr, _mm256_mul_pd(scale, _mm256_add_pd(ai, bi)));
        xi = _mm256_fmadd_pd(rr, di, xi);
        store4(a.re[k], _mm256_fmadd_pd(xr, xr, _mm256_mul_pd(xi, xi)));
    }
}
#endif

#if MELSPEC_FP32_BATCH == 8 && defined(__AVX2__) && defined(__FMA__)
struct alignas(32) Lane8f {
    std::array<float, 8> values;
};

struct Batch8fBuffer {
    std::vector<Lane8f> re, im, power;
    explicit Batch8fBuffer(int n) : re(n), im(n), power(n + 1) {}
};

static inline __m256 load8f(const Lane8f& value) {
    return _mm256_load_ps(value.values.data());
}

static inline void store8f(Lane8f& destination, __m256 value) {
    _mm256_store_ps(destination.values.data(), value);
}

void apply_batch8f(const FFTPlan32& plan, Batch8fBuffer& a) {
    const int n = static_cast<int>(a.im.size());
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            auto rr = load8f(a.re[i]), ri = load8f(a.im[i]);
            store8f(a.re[i], _mm256_add_ps(rr, load8f(a.re[i + 1])));
            store8f(a.im[i], _mm256_add_ps(ri, load8f(a.im[i + 1])));
            store8f(a.re[i + 1], _mm256_sub_ps(rr, load8f(a.re[i + 1])));
            store8f(a.im[i + 1], _mm256_sub_ps(ri, load8f(a.im[i + 1])));
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                auto ar = load8f(a.re[p]), ai = load8f(a.im[p]);
                auto br = load8f(a.re[p + quarter]), bi = load8f(a.im[p + quarter]);
                auto cr = load8f(a.re[p + 2 * quarter]), ci = load8f(a.im[p + 2 * quarter]);
                auto dr = load8f(a.re[p + 3 * quarter]), di = load8f(a.im[p + 3 * quarter]);
                auto w1r = _mm256_set1_ps(stage.w1re[j]);
                auto w1i = _mm256_set1_ps(stage.w1im[j]);
                auto w2r = _mm256_set1_ps(stage.w2re[j]);
                auto w2i = _mm256_set1_ps(stage.w2im[j]);
                auto w3r = _mm256_set1_ps(stage.w3re[j]);
                auto w3i = _mm256_set1_ps(stage.w3im[j]);
                auto br2 = _mm256_fmsub_ps(br, w2r, _mm256_mul_ps(bi, w2i));
                auto bi2 = _mm256_fmadd_ps(br, w2i, _mm256_mul_ps(bi, w2r));
                auto cr1 = _mm256_fmsub_ps(cr, w1r, _mm256_mul_ps(ci, w1i));
                auto ci1 = _mm256_fmadd_ps(cr, w1i, _mm256_mul_ps(ci, w1r));
                auto dr3 = _mm256_fmsub_ps(dr, w3r, _mm256_mul_ps(di, w3i));
                auto di3 = _mm256_fmadd_ps(dr, w3i, _mm256_mul_ps(di, w3r));
                auto sum01r = _mm256_add_ps(ar, br2);
                auto sum01i = _mm256_add_ps(ai, bi2);
                auto dif01r = _mm256_sub_ps(ar, br2);
                auto dif01i = _mm256_sub_ps(ai, bi2);
                auto sum23r = _mm256_add_ps(cr1, dr3);
                auto sum23i = _mm256_add_ps(ci1, di3);
                auto dif23r = _mm256_sub_ps(cr1, dr3);
                auto dif23i = _mm256_sub_ps(ci1, di3);
                store8f(a.re[p], _mm256_add_ps(sum01r, sum23r));
                store8f(a.im[p], _mm256_add_ps(sum01i, sum23i));
                store8f(a.re[p + quarter], _mm256_add_ps(dif01r, dif23i));
                store8f(a.im[p + quarter], _mm256_sub_ps(dif01i, dif23r));
                store8f(a.re[p + 2 * quarter], _mm256_sub_ps(sum01r, sum23r));
                store8f(a.im[p + 2 * quarter], _mm256_sub_ps(sum01i, sum23i));
                store8f(a.re[p + 3 * quarter], _mm256_sub_ps(dif01r, dif23i));
                store8f(a.im[p + 3 * quarter], _mm256_add_ps(dif01i, dif23r));
            }
        }
    }
}

void recover_power_batch8f(Batch8fBuffer& a, const std::vector<double>& recovery_re,
                           const std::vector<double>& recovery_im,
                           int support_first, int support_last) {
    const int half = static_cast<int>(a.im.size());
    const auto scale = _mm256_set1_pd(0.5);
    auto convert_power = [](__m256d lo, __m256d hi) {
        auto low = _mm256_cvtpd_ps(lo);
        auto high = _mm256_cvtpd_ps(hi);
        return _mm256_set_m128(high, low);
    };
    auto ar0 = load8f(a.re[0]), ai0 = load8f(a.im[0]);
    auto arlo0 = _mm256_cvtps_pd(_mm256_castps256_ps128(ar0));
    auto arhi0 = _mm256_cvtps_pd(_mm256_extractf128_ps(ar0, 1));
    auto ailo0 = _mm256_cvtps_pd(_mm256_castps256_ps128(ai0));
    auto aihi0 = _mm256_cvtps_pd(_mm256_extractf128_ps(ai0, 1));
    auto dclo = _mm256_add_pd(arlo0, ailo0);
    auto dchi = _mm256_add_pd(arhi0, aihi0);
    auto nyqlo = _mm256_sub_pd(arlo0, ailo0);
    auto nyqhi = _mm256_sub_pd(arhi0, aihi0);
    store8f(a.power[0], convert_power(_mm256_mul_pd(dclo, dclo),
                                      _mm256_mul_pd(dchi, dchi)));
    store8f(a.power[half], convert_power(_mm256_mul_pd(nyqlo, nyqlo),
                                         _mm256_mul_pd(nyqhi, nyqhi)));
    const int begin = std::max(1, support_first);
    const int end = std::min(half - 1, support_last - 1);
    for (int k = begin; k <= end; ++k) {
        auto arf = load8f(a.re[k]), aif = load8f(a.im[k]);
        auto brf = load8f(a.re[half - k]), bif = load8f(a.im[half - k]);
        auto arlo = _mm256_cvtps_pd(_mm256_castps256_ps128(arf));
        auto arhi = _mm256_cvtps_pd(_mm256_extractf128_ps(arf, 1));
        auto ailo = _mm256_cvtps_pd(_mm256_castps256_ps128(aif));
        auto aihi = _mm256_cvtps_pd(_mm256_extractf128_ps(aif, 1));
        auto brlo = _mm256_cvtps_pd(_mm256_castps256_ps128(brf));
        auto brhi = _mm256_cvtps_pd(_mm256_extractf128_ps(brf, 1));
        auto bilo = _mm256_cvtps_pd(_mm256_castps256_ps128(bif));
        auto bihi = _mm256_cvtps_pd(_mm256_extractf128_ps(bif, 1));
        bilo = _mm256_sub_pd(_mm256_setzero_pd(), bilo);
        bihi = _mm256_sub_pd(_mm256_setzero_pd(), bihi);
        auto rr = _mm256_set1_pd(recovery_re[k]);
        auto ri = _mm256_set1_pd(recovery_im[k]);
        auto calc = [&](__m256d ar, __m256d ai, __m256d br, __m256d bi) {
            auto dr = _mm256_sub_pd(ar, br);
            auto di = _mm256_sub_pd(ai, bi);
            auto xr = _mm256_fnmadd_pd(ri, di,
                _mm256_fmadd_pd(rr, dr, _mm256_mul_pd(scale, _mm256_add_pd(ar, br))));
            auto xi = _mm256_fmadd_pd(ri, dr,
                _mm256_mul_pd(scale, _mm256_add_pd(ai, bi)));
            xi = _mm256_fmadd_pd(rr, di, xi);
            return _mm256_fmadd_pd(xr, xr, _mm256_mul_pd(xi, xi));
        };
        store8f(a.power[k], convert_power(calc(arlo, ailo, brlo, bilo),
                                          calc(arhi, aihi, brhi, bihi)));
    }
}
#endif

#if MELSPEC_FP32_BATCH == 16 && defined(__AVX512F__)
struct alignas(64) Lane16f {
    std::array<float, 16> values;
};

struct Batch16fBuffer {
    std::vector<Lane16f> re, im, power;
    explicit Batch16fBuffer(int n) : re(n), im(n), power(n + 1) {}
};

static inline __m512 load16f(const Lane16f& value) {
    return _mm512_load_ps(value.values.data());
}

static inline void store16f(Lane16f& destination, __m512 value) {
    _mm512_store_ps(destination.values.data(), value);
}

void apply_batch16f(const FFTPlan32& plan, Batch16fBuffer& a) {
    const int n = static_cast<int>(a.im.size());
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            auto rr = load16f(a.re[i]), ri = load16f(a.im[i]);
            store16f(a.re[i], _mm512_add_ps(rr, load16f(a.re[i + 1])));
            store16f(a.im[i], _mm512_add_ps(ri, load16f(a.im[i + 1])));
            store16f(a.re[i + 1], _mm512_sub_ps(rr, load16f(a.re[i + 1])));
            store16f(a.im[i + 1], _mm512_sub_ps(ri, load16f(a.im[i + 1])));
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                auto ar = load16f(a.re[p]), ai = load16f(a.im[p]);
                auto br = load16f(a.re[p + quarter]), bi = load16f(a.im[p + quarter]);
                auto cr = load16f(a.re[p + 2 * quarter]), ci = load16f(a.im[p + 2 * quarter]);
                auto dr = load16f(a.re[p + 3 * quarter]), di = load16f(a.im[p + 3 * quarter]);
                auto w1r = _mm512_set1_ps(stage.w1re[j]);
                auto w1i = _mm512_set1_ps(stage.w1im[j]);
                auto w2r = _mm512_set1_ps(stage.w2re[j]);
                auto w2i = _mm512_set1_ps(stage.w2im[j]);
                auto w3r = _mm512_set1_ps(stage.w3re[j]);
                auto w3i = _mm512_set1_ps(stage.w3im[j]);
                auto br2 = _mm512_fmsub_ps(br, w2r, _mm512_mul_ps(bi, w2i));
                auto bi2 = _mm512_fmadd_ps(br, w2i, _mm512_mul_ps(bi, w2r));
                auto cr1 = _mm512_fmsub_ps(cr, w1r, _mm512_mul_ps(ci, w1i));
                auto ci1 = _mm512_fmadd_ps(cr, w1i, _mm512_mul_ps(ci, w1r));
                auto dr3 = _mm512_fmsub_ps(dr, w3r, _mm512_mul_ps(di, w3i));
                auto di3 = _mm512_fmadd_ps(dr, w3i, _mm512_mul_ps(di, w3r));
                auto sum01r = _mm512_add_ps(ar, br2);
                auto sum01i = _mm512_add_ps(ai, bi2);
                auto dif01r = _mm512_sub_ps(ar, br2);
                auto dif01i = _mm512_sub_ps(ai, bi2);
                auto sum23r = _mm512_add_ps(cr1, dr3);
                auto sum23i = _mm512_add_ps(ci1, di3);
                auto dif23r = _mm512_sub_ps(cr1, dr3);
                auto dif23i = _mm512_sub_ps(ci1, di3);
                store16f(a.re[p], _mm512_add_ps(sum01r, sum23r));
                store16f(a.im[p], _mm512_add_ps(sum01i, sum23i));
                store16f(a.re[p + quarter], _mm512_add_ps(dif01r, dif23i));
                store16f(a.im[p + quarter], _mm512_sub_ps(dif01i, dif23r));
                store16f(a.re[p + 2 * quarter], _mm512_sub_ps(sum01r, sum23r));
                store16f(a.im[p + 2 * quarter], _mm512_sub_ps(sum01i, sum23i));
                store16f(a.re[p + 3 * quarter], _mm512_sub_ps(dif01r, dif23i));
                store16f(a.im[p + 3 * quarter], _mm512_add_ps(dif01i, dif23r));
            }
        }
    }
}

// Widen 16 packed floats to two __m512d halves, and narrow back.
static inline __m512d expand16_lo(__m512 value) {
    return _mm512_cvtps_pd(_mm512_castps512_ps256(value));
}

static inline __m512d expand16_hi(__m512 value) {
    const __m512 upper = _mm512_shuffle_f32x4(value, value, 0xEE);
    return _mm512_cvtps_pd(_mm512_castps512_ps256(upper));
}

static inline __m512 narrow16(__m512d lo, __m512d hi) {
    const __m256 lo_f = _mm512_cvtpd_ps(lo);
    const __m256 hi_f = _mm512_cvtpd_ps(hi);
    const __m512d merged = _mm512_insertf64x4(
        _mm512_castps_pd(_mm512_castps256_ps512(lo_f)),
        _mm256_castps_pd(hi_f), 1);
    return _mm512_castpd_ps(merged);
}

void recover_power_batch16f(Batch16fBuffer& a, const std::vector<double>& recovery_re,
                            const std::vector<double>& recovery_im,
                            int support_first, int support_last) {
    const int half = static_cast<int>(a.im.size());
    const auto scale = _mm512_set1_pd(0.5);
    auto power = [](__m512d xr, __m512d xi) {
        return _mm512_fmadd_pd(xr, xr, _mm512_mul_pd(xi, xi));
    };
    const auto ar0 = load16f(a.re[0]), ai0 = load16f(a.im[0]);
    const auto arlo0 = expand16_lo(ar0), arhi0 = expand16_hi(ar0);
    const auto ailo0 = expand16_lo(ai0), aihi0 = expand16_hi(ai0);
    const auto dclo = _mm512_add_pd(arlo0, ailo0);
    const auto dchi = _mm512_add_pd(arhi0, aihi0);
    const auto nyqlo = _mm512_sub_pd(arlo0, ailo0);
    const auto nyqhi = _mm512_sub_pd(arhi0, aihi0);
    store16f(a.power[0], narrow16(_mm512_mul_pd(dclo, dclo),
                                  _mm512_mul_pd(dchi, dchi)));
    store16f(a.power[half], narrow16(_mm512_mul_pd(nyqlo, nyqlo),
                                     _mm512_mul_pd(nyqhi, nyqhi)));
    const int begin = std::max(1, support_first);
    const int end = std::min(half - 1, support_last - 1);
    for (int k = begin; k <= end; ++k) {
        const auto arf = load16f(a.re[k]), aif = load16f(a.im[k]);
        const auto brf = load16f(a.re[half - k]), bif = load16f(a.im[half - k]);
        const auto arlo = expand16_lo(arf), arhi = expand16_hi(arf);
        const auto ailo = expand16_lo(aif), aihi = expand16_hi(aif);
        const auto brlo = expand16_lo(brf), brhi = expand16_hi(brf);
        const auto bilo = _mm512_sub_pd(_mm512_setzero_pd(), expand16_lo(bif));
        const auto bihi = _mm512_sub_pd(_mm512_setzero_pd(), expand16_hi(bif));
        const auto rr = _mm512_set1_pd(recovery_re[k]);
        const auto ri = _mm512_set1_pd(recovery_im[k]);
        auto calc = [&](__m512d ar, __m512d ai, __m512d br, __m512d bi) {
            const auto dr = _mm512_sub_pd(ar, br);
            const auto di = _mm512_sub_pd(ai, bi);
            auto xr = _mm512_fnmadd_pd(ri, di,
                _mm512_fmadd_pd(rr, dr, _mm512_mul_pd(scale, _mm512_add_pd(ar, br))));
            auto xi = _mm512_fmadd_pd(ri, dr,
                _mm512_mul_pd(scale, _mm512_add_pd(ai, bi)));
            xi = _mm512_fmadd_pd(rr, di, xi);
            return power(xr, xi);
        };
        store16f(a.power[k], narrow16(calc(arlo, ailo, brlo, bilo),
                                      calc(arhi, aihi, brhi, bihi)));
    }
}
#endif

#if defined(__AVX512F__)
struct alignas(64) Lane8 {
    std::array<double, 8> values;
};

static inline __m512d load8(const Lane8& value) {
    return _mm512_load_pd(value.values.data());
}

static inline void store8(Lane8& destination, __m512d value) {
    _mm512_store_pd(destination.values.data(), value);
}

struct Batch8Buffer {
    std::vector<Lane8> re, im;
    explicit Batch8Buffer(int n) : re(n + 1), im(n) {}
};

void apply_batch8(const FFTPlan& plan, Batch8Buffer& a) {
    const int n = static_cast<int>(a.im.size());
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            auto rr = load8(a.re[i]), ri = load8(a.im[i]);
            store8(a.re[i], _mm512_add_pd(rr, load8(a.re[i + 1])));
            store8(a.im[i], _mm512_add_pd(ri, load8(a.im[i + 1])));
            store8(a.re[i + 1], _mm512_sub_pd(rr, load8(a.re[i + 1])));
            store8(a.im[i + 1], _mm512_sub_pd(ri, load8(a.im[i + 1])));
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            if (quarter == 1) {
                auto ar = load8(a.re[base]), ai = load8(a.im[base]);
                auto br = load8(a.re[base + 1]), bi = load8(a.im[base + 1]);
                auto cr = load8(a.re[base + 2]), ci = load8(a.im[base + 2]);
                auto dr = load8(a.re[base + 3]), di = load8(a.im[base + 3]);
                auto sum01r = _mm512_add_pd(ar, br);
                auto sum01i = _mm512_add_pd(ai, bi);
                auto dif01r = _mm512_sub_pd(ar, br);
                auto dif01i = _mm512_sub_pd(ai, bi);
                auto sum23r = _mm512_add_pd(cr, dr);
                auto sum23i = _mm512_add_pd(ci, di);
                auto dif23r = _mm512_sub_pd(cr, dr);
                auto dif23i = _mm512_sub_pd(ci, di);
                store8(a.re[base], _mm512_add_pd(sum01r, sum23r));
                store8(a.im[base], _mm512_add_pd(sum01i, sum23i));
                store8(a.re[base + 1], _mm512_add_pd(dif01r, dif23i));
                store8(a.im[base + 1], _mm512_sub_pd(dif01i, dif23r));
                store8(a.re[base + 2], _mm512_sub_pd(sum01r, sum23r));
                store8(a.im[base + 2], _mm512_sub_pd(sum01i, sum23i));
                store8(a.re[base + 3], _mm512_sub_pd(dif01r, dif23i));
                store8(a.im[base + 3], _mm512_add_pd(dif01i, dif23r));
                continue;
            }
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                auto ar = load8(a.re[p]), ai = load8(a.im[p]);
                auto br = load8(a.re[p + quarter]), bi = load8(a.im[p + quarter]);
                auto cr = load8(a.re[p + 2 * quarter]), ci = load8(a.im[p + 2 * quarter]);
                auto dr = load8(a.re[p + 3 * quarter]), di = load8(a.im[p + 3 * quarter]);
                auto w1r = _mm512_set1_pd(stage.w1re[j]);
                auto w1i = _mm512_set1_pd(stage.w1im[j]);
                auto w2r = _mm512_set1_pd(stage.w2re[j]);
                auto w2i = _mm512_set1_pd(stage.w2im[j]);
                auto w3r = _mm512_set1_pd(stage.w3re[j]);
                auto w3i = _mm512_set1_pd(stage.w3im[j]);
                auto br2 = _mm512_fmsub_pd(br, w2r, _mm512_mul_pd(bi, w2i));
                auto bi2 = _mm512_fmadd_pd(br, w2i, _mm512_mul_pd(bi, w2r));
                auto cr1 = _mm512_fmsub_pd(cr, w1r, _mm512_mul_pd(ci, w1i));
                auto ci1 = _mm512_fmadd_pd(cr, w1i, _mm512_mul_pd(ci, w1r));
                auto dr3 = _mm512_fmsub_pd(dr, w3r, _mm512_mul_pd(di, w3i));
                auto di3 = _mm512_fmadd_pd(dr, w3i, _mm512_mul_pd(di, w3r));
                auto sum01r = _mm512_add_pd(ar, br2);
                auto sum01i = _mm512_add_pd(ai, bi2);
                auto dif01r = _mm512_sub_pd(ar, br2);
                auto dif01i = _mm512_sub_pd(ai, bi2);
                auto sum23r = _mm512_add_pd(cr1, dr3);
                auto sum23i = _mm512_add_pd(ci1, di3);
                auto dif23r = _mm512_sub_pd(cr1, dr3);
                auto dif23i = _mm512_sub_pd(ci1, di3);
                store8(a.re[p], _mm512_add_pd(sum01r, sum23r));
                store8(a.im[p], _mm512_add_pd(sum01i, sum23i));
                store8(a.re[p + quarter], _mm512_add_pd(dif01r, dif23i));
                store8(a.im[p + quarter], _mm512_sub_pd(dif01i, dif23r));
                store8(a.re[p + 2 * quarter], _mm512_sub_pd(sum01r, sum23r));
                store8(a.im[p + 2 * quarter], _mm512_sub_pd(sum01i, sum23i));
                store8(a.re[p + 3 * quarter], _mm512_sub_pd(dif01r, dif23i));
                store8(a.im[p + 3 * quarter], _mm512_add_pd(dif01i, dif23r));
            }
        }
    }
}

void recover_power_batch8(Batch8Buffer& a, const std::vector<double>& recovery_re,
                          const std::vector<double>& recovery_im,
                          int support_first, int support_last) {
    const int half = static_cast<int>(a.im.size());
    auto dc = _mm512_add_pd(load8(a.re[0]), load8(a.im[0]));
    auto nyquist = _mm512_sub_pd(load8(a.re[0]), load8(a.im[0]));
    store8(a.re[0], _mm512_mul_pd(dc, dc));
    store8(a.re[half], _mm512_mul_pd(nyquist, nyquist));
    const auto scale = _mm512_set1_pd(0.5);
    if (2 * (support_last - 1) >= half) {
        for (int k = 1; k < half - k; ++k) {
            auto ar = load8(a.re[k]), ai = load8(a.im[k]);
            auto br = load8(a.re[half - k]);
            auto bi = _mm512_sub_pd(_mm512_setzero_pd(), load8(a.im[half - k]));
            auto rr = _mm512_set1_pd(recovery_re[k]);
            auto ri = _mm512_set1_pd(recovery_im[k]);
            auto dr = _mm512_sub_pd(ar, br);
            auto di = _mm512_sub_pd(ai, bi);
            auto sumr = _mm512_mul_pd(scale, _mm512_add_pd(ar, br));
            auto xr = _mm512_fnmadd_pd(ri, di, _mm512_fmadd_pd(rr, dr, sumr));
            auto xi = _mm512_fmadd_pd(ri, dr, _mm512_mul_pd(scale, _mm512_add_pd(ai, bi)));
            xi = _mm512_fmadd_pd(rr, di, xi);
            int j = half - k;
            auto rr2 = _mm512_set1_pd(recovery_re[j]);
            auto ri2 = _mm512_set1_pd(recovery_im[j]);
            auto dr2 = _mm512_sub_pd(_mm512_setzero_pd(), dr);
            auto di2 = di;
            auto zr = _mm512_fnmadd_pd(ri2, di2, _mm512_fmadd_pd(rr2, dr2, sumr));
            auto zi = _mm512_fmadd_pd(ri2, dr2,
                _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), scale),
                              _mm512_add_pd(ai, bi)));
            zi = _mm512_fmadd_pd(rr2, di2, zi);
            store8(a.re[k], _mm512_fmadd_pd(xr, xr, _mm512_mul_pd(xi, xi)));
            store8(a.re[j], _mm512_fmadd_pd(zr, zr, _mm512_mul_pd(zi, zi)));
        }
        if (half % 2 == 0) {
            const int k = half / 2;
            auto ar = load8(a.re[k]), ai = load8(a.im[k]);
            auto bi = _mm512_sub_pd(_mm512_setzero_pd(), ai);
            auto rr = _mm512_set1_pd(recovery_re[k]);
            auto ri = _mm512_set1_pd(recovery_im[k]);
            auto di = _mm512_sub_pd(ai, bi);
            auto xr = _mm512_fnmadd_pd(ri, di, ar);
            auto xi = _mm512_fmadd_pd(rr, di, _mm512_mul_pd(scale, _mm512_add_pd(ai, bi)));
            store8(a.re[k], _mm512_fmadd_pd(xr, xr, _mm512_mul_pd(xi, xi)));
        }
        return;
    }
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
        store8(a.re[k], _mm512_fmadd_pd(xr, xr, _mm512_mul_pd(xi, xi)));
    }
}
#endif

void recover_power(const ComplexBuffer& a, const std::vector<double>& recovery_re,
                   const std::vector<double>& recovery_im, std::vector<double>& power) {
    const int half = static_cast<int>(a.im.size());
    double dc = a.re[0] + a.im[0];
    double nyquist = a.re[0] - a.im[0];
    power[0] = dc * dc;
    power[half] = nyquist * nyquist;
    int k = 1;
#if defined(__AVX2__) && defined(__FMA__)
    const auto scale = _mm256_set1_pd(0.5);
    for (; k + 3 < half; k += 4) {
        auto ar = _mm256_loadu_pd(a.re.data() + k);
        auto ai = _mm256_loadu_pd(a.im.data() + k);
        auto br = _mm256_loadu_pd(a.re.data() + half - k - 3);
        auto bi = _mm256_loadu_pd(a.im.data() + half - k - 3);
        br = _mm256_permute4x64_pd(br, _MM_SHUFFLE(0, 1, 2, 3));
        bi = _mm256_permute4x64_pd(bi, _MM_SHUFFLE(0, 1, 2, 3));
        bi = _mm256_sub_pd(_mm256_setzero_pd(), bi);
        auto rr = _mm256_loadu_pd(recovery_re.data() + k);
        auto ri = _mm256_loadu_pd(recovery_im.data() + k);
        auto dr = _mm256_sub_pd(ar, br);
        auto di = _mm256_sub_pd(ai, bi);
        auto xr = _mm256_fmadd_pd(rr, dr, _mm256_mul_pd(scale, _mm256_add_pd(ar, br)));
        xr = _mm256_fnmadd_pd(ri, di, xr);
        auto xi = _mm256_fmadd_pd(ri, dr, _mm256_mul_pd(scale, _mm256_add_pd(ai, bi)));
        xi = _mm256_fmadd_pd(rr, di, xi);
        _mm256_storeu_pd(power.data() + k,
                         _mm256_fmadd_pd(xr, xr, _mm256_mul_pd(xi, xi)));
    }
#endif
    for (; k < half; ++k) {
        double ar = a.re[k], ai = a.im[k];
        double br = a.re[half - k], bi = -a.im[half - k];
        double dr = ar - br, di = ai - bi;
        double xr = 0.5 * (ar + br) + recovery_re[k] * dr - recovery_im[k] * di;
        double xi = 0.5 * (ai + bi) + recovery_im[k] * dr + recovery_re[k] * di;
        power[k] = xr * xr + xi * xi;
    }
}

// Real FFT: input n real values → output n/2+1 complex values.
void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    FFTPlan plan(n);
    ComplexBuffer a(n);
    for (int i = 0; i < n; ++i) {
        a.re[plan.reversed[i]] = frame[i];
        a.im[plan.reversed[i]] = 0.0;
    }
    plan.apply_permuted(a);
    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k) out[k] = {a.re[k], a.im[k]};
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel filterbank
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<double>> build_mel_filterbank(int sr, int n_fft,
                                                      int n_mels, double f_min,
                                                      double f_max) {
    const int n_bins = n_fft / 2 + 1;

    // FFT bin center frequencies
    std::vector<double> fft_freqs(n_bins);
    for (int k = 0; k < n_bins; ++k)
        fft_freqs[k] = static_cast<double>(k) * sr / static_cast<double>(n_fft);

    // n_mels+2 equally-spaced mel points
    double mel_min = hz_to_mel(f_min);
    double mel_max = hz_to_mel(f_max);
    std::vector<double> hz_points(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; ++i)
        hz_points[i] =
            mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1));

    // Build triangular filters with Slaney area normalization
    std::vector<std::vector<double>> weights(n_mels,
                                             std::vector<double>(n_bins, 0.0));

    for (int m = 0; m < n_mels; ++m) {
        double f_l = hz_points[m];
        double f_c = hz_points[m + 1];
        double f_r = hz_points[m + 2];
        double enorm = 2.0 / (f_r - f_l);

        for (int k = 0; k < n_bins; ++k) {
            double rising = (fft_freqs[k] - f_l) / (f_c - f_l);
            double falling = (f_r - fft_freqs[k]) / (f_r - f_c);
            double val = std::max(0.0, std::min(rising, falling));
            weights[m][k] = val * enorm;
        }
    }
    return weights;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel spectrogram computation
// ─────────────────────────────────────────────────────────────────────────────

void compute_melspectrogram_cpu(const std::vector<double>& y, int sr, int n_fft,
                                int hop_length, int n_mels, double f_min,
                                double f_max, std::vector<double>& output) {
    const int n_samples = static_cast<int>(y.size());
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    const int n_bins = n_fft / 2 + 1;

    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    // Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n/n_fft))
    std::vector<double> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] =
            0.5 * (1.0 - std::cos(2.0 * M_PI * i / static_cast<double>(n_fft)));

    auto mel_fb = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    // Triangular filters have contiguous support. Skip only exact zeros.
    std::vector<int> first(n_mels), last(n_mels, n_bins);
    int support_first = n_bins, support_last = 0;
    for (int m = 0; m < n_mels; ++m) {
        while (first[m] < n_bins && mel_fb[m][first[m]] == 0.0) ++first[m];
        while (last[m] > first[m] && mel_fb[m][last[m] - 1] == 0.0) --last[m];
        if (first[m] < last[m]) {
            support_first = std::min(support_first, first[m]);
            support_last = std::max(support_last, last[m]);
        }
    }

    const int half = std::max(1, n_fft / 2);
    FFTPlan plan(half);
#if MELSPEC_FP32_BATCH == 8 || MELSPEC_FP32_BATCH == 16
    FFTPlan32 plan32(plan);
#endif
    // -i/2 * exp(-2*pi*i*k/n_fft), for real-FFT reconstruction.
    std::vector<double> recovery_re(n_bins), recovery_im(n_bins);
    for (int k = 1; k < n_fft / 2; ++k) {
        double ang = -2.0 * M_PI * k / static_cast<double>(n_fft);
        recovery_re[k] = 0.5 * std::sin(ang);
        recovery_im[k] = -0.5 * std::cos(ang);
    }
    output.resize(static_cast<size_t>(n_mels) * n_frames);

    // Keep short inputs single-threaded, and use at most the judge's 16 cores.
    const int n_threads = std::min({16,
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency())),
        std::max(1, n_frames / 256)});
    // Allocate scratch before launching threads, so workers cannot throw.
    std::vector<ComplexBuffer> fft_buffers(n_threads, ComplexBuffer(half));
    std::vector<std::vector<double>> power_buffers(
        n_threads, std::vector<double>(n_bins));
    auto process_single = [&](int t, ComplexBuffer& fft_out, std::vector<double>& power) {
        const int start = t * hop_length;
        if (n_fft == 1) {
            double sample = y[start] * window[0];
            power[0] = sample * sample;
        } else {
            for (int i = 0; i < half; ++i) {
                fft_out.re[plan.reversed[i]] = y[start + 2 * i] * window[2 * i];
                fft_out.im[plan.reversed[i]] =
                    y[start + 2 * i + 1] * window[2 * i + 1];
            }
            plan.apply_permuted(fft_out);
            recover_power(fft_out, recovery_re, recovery_im, power);
        }
        for (int m = 0; m < n_mels; ++m) {
            double acc = 0.0;
            for (int k = first[m]; k < last[m]; ++k)
                acc += mel_fb[m][k] * power[k];
            output[static_cast<size_t>(m) * n_frames + t] = acc;
        }
    };

    int tail_begin = 0;
#if MELSPEC_FP32_BATCH == 8
    std::vector<Batch8fBuffer> batch_buffers;
    if (n_fft > 1)
        batch_buffers.assign(n_threads, Batch8fBuffer(half));
    const int n_batches = n_fft > 1 ? n_frames / 8 : 0;
    tail_begin = n_batches * 8;
#elif MELSPEC_FP32_BATCH == 16 && defined(__AVX512F__)
    std::vector<Batch16fBuffer> batch_buffers;
    if (n_fft > 1)
        batch_buffers.assign(n_threads, Batch16fBuffer(half));
    const int n_batches = n_fft > 1 ? n_frames / 16 : 0;
    tail_begin = n_batches * 16;
#elif defined(__AVX512F__)
    std::vector<Batch8Buffer> batch_buffers;
    if (n_fft > 1)
        batch_buffers.assign(n_threads, Batch8Buffer(half));
    const int n_batches = n_fft > 1 ? n_frames / 8 : 0;
    tail_begin = n_batches * 8;
#elif defined(__AVX2__) && defined(__FMA__)
    std::vector<Batch4Buffer> batch_buffers;
    if (n_fft > 1)
        batch_buffers.assign(n_threads, Batch4Buffer(half));
    const int n_batches = n_fft > 1 ? n_frames / 4 : 0;
    tail_begin = n_batches * 4;
#endif
    auto worker = [&](int id) {
#if MELSPEC_FP32_BATCH == 8
        auto& a = batch_buffers[id];
        const int begin = static_cast<long long>(n_batches) * id / n_threads;
        const int end = static_cast<long long>(n_batches) * (id + 1) / n_threads;
        for (int batch = begin; batch < end; ++batch) {
            const int t = batch * 8;
            const int start = t * hop_length;
            for (int i = 0; i < half; ++i) {
                const int p = plan.reversed[i];
                const double we = window[2 * i], wo = window[2 * i + 1];
                store8f(a.re[p], _mm256_setr_ps(
                    static_cast<float>(y[start + 2 * i] * we),
                    static_cast<float>(y[start + hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 2 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 3 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 4 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 5 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 6 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 7 * hop_length + 2 * i] * we)));
                store8f(a.im[p], _mm256_setr_ps(
                    static_cast<float>(y[start + 2 * i + 1] * wo),
                    static_cast<float>(y[start + hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 2 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 3 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 4 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 5 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 6 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 7 * hop_length + 2 * i + 1] * wo)));
            }
            apply_batch8f(plan32, a);
            recover_power_batch8f(a, recovery_re, recovery_im, support_first, support_last);
            for (int m = 0; m < n_mels; ++m) {
                auto acc_lo = _mm256_setzero_pd();
                auto acc_hi = _mm256_setzero_pd();
                for (int k = first[m]; k < last[m]; ++k) {
                    auto p = load8f(a.power[k]);
                    auto weight = _mm256_set1_pd(mel_fb[m][k]);
                    acc_lo = _mm256_fmadd_pd(weight,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(p)), acc_lo);
                    acc_hi = _mm256_fmadd_pd(weight,
                        _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1)), acc_hi);
                }
                _mm256_storeu_pd(output.data() + static_cast<size_t>(m) * n_frames + t, acc_lo);
                _mm256_storeu_pd(output.data() + static_cast<size_t>(m) * n_frames + t + 4, acc_hi);
            }
        }
#elif MELSPEC_FP32_BATCH == 16 && defined(__AVX512F__)
        auto& a = batch_buffers[id];
        const int begin = static_cast<long long>(n_batches) * id / n_threads;
        const int end = static_cast<long long>(n_batches) * (id + 1) / n_threads;
        for (int batch = begin; batch < end; ++batch) {
            const int t = batch * 16;
            const int start = t * hop_length;
            for (int i = 0; i < half; ++i) {
                const int p = plan.reversed[i];
                const double we = window[2 * i], wo = window[2 * i + 1];
                store16f(a.re[p], _mm512_setr_ps(
                    static_cast<float>(y[start + 2 * i] * we),
                    static_cast<float>(y[start + hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 2 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 3 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 4 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 5 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 6 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 7 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 8 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 9 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 10 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 11 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 12 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 13 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 14 * hop_length + 2 * i] * we),
                    static_cast<float>(y[start + 15 * hop_length + 2 * i] * we)));
                store16f(a.im[p], _mm512_setr_ps(
                    static_cast<float>(y[start + 2 * i + 1] * wo),
                    static_cast<float>(y[start + hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 2 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 3 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 4 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 5 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 6 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 7 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 8 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 9 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 10 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 11 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 12 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 13 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 14 * hop_length + 2 * i + 1] * wo),
                    static_cast<float>(y[start + 15 * hop_length + 2 * i + 1] * wo)));
            }
            apply_batch16f(plan32, a);
            recover_power_batch16f(a, recovery_re, recovery_im, support_first, support_last);
            for (int m = 0; m < n_mels; ++m) {
                auto acc0 = _mm256_setzero_pd();
                auto acc1 = _mm256_setzero_pd();
                auto acc2 = _mm256_setzero_pd();
                auto acc3 = _mm256_setzero_pd();
                for (int k = first[m]; k < last[m]; ++k) {
                    auto p = load16f(a.power[k]);
                    auto weight = _mm256_set1_pd(mel_fb[m][k]);
                    acc0 = _mm256_fmadd_pd(weight,
                        _mm256_cvtps_pd(_mm512_castps512_ps128(p)), acc0);
                    acc1 = _mm256_fmadd_pd(weight,
                        _mm256_cvtps_pd(_mm512_extractf32x4_ps(p, 1)), acc1);
                    acc2 = _mm256_fmadd_pd(weight,
                        _mm256_cvtps_pd(_mm512_extractf32x4_ps(p, 2)), acc2);
                    acc3 = _mm256_fmadd_pd(weight,
                        _mm256_cvtps_pd(_mm512_extractf32x4_ps(p, 3)), acc3);
                }
                double* out = output.data() + static_cast<size_t>(m) * n_frames + t;
                _mm256_storeu_pd(out, acc0);
                _mm256_storeu_pd(out + 4, acc1);
                _mm256_storeu_pd(out + 8, acc2);
                _mm256_storeu_pd(out + 12, acc3);
            }
        }
#elif defined(__AVX512F__)
        auto& a = batch_buffers[id];
        const int begin = static_cast<long long>(n_batches) * id / n_threads;
        const int end = static_cast<long long>(n_batches) * (id + 1) / n_threads;
        for (int batch = begin; batch < end; ++batch) {
            const int t = batch * 8;
            const int start = t * hop_length;
            for (int i = 0; i < half; ++i) {
                const int p = plan.reversed[i];
                const double we = window[2 * i], wo = window[2 * i + 1];
                store8(a.re[p], _mm512_setr_pd(
                    y[start + 2 * i] * we,
                    y[start + hop_length + 2 * i] * we,
                    y[start + 2 * hop_length + 2 * i] * we,
                    y[start + 3 * hop_length + 2 * i] * we,
                    y[start + 4 * hop_length + 2 * i] * we,
                    y[start + 5 * hop_length + 2 * i] * we,
                    y[start + 6 * hop_length + 2 * i] * we,
                    y[start + 7 * hop_length + 2 * i] * we));
                store8(a.im[p], _mm512_setr_pd(
                    y[start + 2 * i + 1] * wo,
                    y[start + hop_length + 2 * i + 1] * wo,
                    y[start + 2 * hop_length + 2 * i + 1] * wo,
                    y[start + 3 * hop_length + 2 * i + 1] * wo,
                    y[start + 4 * hop_length + 2 * i + 1] * wo,
                    y[start + 5 * hop_length + 2 * i + 1] * wo,
                    y[start + 6 * hop_length + 2 * i + 1] * wo,
                    y[start + 7 * hop_length + 2 * i + 1] * wo));
            }
            apply_batch8(plan, a);
            recover_power_batch8(a, recovery_re, recovery_im, support_first, support_last);
            for (int m = 0; m < n_mels; ++m) {
                auto acc = _mm512_setzero_pd();
                for (int k = first[m]; k < last[m]; ++k)
                    acc = _mm512_fmadd_pd(_mm512_set1_pd(mel_fb[m][k]),
                                          load8(a.re[k]), acc);
                _mm512_storeu_pd(output.data() + static_cast<size_t>(m) * n_frames + t, acc);
            }
        }
#elif defined(__AVX2__) && defined(__FMA__)
        auto& a = batch_buffers[id];
        const int begin = static_cast<long long>(n_batches) * id / n_threads;
        const int end = static_cast<long long>(n_batches) * (id + 1) / n_threads;
        for (int batch = begin; batch < end; ++batch) {
            const int t = batch * 4;
            const int start = t * hop_length;
            for (int i = 0; i < half; ++i) {
                const int p = plan.reversed[i];
                const double we = window[2 * i], wo = window[2 * i + 1];
                store4(a.re[p], _mm256_setr_pd(
                    y[start + 2 * i] * we,
                    y[start + hop_length + 2 * i] * we,
                    y[start + 2 * hop_length + 2 * i] * we,
                    y[start + 3 * hop_length + 2 * i] * we));
                store4(a.im[p], _mm256_setr_pd(
                    y[start + 2 * i + 1] * wo,
                    y[start + hop_length + 2 * i + 1] * wo,
                    y[start + 2 * hop_length + 2 * i + 1] * wo,
                    y[start + 3 * hop_length + 2 * i + 1] * wo));
            }
            apply_batch4(plan, a);
            recover_power_batch4(a, recovery_re, recovery_im, support_first, support_last);
            for (int m = 0; m < n_mels; ++m) {
                auto acc = _mm256_setzero_pd();
                for (int k = first[m]; k < last[m]; ++k)
                    acc = _mm256_fmadd_pd(_mm256_set1_pd(mel_fb[m][k]),
                                          load4(a.re[k]), acc);
                _mm256_storeu_pd(output.data() + static_cast<size_t>(m) * n_frames + t, acc);
            }
        }
#else
        (void)id;
#endif
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    int launched = 1;
    try {
        for (; launched < n_threads; ++launched)
            threads.emplace_back(worker, launched);
    } catch (const std::system_error&) {
        // If resources are limited, finish unlaunched partitions locally.
    }
    worker(0);
    for (int id = launched; id < n_threads; ++id) worker(id);
    for (auto& thread : threads) thread.join();
    for (int t = tail_begin; t < n_frames; ++t)
        process_single(t, fft_buffers[0], power_buffers[0]);
}
