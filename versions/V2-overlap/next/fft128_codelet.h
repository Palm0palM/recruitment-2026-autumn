#pragma once

#include <cmath>
#include <complex>
#include <vector>
#include <immintrin.h>

// Fixed 128-point complex FFT, natural input, DIF radix-4 plus a final radix-2.
// Output is in mixed-radix digit order; perm[m] maps natural m to that order.
struct Codelet128Plan {
    double wr[128];
    double wi[128];
    int perm[128];

    Codelet128Plan() {
        for (int k = 0; k < 128; ++k) {
            double ang = -2.0 * M_PI * k / 128.0;
            wr[k] = std::cos(ang);
            wi[k] = std::sin(ang);
        }
        std::vector<int> order(1, 0);
        // DIF subproblem r emits k=r+radix*m. Build the map bottom-up.
        for (int radix : {2, 4, 4, 4}) {
            std::vector<int> next;
            for (int r = 0; r < radix; ++r)
                for (int m : order) next.push_back(r + radix * m);
            order = next;
        }
        for (int position = 0; position < 128; ++position)
            perm[order[position]] = position;
    }

    void apply(double* re, double* im) const {
        for (int len : {128, 32, 8}) {
            const int quarter = len / 4;
            const int step = 128 / len;
            for (int i = 0; i < 128; i += len) {
                int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
                for (; j + 4 <= quarter; j += 4) {
                    auto ar = _mm256_loadu_pd(re + i + j);
                    auto ai = _mm256_loadu_pd(im + i + j);
                    auto br = _mm256_loadu_pd(re + i + j + quarter);
                    auto bi = _mm256_loadu_pd(im + i + j + quarter);
                    auto cr = _mm256_loadu_pd(re + i + j + 2 * quarter);
                    auto ci = _mm256_loadu_pd(im + i + j + 2 * quarter);
                    auto dr = _mm256_loadu_pd(re + i + j + 3 * quarter);
                    auto di = _mm256_loadu_pd(im + i + j + 3 * quarter);
                    auto t0r = _mm256_add_pd(ar, cr), t0i = _mm256_add_pd(ai, ci);
                    auto t1r = _mm256_sub_pd(ar, cr), t1i = _mm256_sub_pd(ai, ci);
                    auto t2r = _mm256_add_pd(br, dr), t2i = _mm256_add_pd(bi, di);
                    // (b-d)*(-i)
                    auto t3r = _mm256_sub_pd(bi, di), t3i = _mm256_sub_pd(dr, br);
                    _mm256_storeu_pd(re + i + j, _mm256_add_pd(t0r, t2r));
                    _mm256_storeu_pd(im + i + j, _mm256_add_pd(t0i, t2i));
                    auto y1r = _mm256_add_pd(t1r, t3r), y1i = _mm256_add_pd(t1i, t3i);
                    auto y2r = _mm256_sub_pd(t0r, t2r), y2i = _mm256_sub_pd(t0i, t2i);
                    auto y3r = _mm256_sub_pd(t1r, t3r), y3i = _mm256_sub_pd(t1i, t3i);
                    auto w1r = _mm256_set_pd(wr[(j + 3) * step], wr[(j + 2) * step],
                                             wr[(j + 1) * step], wr[j * step]);
                    auto w1i = _mm256_set_pd(wi[(j + 3) * step], wi[(j + 2) * step],
                                             wi[(j + 1) * step], wi[j * step]);
                    auto w2r = _mm256_set_pd(wr[2 * (j + 3) * step], wr[2 * (j + 2) * step],
                                             wr[2 * (j + 1) * step], wr[2 * j * step]);
                    auto w2i = _mm256_set_pd(wi[2 * (j + 3) * step], wi[2 * (j + 2) * step],
                                             wi[2 * (j + 1) * step], wi[2 * j * step]);
                    auto w3r = _mm256_set_pd(wr[3 * (j + 3) * step], wr[3 * (j + 2) * step],
                                             wr[3 * (j + 1) * step], wr[3 * j * step]);
                    auto w3i = _mm256_set_pd(wi[3 * (j + 3) * step], wi[3 * (j + 2) * step],
                                             wi[3 * (j + 1) * step], wi[3 * j * step]);
                    auto complex_mul = [](__m256d xr, __m256d xi, __m256d wr,
                                          __m256d wi, double* re_out, double* im_out) {
                        _mm256_storeu_pd(re_out, _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi)));
                        _mm256_storeu_pd(im_out, _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr)));
                    };
                    complex_mul(y1r, y1i, w1r, w1i, re + i + j + quarter,
                                im + i + j + quarter);
                    complex_mul(y2r, y2i, w2r, w2i, re + i + j + 2 * quarter,
                                im + i + j + 2 * quarter);
                    complex_mul(y3r, y3i, w3r, w3i, re + i + j + 3 * quarter,
                                im + i + j + 3 * quarter);
                }
#endif
                for (; j < quarter; ++j) {
                    double ar = re[i + j], ai = im[i + j];
                    double br = re[i + j + quarter], bi = im[i + j + quarter];
                    double cr = re[i + j + 2 * quarter], ci = im[i + j + 2 * quarter];
                    double dr = re[i + j + 3 * quarter], di = im[i + j + 3 * quarter];
                    double t0r = ar + cr, t0i = ai + ci;
                    double t1r = ar - cr, t1i = ai - ci;
                    double t2r = br + dr, t2i = bi + di;
                    double t3r = bi - di, t3i = dr - br;
                    re[i + j] = t0r + t2r;
                    im[i + j] = t0i + t2i;
                    double y1r = t1r + t3r, y1i = t1i + t3i;
                    double y2r = t0r - t2r, y2i = t0i - t2i;
                    double y3r = t1r - t3r, y3i = t1i - t3i;
                    auto store = [&](int offset, double xr, double xi, int root) {
                        re[i + j + offset] = xr * wr[root] - xi * wi[root];
                        im[i + j + offset] = xr * wi[root] + xi * wr[root];
                    };
                    store(quarter, y1r, y1i, j * step);
                    store(2 * quarter, y2r, y2i, 2 * j * step);
                    store(3 * quarter, y3r, y3i, 3 * j * step);
                }
            }
        }
        for (int i = 0; i < 128; i += 2) {
            double ar = re[i], ai = im[i], br = re[i + 1], bi = im[i + 1];
            re[i] = ar + br; im[i] = ai + bi;
            re[i + 1] = ar - br; im[i + 1] = ai - bi;
        }
    }
};
