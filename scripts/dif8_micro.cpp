// 内部八帧 AVX512F DIT/DIF FFT microbenchmark；不作为正式 Melspec 程序链接。
// DIF 使用自然顺序输入并产生 bit-reversed 频谱；DIT 使用当前输入重排。
#include "../src/melspec.cpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#if defined(__AVX512F__)

struct alignas(64) DifLane8 {
    std::array<double, 8> values;
};

struct DifBatch8Buffer {
    std::vector<DifLane8> re, im;
    explicit DifBatch8Buffer(int n) : re(n), im(n) {}
};

static inline __m512d dif_load8(const DifLane8& value) {
    return _mm512_load_pd(value.values.data());
}

static inline void dif_store8(DifLane8& destination, __m512d value) {
    _mm512_store_pd(destination.values.data(), value);
}

static inline double dif_input(const std::vector<double>& input, int frame, int i) {
    const int n = static_cast<int>(input.size());
    return input[(i + frame) % n] * (1.0 + 1e-9 * frame);
}

// 与当前 DIT Batch=8 的 pack 工作相同，仅将 reversed 散写改为自然写。
static void pack_natural_batch8(const std::vector<double>& input, int frame,
                                DifBatch8Buffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        DifLane8 re_lane, im_lane;
        for (int lane = 0; lane < 8; ++lane) {
            const double value = dif_input(input, frame + lane, i);
            re_lane.values[lane] = value;
            im_lane.values[lane] = value * 0.125;
        }
        a.re[i] = re_lane;
        a.im[i] = im_lane;
    }
}

static void pack_reversed_batch8(const FFTPlan& plan, const std::vector<double>& input,
                                 int frame, DifBatch8Buffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        DifLane8 re_lane, im_lane;
        for (int lane = 0; lane < 8; ++lane) {
            const double value = dif_input(input, frame + lane, i);
            re_lane.values[lane] = value;
            im_lane.values[lane] = value * 0.125;
        }
        const int p = plan.reversed[i];
        a.re[p] = re_lane;
        a.im[p] = im_lane;
    }
}

// 与 src/melspec.cpp 的 apply_batch8 使用相同 stage/twiddle 和算术顺序。
static void apply_dit_batch8(const FFTPlan& plan, DifBatch8Buffer& a) {
    const int n = static_cast<int>(a.re.size());
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            const auto rr = dif_load8(a.re[i]), ri = dif_load8(a.im[i]);
            dif_store8(a.re[i], _mm512_add_pd(rr, dif_load8(a.re[i + 1])));
            dif_store8(a.im[i], _mm512_add_pd(ri, dif_load8(a.im[i + 1])));
            dif_store8(a.re[i + 1], _mm512_sub_pd(rr, dif_load8(a.re[i + 1])));
            dif_store8(a.im[i + 1], _mm512_sub_pd(ri, dif_load8(a.im[i + 1])));
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            if (quarter == 1) {
                const auto ar = dif_load8(a.re[base]), ai = dif_load8(a.im[base]);
                const auto br = dif_load8(a.re[base + 1]), bi = dif_load8(a.im[base + 1]);
                const auto cr = dif_load8(a.re[base + 2]), ci = dif_load8(a.im[base + 2]);
                const auto dr = dif_load8(a.re[base + 3]), di = dif_load8(a.im[base + 3]);
                const auto sum01r = _mm512_add_pd(ar, br);
                const auto sum01i = _mm512_add_pd(ai, bi);
                const auto dif01r = _mm512_sub_pd(ar, br);
                const auto dif01i = _mm512_sub_pd(ai, bi);
                const auto sum23r = _mm512_add_pd(cr, dr);
                const auto sum23i = _mm512_add_pd(ci, di);
                const auto dif23r = _mm512_sub_pd(cr, dr);
                const auto dif23i = _mm512_sub_pd(ci, di);
                dif_store8(a.re[base], _mm512_add_pd(sum01r, sum23r));
                dif_store8(a.im[base], _mm512_add_pd(sum01i, sum23i));
                dif_store8(a.re[base + 1], _mm512_add_pd(dif01r, dif23i));
                dif_store8(a.im[base + 1], _mm512_sub_pd(dif01i, dif23r));
                dif_store8(a.re[base + 2], _mm512_sub_pd(sum01r, sum23r));
                dif_store8(a.im[base + 2], _mm512_sub_pd(sum01i, sum23i));
                dif_store8(a.re[base + 3], _mm512_sub_pd(dif01r, dif23i));
                dif_store8(a.im[base + 3], _mm512_add_pd(dif01i, dif23r));
                continue;
            }
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                const auto ar = dif_load8(a.re[p]), ai = dif_load8(a.im[p]);
                const auto br = dif_load8(a.re[p + quarter]), bi = dif_load8(a.im[p + quarter]);
                const auto cr = dif_load8(a.re[p + 2 * quarter]), ci = dif_load8(a.im[p + 2 * quarter]);
                const auto dr = dif_load8(a.re[p + 3 * quarter]), di = dif_load8(a.im[p + 3 * quarter]);
                const auto w1r = _mm512_set1_pd(stage.w1re[j]);
                const auto w1i = _mm512_set1_pd(stage.w1im[j]);
                const auto w2r = _mm512_set1_pd(stage.w2re[j]);
                const auto w2i = _mm512_set1_pd(stage.w2im[j]);
                const auto w3r = _mm512_set1_pd(stage.w3re[j]);
                const auto w3i = _mm512_set1_pd(stage.w3im[j]);
                const auto br2 = _mm512_fmsub_pd(br, w2r, _mm512_mul_pd(bi, w2i));
                const auto bi2 = _mm512_fmadd_pd(br, w2i, _mm512_mul_pd(bi, w2r));
                const auto cr1 = _mm512_fmsub_pd(cr, w1r, _mm512_mul_pd(ci, w1i));
                const auto ci1 = _mm512_fmadd_pd(cr, w1i, _mm512_mul_pd(ci, w1r));
                const auto dr3 = _mm512_fmsub_pd(dr, w3r, _mm512_mul_pd(di, w3i));
                const auto di3 = _mm512_fmadd_pd(dr, w3i, _mm512_mul_pd(di, w3r));
                const auto sum01r = _mm512_add_pd(ar, br2);
                const auto sum01i = _mm512_add_pd(ai, bi2);
                const auto dif01r = _mm512_sub_pd(ar, br2);
                const auto dif01i = _mm512_sub_pd(ai, bi2);
                const auto sum23r = _mm512_add_pd(cr1, dr3);
                const auto sum23i = _mm512_add_pd(ci1, di3);
                const auto dif23r = _mm512_sub_pd(cr1, dr3);
                const auto dif23i = _mm512_sub_pd(ci1, di3);
                dif_store8(a.re[p], _mm512_add_pd(sum01r, sum23r));
                dif_store8(a.im[p], _mm512_add_pd(sum01i, sum23i));
                dif_store8(a.re[p + quarter], _mm512_add_pd(dif01r, dif23i));
                dif_store8(a.im[p + quarter], _mm512_sub_pd(dif01i, dif23r));
                dif_store8(a.re[p + 2 * quarter], _mm512_sub_pd(sum01r, sum23r));
                dif_store8(a.im[p + 2 * quarter], _mm512_sub_pd(sum01i, sum23i));
                dif_store8(a.re[p + 3 * quarter], _mm512_sub_pd(dif01r, dif23i));
                dif_store8(a.im[p + 3 * quarter], _mm512_add_pd(dif01i, dif23r));
            }
        }
    }
}

// 独立的 radix-2 DIF twiddle 表：W_len^j 存在 [len/2 + j]。
// 只在预计算时调用 trig，FFT 蝴蝶中不调用 trig。
struct DifTwiddles {
    std::vector<double> re, im;
    explicit DifTwiddles(int n) : re(n), im(n) {
        for (int len = 2; len <= n; len <<= 1) {
            const double angle = -2.0 * M_PI / static_cast<double>(len);
            const double step_re = std::cos(angle);
            const double step_im = std::sin(angle);
            double w_re = 1.0, w_im = 0.0;
            for (int j = 0; j < len / 2; ++j) {
                re[len / 2 + j] = w_re;
                im[len / 2 + j] = w_im;
                const double next_re = w_re * step_re - w_im * step_im;
                w_im = w_re * step_im + w_im * step_re;
                w_re = next_re;
            }
        }
    }
};

// 标准 radix-2 DIF：自然输入，len 从 n 递减到 2，输出为 base-2 bit-reversed。
static void apply_dif_batch8(const DifTwiddles& twiddles, DifBatch8Buffer& a) {
    const int n = static_cast<int>(a.re.size());
    for (int len = n; len >= 2; len >>= 1) {
        const int half = len / 2;
        for (int base = 0; base < n; base += len) {
            for (int j = 0; j < half; ++j) {
                const int p = base + j;
                const auto ur = dif_load8(a.re[p]), ui = dif_load8(a.im[p]);
                const auto vr = dif_load8(a.re[p + half]), vi = dif_load8(a.im[p + half]);
                const auto dr = _mm512_sub_pd(ur, vr);
                const auto di = _mm512_sub_pd(ui, vi);
                const auto wr = _mm512_set1_pd(twiddles.re[half + j]);
                const auto wi = _mm512_set1_pd(twiddles.im[half + j]);
                dif_store8(a.re[p], _mm512_add_pd(ur, vr));
                dif_store8(a.im[p], _mm512_add_pd(ui, vi));
                dif_store8(a.re[p + half],
                           _mm512_fmsub_pd(dr, wr, _mm512_mul_pd(di, wi)));
                dif_store8(a.im[p + half],
                           _mm512_fmadd_pd(dr, wi, _mm512_mul_pd(di, wr)));
            }
        }
    }
}

// 将 DIF 的 bit-reversed 输出按 reversed 表恢复为自然频谱。
static void reorder_dif_output(const FFTPlan& plan, DifBatch8Buffer& a) {
    const int n = static_cast<int>(a.re.size());
    for (int i = 0; i < n; ++i) {
        const int p = plan.reversed[i];
        if (p <= i) continue;
        std::swap(a.re[i], a.re[p]);
        std::swap(a.im[i], a.im[p]);
    }
}

static double validate_dif(const FFTPlan& plan, const DifTwiddles& twiddles,
                           const std::vector<double>& input) {
    const int n = static_cast<int>(input.size());
    DifBatch8Buffer dit(n), dif(n);
    pack_reversed_batch8(plan, input, 0, dit);
    pack_natural_batch8(input, 0, dif);
    apply_dit_batch8(plan, dit);
    apply_dif_batch8(twiddles, dif);

    double max_error = 0.0;
    double scale = 1.0;
    for (int i = 0; i < n; ++i) {
        const int p = plan.reversed[i];
        for (int lane = 0; lane < 8; ++lane) {
            const double expected_re = dit.re[i].values[lane];
            const double expected_im = dit.im[i].values[lane];
            const double error_re = std::abs(expected_re - dif.re[p].values[lane]);
            const double error_im = std::abs(expected_im - dif.im[p].values[lane]);
            if (!std::isfinite(error_re) || !std::isfinite(error_im))
                return std::numeric_limits<double>::infinity();
            max_error = std::max({max_error, error_re, error_im});
            scale = std::max({scale, std::abs(expected_re), std::abs(expected_im)});
        }
    }
    return max_error / scale;
}

using DifClock = std::chrono::steady_clock;

static double dif_seconds(DifClock::time_point begin, DifClock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

#endif

int main(int argc, char** argv) {
#if !defined(__AVX512F__)
    (void)argc;
    (void)argv;
    std::cout << "AVX512F is required; DIF batch8 benchmark was not built.\n";
    return 0;
#else
    int n = 4096;
    int frames = 20000;
    int repeats = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) return 2;
        const int value = std::atoi(argv[++i]);
        if (key == "--n") n = value;
        else if (key == "--frames") frames = value;
        else if (key == "--repeats") repeats = value;
        else return 2;
    }
    if ((n != 4096 && n != 8192) || frames < 8 || frames % 8 != 0 || repeats < 1) {
        std::cerr << "usage: dif8_micro --n {4096|8192} --frames MULTIPLE_OF_8 --repeats N\n";
        return 2;
    }

    FFTPlan plan(n);
    DifTwiddles dif_twiddles(n);
    DifBatch8Buffer dit(n), dif(n), reordered(n);
    std::vector<double> input(n);
    for (int i = 0; i < n; ++i)
        input[i] = 0.25 + 0.5 * std::sin(0.017 * i) + 0.1 * std::cos(0.071 * i);

    const double validation_error = validate_dif(plan, dif_twiddles, input);
    std::cout << "validation_dif_vs_reordered_dit scaled_max_error="
              << std::setprecision(12) << validation_error << '\n';
    if (!(validation_error <= 1e-9)) {
        std::cerr << "DIF numerical validation failed\n";
        return 1;
    }

    std::cout << "mode,n,repeat,pack_ns_per_frame,fft_ns_per_frame,total_ns_per_frame,"
                 "reorder_ns_per_frame,fft_plus_reorder_ns_per_frame,checksum\n";
    for (int rep = 0; rep < repeats; ++rep) {
        double dit_pack = 0.0, dit_fft = 0.0;
        volatile double dit_checksum = 0.0;
        for (int frame = 0; frame < frames; frame += 8) {
            const auto begin = DifClock::now();
            pack_reversed_batch8(plan, input, frame, dit);
            const auto packed = DifClock::now();
            apply_dit_batch8(plan, dit);
            const auto end = DifClock::now();
            dit_pack += dif_seconds(begin, packed);
            dit_fft += dif_seconds(packed, end);
            for (int lane = 0; lane < 8; ++lane) {
                const int current = frame + lane;
                dit_checksum += dit.re[current % n].values[lane]
                    + dit.im[(current * 7) % n].values[lane];
            }
        }

        double dif_pack = 0.0, dif_fft = 0.0;
        volatile double dif_checksum = 0.0;
        for (int frame = 0; frame < frames; frame += 8) {
            const auto begin = DifClock::now();
            pack_natural_batch8(input, frame, dif);
            const auto packed = DifClock::now();
            apply_dif_batch8(dif_twiddles, dif);
            const auto end = DifClock::now();
            dif_pack += dif_seconds(begin, packed);
            dif_fft += dif_seconds(packed, end);
            for (int lane = 0; lane < 8; ++lane) {
                const int current = frame + lane;
                dif_checksum += dif.re[plan.reversed[current % n]].values[lane]
                    + dif.im[plan.reversed[(current * 7) % n]].values[lane];
            }
        }

        double reorder = 0.0;
        volatile double reorder_checksum = 0.0;
        for (int frame = 0; frame < frames; frame += 8) {
            const auto begin = DifClock::now();
            reorder_dif_output(plan, dif);
            const auto end = DifClock::now();
            reorder += dif_seconds(begin, end);
            // 再反向排列，保持输出计时独立于前一批的内容并维持有效 checksum。
            reorder_dif_output(plan, dif);
            for (int lane = 0; lane < 8; ++lane) {
                const int current = frame + lane;
                reorder_checksum += dif.re[plan.reversed[current % n]].values[lane]
                    + dif.im[plan.reversed[(current * 7) % n]].values[lane];
            }
        }

        std::cout << "dit_bitrev_pack," << n << ',' << rep << ',' << std::setprecision(12)
                  << dit_pack * 1e9 / frames << ',' << dit_fft * 1e9 / frames << ','
                  << (dit_pack + dit_fft) * 1e9 / frames << ",0,0,"
                  << dit_checksum << '\n';
        std::cout << "dif_natural_pack," << n << ',' << rep << ','
                  << dif_pack * 1e9 / frames << ',' << dif_fft * 1e9 / frames << ','
                  << (dif_pack + dif_fft) * 1e9 / frames << ','
                  << reorder * 1e9 / frames << ','
                  << (dif_fft + reorder) * 1e9 / frames << ',' << dif_checksum << '\n';
        std::cout << "dif_reorder_checksum," << n << ',' << rep << ",0,0,0,0,0,"
                  << reorder_checksum << '\n';
    }
    return 0;
#endif
}
