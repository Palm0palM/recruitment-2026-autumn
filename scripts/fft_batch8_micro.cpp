// 内部八帧 AVX512F FFT microbenchmark；不作为正式 Melspec 程序链接。
#include "../src/melspec.cpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#if defined(__AVX512F__)

struct alignas(64) MicroLane8 {
    std::array<double, 8> values;
};

struct MicroBatch8Buffer {
    std::vector<MicroLane8> re, im;
    explicit MicroBatch8Buffer(int n) : re(n), im(n) {}
};

static inline __m512d micro_load8(const MicroLane8& value) {
    return _mm512_load_pd(value.values.data());
}

static inline void micro_store8(MicroLane8& destination, __m512d value) {
    _mm512_store_pd(destination.values.data(), value);
}

static inline double batch8_input(const std::vector<double>& input, int frame, int i) {
    const int n = static_cast<int>(input.size());
    return input[(i + frame) % n] * (1.0 + 1e-9 * frame);
}

static void pack_single(const FFTPlan& plan, const std::vector<double>& input,
                        int frame, ComplexBuffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        const double value = batch8_input(input, frame, i);
        a.re[plan.reversed[i]] = value;
        a.im[plan.reversed[i]] = value * 0.125;
    }
}

static void pack_batch8(const FFTPlan& plan, const std::vector<double>& input,
                        int frame, MicroBatch8Buffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        MicroLane8 re_lane, im_lane;
        for (int lane = 0; lane < 8; ++lane) {
            const double value = batch8_input(input, frame + lane, i);
            re_lane.values[lane] = value;
            im_lane.values[lane] = value * 0.125;
        }
        const int p = plan.reversed[i];
        a.re[p] = re_lane;
        a.im[p] = im_lane;
    }
}

static void micro_apply_batch8(const FFTPlan& plan, MicroBatch8Buffer& a) {
    const int n = static_cast<int>(a.re.size());
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            const auto rr = micro_load8(a.re[i]), ri = micro_load8(a.im[i]);
            micro_store8(a.re[i], _mm512_add_pd(rr, micro_load8(a.re[i + 1])));
            micro_store8(a.im[i], _mm512_add_pd(ri, micro_load8(a.im[i + 1])));
            micro_store8(a.re[i + 1], _mm512_sub_pd(rr, micro_load8(a.re[i + 1])));
            micro_store8(a.im[i + 1], _mm512_sub_pd(ri, micro_load8(a.im[i + 1])));
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            if (quarter == 1) {
                const auto ar = micro_load8(a.re[base]), ai = micro_load8(a.im[base]);
                const auto br = micro_load8(a.re[base + 1]), bi = micro_load8(a.im[base + 1]);
                const auto cr = micro_load8(a.re[base + 2]), ci = micro_load8(a.im[base + 2]);
                const auto dr = micro_load8(a.re[base + 3]), di = micro_load8(a.im[base + 3]);
                const auto sum01r = _mm512_add_pd(ar, br);
                const auto sum01i = _mm512_add_pd(ai, bi);
                const auto dif01r = _mm512_sub_pd(ar, br);
                const auto dif01i = _mm512_sub_pd(ai, bi);
                const auto sum23r = _mm512_add_pd(cr, dr);
                const auto sum23i = _mm512_add_pd(ci, di);
                const auto dif23r = _mm512_sub_pd(cr, dr);
                const auto dif23i = _mm512_sub_pd(ci, di);
                micro_store8(a.re[base], _mm512_add_pd(sum01r, sum23r));
                micro_store8(a.im[base], _mm512_add_pd(sum01i, sum23i));
                micro_store8(a.re[base + 1], _mm512_add_pd(dif01r, dif23i));
                micro_store8(a.im[base + 1], _mm512_sub_pd(dif01i, dif23r));
                micro_store8(a.re[base + 2], _mm512_sub_pd(sum01r, sum23r));
                micro_store8(a.im[base + 2], _mm512_sub_pd(sum01i, sum23i));
                micro_store8(a.re[base + 3], _mm512_sub_pd(dif01r, dif23i));
                micro_store8(a.im[base + 3], _mm512_add_pd(dif01i, dif23r));
                continue;
            }
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                const auto ar = micro_load8(a.re[p]), ai = micro_load8(a.im[p]);
                const auto br = micro_load8(a.re[p + quarter]), bi = micro_load8(a.im[p + quarter]);
                const auto cr = micro_load8(a.re[p + 2 * quarter]), ci = micro_load8(a.im[p + 2 * quarter]);
                const auto dr = micro_load8(a.re[p + 3 * quarter]), di = micro_load8(a.im[p + 3 * quarter]);
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
                micro_store8(a.re[p], _mm512_add_pd(sum01r, sum23r));
                micro_store8(a.im[p], _mm512_add_pd(sum01i, sum23i));
                micro_store8(a.re[p + quarter], _mm512_add_pd(dif01r, dif23i));
                micro_store8(a.im[p + quarter], _mm512_sub_pd(dif01i, dif23r));
                micro_store8(a.re[p + 2 * quarter], _mm512_sub_pd(sum01r, sum23r));
                micro_store8(a.im[p + 2 * quarter], _mm512_sub_pd(sum01i, sum23i));
                micro_store8(a.re[p + 3 * quarter], _mm512_sub_pd(dif01r, dif23i));
                micro_store8(a.im[p + 3 * quarter], _mm512_add_pd(dif01i, dif23r));
            }
        }
    }
}

static void micro_apply_batch8_jouter(const FFTPlan& plan, MicroBatch8Buffer& a) {
    const int n = static_cast<int>(a.re.size());
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            const auto rr = micro_load8(a.re[i]), ri = micro_load8(a.im[i]);
            micro_store8(a.re[i], _mm512_add_pd(rr, micro_load8(a.re[i + 1])));
            micro_store8(a.im[i], _mm512_add_pd(ri, micro_load8(a.im[i + 1])));
            micro_store8(a.re[i + 1], _mm512_sub_pd(rr, micro_load8(a.re[i + 1])));
            micro_store8(a.im[i + 1], _mm512_sub_pd(ri, micro_load8(a.im[i + 1])));
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        if (quarter == 1) {
            for (int base = 0; base < n; base += len) {
                const auto ar = micro_load8(a.re[base]), ai = micro_load8(a.im[base]);
                const auto br = micro_load8(a.re[base + 1]), bi = micro_load8(a.im[base + 1]);
                const auto cr = micro_load8(a.re[base + 2]), ci = micro_load8(a.im[base + 2]);
                const auto dr = micro_load8(a.re[base + 3]), di = micro_load8(a.im[base + 3]);
                const auto sum01r = _mm512_add_pd(ar, br);
                const auto sum01i = _mm512_add_pd(ai, bi);
                const auto dif01r = _mm512_sub_pd(ar, br);
                const auto dif01i = _mm512_sub_pd(ai, bi);
                const auto sum23r = _mm512_add_pd(cr, dr);
                const auto sum23i = _mm512_add_pd(ci, di);
                const auto dif23r = _mm512_sub_pd(cr, dr);
                const auto dif23i = _mm512_sub_pd(ci, di);
                micro_store8(a.re[base], _mm512_add_pd(sum01r, sum23r));
                micro_store8(a.im[base], _mm512_add_pd(sum01i, sum23i));
                micro_store8(a.re[base + 1], _mm512_add_pd(dif01r, dif23i));
                micro_store8(a.im[base + 1], _mm512_sub_pd(dif01i, dif23r));
                micro_store8(a.re[base + 2], _mm512_sub_pd(sum01r, sum23r));
                micro_store8(a.im[base + 2], _mm512_sub_pd(sum01i, sum23i));
                micro_store8(a.re[base + 3], _mm512_sub_pd(dif01r, dif23i));
                micro_store8(a.im[base + 3], _mm512_add_pd(dif01i, dif23r));
            }
            continue;
        }
        for (int j = 0; j < quarter; ++j) {
            const auto w1r = _mm512_set1_pd(stage.w1re[j]);
            const auto w1i = _mm512_set1_pd(stage.w1im[j]);
            const auto w2r = _mm512_set1_pd(stage.w2re[j]);
            const auto w2i = _mm512_set1_pd(stage.w2im[j]);
            const auto w3r = _mm512_set1_pd(stage.w3re[j]);
            const auto w3i = _mm512_set1_pd(stage.w3im[j]);
            for (int base = 0; base < n; base += len) {
                const int p = base + j;
                const auto ar = micro_load8(a.re[p]), ai = micro_load8(a.im[p]);
                const auto br = micro_load8(a.re[p + quarter]), bi = micro_load8(a.im[p + quarter]);
                const auto cr = micro_load8(a.re[p + 2 * quarter]), ci = micro_load8(a.im[p + 2 * quarter]);
                const auto dr = micro_load8(a.re[p + 3 * quarter]), di = micro_load8(a.im[p + 3 * quarter]);
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
                micro_store8(a.re[p], _mm512_add_pd(sum01r, sum23r));
                micro_store8(a.im[p], _mm512_add_pd(sum01i, sum23i));
                micro_store8(a.re[p + quarter], _mm512_add_pd(dif01r, dif23i));
                micro_store8(a.im[p + quarter], _mm512_sub_pd(dif01i, dif23r));
                micro_store8(a.re[p + 2 * quarter], _mm512_sub_pd(sum01r, sum23r));
                micro_store8(a.im[p + 2 * quarter], _mm512_sub_pd(sum01i, sum23i));
                micro_store8(a.re[p + 3 * quarter], _mm512_sub_pd(dif01r, dif23i));
                micro_store8(a.im[p + 3 * quarter], _mm512_add_pd(dif01i, dif23r));
            }
        }
    }
}

static double validate_batch8(const FFTPlan& plan, const std::vector<double>& input,
                              void (*apply)(const FFTPlan&, MicroBatch8Buffer&)) {
    const int n = static_cast<int>(input.size());
    ComplexBuffer single(n);
    MicroBatch8Buffer batch(n);
    pack_batch8(plan, input, 0, batch);
    apply(plan, batch);
    double max_error = 0.0;
    double scale = 1.0;
    for (int lane = 0; lane < 8; ++lane) {
        pack_single(plan, input, lane, single);
        plan.apply_permuted(single);
        for (int i = 0; i < n; ++i) {
            const double error_re = std::abs(single.re[i] - batch.re[i].values[lane]);
            const double error_im = std::abs(single.im[i] - batch.im[i].values[lane]);
            if (!std::isfinite(error_re) || !std::isfinite(error_im))
                return std::numeric_limits<double>::infinity();
            max_error = std::max({max_error, error_re, error_im});
            scale = std::max({scale, std::abs(single.re[i]), std::abs(single.im[i])});
        }
    }
    return max_error / scale;
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
    std::cout << "AVX512F is required; batch8 benchmark was not built.\n";
    return 0;
#else
    int n = 4096;
    int frames = 20000;
    int repeats = 5;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 >= argc) return 2;
        const int value = std::atoi(argv[++i]);
        if (key == "--n") n = value;
        else if (key == "--frames") frames = value;
        else if (key == "--repeats") repeats = value;
        else return 2;
    }
    if (n < 1 || (n & (n - 1)) != 0 || frames < 8 || frames % 8 != 0 || repeats < 1) {
        std::cerr << "usage: fft_batch8_micro --n POW2 --frames MULTIPLE_OF_8 --repeats N\n";
        return 2;
    }

    FFTPlan plan(n);
    ComplexBuffer single(n);
    MicroBatch8Buffer batch(n);
    std::vector<double> input(n);
    for (int i = 0; i < n; ++i)
        input[i] = 0.25 + 0.5 * std::sin(0.017 * i) + 0.1 * std::cos(0.071 * i);

    const double validation_error = validate_batch8(plan, input, micro_apply_batch8);
    const double jouter_validation_error = validate_batch8(plan, input, micro_apply_batch8_jouter);
    std::cout << "validation scaled_max_error=" << std::setprecision(12)
              << validation_error << '\n';
    std::cout << "validation_jouter scaled_max_error=" << std::setprecision(12)
              << jouter_validation_error << '\n';
    if (!(validation_error <= 1e-9)) {
        std::cerr << "batch8 numerical validation failed\n";
        return 1;
    }
    if (!(jouter_validation_error <= 1e-9)) {
        std::cerr << "batch8_jouter numerical validation failed\n";
        return 1;
    }

    std::cout << "mode,n,repeat,pack_ns_per_frame,fft_ns_per_frame,total_ns_per_frame,checksum\n";
    for (int rep = 0; rep < repeats; ++rep) {
        double single_pack = 0.0, single_fft = 0.0;
        volatile double single_checksum = 0.0;
        for (int frame = 0; frame < frames; ++frame) {
            auto begin = Clock::now();
            pack_single(plan, input, frame, single);
            auto packed = Clock::now();
            plan.apply_permuted(single);
            auto end = Clock::now();
            single_pack += seconds(begin, packed);
            single_fft += seconds(packed, end);
            single_checksum += single.re[frame % n] + single.im[(frame * 7) % n];
        }

        double batch_pack = 0.0, batch_fft = 0.0;
        volatile double batch_checksum = 0.0;
        for (int frame = 0; frame < frames; frame += 8) {
            auto begin = Clock::now();
            pack_batch8(plan, input, frame, batch);
            auto packed = Clock::now();
            micro_apply_batch8(plan, batch);
            auto end = Clock::now();
            batch_pack += seconds(begin, packed);
            batch_fft += seconds(packed, end);
            for (int lane = 0; lane < 8; ++lane) {
                const int current = frame + lane;
                batch_checksum += batch.re[current % n].values[lane]
                    + batch.im[(current * 7) % n].values[lane];
            }
        }

        double jouter_pack = 0.0, jouter_fft = 0.0;
        volatile double jouter_checksum = 0.0;
        for (int frame = 0; frame < frames; frame += 8) {
            auto begin = Clock::now();
            pack_batch8(plan, input, frame, batch);
            auto packed = Clock::now();
            micro_apply_batch8_jouter(plan, batch);
            auto end = Clock::now();
            jouter_pack += seconds(begin, packed);
            jouter_fft += seconds(packed, end);
            for (int lane = 0; lane < 8; ++lane) {
                const int current = frame + lane;
                jouter_checksum += batch.re[current % n].values[lane]
                    + batch.im[(current * 7) % n].values[lane];
            }
        }

        std::cout << "single," << n << ',' << rep << ',' << std::setprecision(12)
                  << single_pack * 1e9 / frames << ',' << single_fft * 1e9 / frames << ','
                  << (single_pack + single_fft) * 1e9 / frames << ','
                  << single_checksum << '\n';
        std::cout << "batch8," << n << ',' << rep << ',' << std::setprecision(12)
                  << batch_pack * 1e9 / frames << ',' << batch_fft * 1e9 / frames << ','
                  << (batch_pack + batch_fft) * 1e9 / frames << ','
                  << batch_checksum << '\n';
        std::cout << "batch8_jouter," << n << ',' << rep << ',' << std::setprecision(12)
                  << jouter_pack * 1e9 / frames << ',' << jouter_fft * 1e9 / frames << ','
                  << (jouter_pack + jouter_fft) * 1e9 / frames << ','
                  << jouter_checksum << '\n';
    }
    return 0;
#endif
}
