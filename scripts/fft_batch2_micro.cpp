// 内部双帧 FFT microbenchmark；不作为正式 Melspec 程序链接。
#include "../src/melspec.cpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#if defined(__SSE2__)

struct Batch2Buffer {
    std::vector<__m128d> re, im;
    explicit Batch2Buffer(int n) : re(n), im(n) {}
};

static inline double batch2_input(const std::vector<double>& input, int frame, int i) {
    const int n = static_cast<int>(input.size());
    return input[(i + frame) % n] * (1.0 + 1e-9 * frame);
}

static void pack_single(const FFTPlan& plan, const std::vector<double>& input,
                        int frame, ComplexBuffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        const double value = batch2_input(input, frame, i);
        a.re[plan.reversed[i]] = value;
        a.im[plan.reversed[i]] = value * 0.125;
    }
}

static void pack_batch2(const FFTPlan& plan, const std::vector<double>& input,
                        int frame, Batch2Buffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        const double v0 = batch2_input(input, frame, i);
        const double v1 = batch2_input(input, frame + 1, i);
        const int p = plan.reversed[i];
        a.re[p] = _mm_setr_pd(v0, v1);
        a.im[p] = _mm_setr_pd(v0 * 0.125, v1 * 0.125);
    }
}

static inline __m128d batch2_complex_re(__m128d xr, __m128d xi,
                                        __m128d wr, __m128d wi) {
#if defined(__FMA__)
    return _mm_fmsub_pd(xr, wr, _mm_mul_pd(xi, wi));
#else
    return _mm_sub_pd(_mm_mul_pd(xr, wr), _mm_mul_pd(xi, wi));
#endif
}

static inline __m128d batch2_complex_im(__m128d xr, __m128d xi,
                                        __m128d wr, __m128d wi) {
#if defined(__FMA__)
    return _mm_fmadd_pd(xr, wi, _mm_mul_pd(xi, wr));
#else
    return _mm_add_pd(_mm_mul_pd(xr, wi), _mm_mul_pd(xi, wr));
#endif
}

static void apply_batch2(const FFTPlan& plan, Batch2Buffer& a) {
    const int n = static_cast<int>(a.re.size());
    __m128d* re = a.re.data();
    __m128d* im = a.im.data();
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            const __m128d rr = re[i], ri = im[i];
            re[i] = _mm_add_pd(rr, re[i + 1]);
            im[i] = _mm_add_pd(ri, im[i + 1]);
            re[i + 1] = _mm_sub_pd(rr, re[i + 1]);
            im[i + 1] = _mm_sub_pd(ri, im[i + 1]);
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            if (quarter == 1) {
                const __m128d ar = re[base], ai = im[base];
                const __m128d br = re[base + 1], bi = im[base + 1];
                const __m128d cr = re[base + 2], ci = im[base + 2];
                const __m128d dr = re[base + 3], di = im[base + 3];
                const __m128d sum01r = _mm_add_pd(ar, br);
                const __m128d sum01i = _mm_add_pd(ai, bi);
                const __m128d dif01r = _mm_sub_pd(ar, br);
                const __m128d dif01i = _mm_sub_pd(ai, bi);
                const __m128d sum23r = _mm_add_pd(cr, dr);
                const __m128d sum23i = _mm_add_pd(ci, di);
                const __m128d dif23r = _mm_sub_pd(cr, dr);
                const __m128d dif23i = _mm_sub_pd(ci, di);
                re[base] = _mm_add_pd(sum01r, sum23r);
                im[base] = _mm_add_pd(sum01i, sum23i);
                re[base + 1] = _mm_add_pd(dif01r, dif23i);
                im[base + 1] = _mm_sub_pd(dif01i, dif23r);
                re[base + 2] = _mm_sub_pd(sum01r, sum23r);
                im[base + 2] = _mm_sub_pd(sum01i, sum23i);
                re[base + 3] = _mm_sub_pd(dif01r, dif23i);
                im[base + 3] = _mm_add_pd(dif01i, dif23r);
                continue;
            }
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                const __m128d ar = re[p], ai = im[p];
                const __m128d br = re[p + quarter], bi = im[p + quarter];
                const __m128d cr = re[p + 2 * quarter], ci = im[p + 2 * quarter];
                const __m128d dr = re[p + 3 * quarter], di = im[p + 3 * quarter];
                const __m128d w1r = _mm_set1_pd(stage.w1re[j]);
                const __m128d w1i = _mm_set1_pd(stage.w1im[j]);
                const __m128d w2r = _mm_set1_pd(stage.w2re[j]);
                const __m128d w2i = _mm_set1_pd(stage.w2im[j]);
                const __m128d w3r = _mm_set1_pd(stage.w3re[j]);
                const __m128d w3i = _mm_set1_pd(stage.w3im[j]);
                const __m128d br2 = batch2_complex_re(br, bi, w2r, w2i);
                const __m128d bi2 = batch2_complex_im(br, bi, w2r, w2i);
                const __m128d cr1 = batch2_complex_re(cr, ci, w1r, w1i);
                const __m128d ci1 = batch2_complex_im(cr, ci, w1r, w1i);
                const __m128d dr3 = batch2_complex_re(dr, di, w3r, w3i);
                const __m128d di3 = batch2_complex_im(dr, di, w3r, w3i);
                const __m128d sum01r = _mm_add_pd(ar, br2);
                const __m128d sum01i = _mm_add_pd(ai, bi2);
                const __m128d dif01r = _mm_sub_pd(ar, br2);
                const __m128d dif01i = _mm_sub_pd(ai, bi2);
                const __m128d sum23r = _mm_add_pd(cr1, dr3);
                const __m128d sum23i = _mm_add_pd(ci1, di3);
                const __m128d dif23r = _mm_sub_pd(cr1, dr3);
                const __m128d dif23i = _mm_sub_pd(ci1, di3);
                re[p] = _mm_add_pd(sum01r, sum23r);
                im[p] = _mm_add_pd(sum01i, sum23i);
                re[p + quarter] = _mm_add_pd(dif01r, dif23i);
                im[p + quarter] = _mm_sub_pd(dif01i, dif23r);
                re[p + 2 * quarter] = _mm_sub_pd(sum01r, sum23r);
                im[p + 2 * quarter] = _mm_sub_pd(sum01i, sum23i);
                re[p + 3 * quarter] = _mm_sub_pd(dif01r, dif23i);
                im[p + 3 * quarter] = _mm_add_pd(dif01i, dif23r);
            }
        }
    }
}

static double validate_batch2(const FFTPlan& plan, const std::vector<double>& input) {
    const int n = static_cast<int>(input.size());
    ComplexBuffer single(n);
    Batch2Buffer batch(n);
    pack_batch2(plan, input, 0, batch);
    apply_batch2(plan, batch);
    double max_error = 0.0;
    double scale = 1.0;
    alignas(16) double lane_re[2], lane_im[2];
    for (int lane = 0; lane < 2; ++lane) {
        pack_single(plan, input, lane, single);
        plan.apply_permuted(single);
        for (int i = 0; i < n; ++i) {
            _mm_store_pd(lane_re, batch.re[i]);
            _mm_store_pd(lane_im, batch.im[i]);
            const double error_re = std::abs(single.re[i] - lane_re[lane]);
            const double error_im = std::abs(single.im[i] - lane_im[lane]);
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
#if !defined(__SSE2__)
    (void)argc;
    (void)argv;
    std::cout << "SSE2 is required; batch2 benchmark was not built.\n";
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
    if (n < 1 || (n & (n - 1)) != 0 || frames < 2 || frames % 2 != 0 || repeats < 1) {
        std::cerr << "usage: fft_batch2_micro --n POW2 --frames MULTIPLE_OF_2 --repeats N\n";
        return 2;
    }

    FFTPlan plan(n);
    ComplexBuffer single(n);
    Batch2Buffer batch(n);
    std::vector<double> input(n);
    for (int i = 0; i < n; ++i)
        input[i] = 0.25 + 0.5 * std::sin(0.017 * i) + 0.1 * std::cos(0.071 * i);

    const double validation_error = validate_batch2(plan, input);
    std::cout << "validation scaled_max_error=" << std::setprecision(12)
              << validation_error << '\n';
    if (!(validation_error <= 1e-9)) {
        std::cerr << "batch2 numerical validation failed\n";
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
        for (int frame = 0; frame < frames; frame += 2) {
            auto begin = Clock::now();
            pack_batch2(plan, input, frame, batch);
            auto packed = Clock::now();
            apply_batch2(plan, batch);
            auto end = Clock::now();
            batch_pack += seconds(begin, packed);
            batch_fft += seconds(packed, end);
            for (int lane = 0; lane < 2; ++lane) {
                const int current = frame + lane;
                alignas(16) double re_lane[2], im_lane[2];
                _mm_store_pd(re_lane, batch.re[current % n]);
                _mm_store_pd(im_lane, batch.im[(current * 7) % n]);
                batch_checksum += re_lane[lane] + im_lane[lane];
            }
        }

        std::cout << "single," << n << ',' << rep << ',' << std::setprecision(12)
                  << single_pack * 1e9 / frames << ',' << single_fft * 1e9 / frames << ','
                  << (single_pack + single_fft) * 1e9 / frames << ','
                  << single_checksum << '\n';
        std::cout << "batch2," << n << ',' << rep << ',' << std::setprecision(12)
                  << batch_pack * 1e9 / frames << ',' << batch_fft * 1e9 / frames << ','
                  << (batch_pack + batch_fft) * 1e9 / frames << ','
                  << batch_checksum << '\n';
    }
    return 0;
#endif
}
