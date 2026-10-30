// 内部四帧 FFT microbenchmark；不作为正式 Melspec 程序链接。
#include "../src/melspec.cpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#if defined(__AVX2__) && defined(__FMA__)

struct Batch4Buffer {
    std::vector<__m256d> re, im;
    explicit Batch4Buffer(int n) : re(n), im(n) {}
};

static inline double batch4_input(const std::vector<double>& input, int frame, int i) {
    const int n = static_cast<int>(input.size());
    return input[(i + frame) % n] * (1.0 + 1e-9 * frame);
}

static void pack_single(const FFTPlan& plan, const std::vector<double>& input,
                        int frame, ComplexBuffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        const double value = batch4_input(input, frame, i);
        a.re[plan.reversed[i]] = value;
        a.im[plan.reversed[i]] = value * 0.125;
    }
}

static void pack_batch4(const FFTPlan& plan, const std::vector<double>& input,
                        int frame, Batch4Buffer& a) {
    const int n = static_cast<int>(input.size());
    for (int i = 0; i < n; ++i) {
        const double v0 = batch4_input(input, frame, i);
        const double v1 = batch4_input(input, frame + 1, i);
        const double v2 = batch4_input(input, frame + 2, i);
        const double v3 = batch4_input(input, frame + 3, i);
        const int p = plan.reversed[i];
        a.re[p] = _mm256_setr_pd(v0, v1, v2, v3);
        a.im[p] = _mm256_setr_pd(v0 * 0.125, v1 * 0.125,
                                 v2 * 0.125, v3 * 0.125);
    }
}

static void apply_batch4(const FFTPlan& plan, Batch4Buffer& a) {
    const int n = static_cast<int>(a.re.size());
    __m256d* re = a.re.data();
    __m256d* im = a.im.data();
    if (plan.initial_radix2) {
        for (int i = 0; i < n; i += 2) {
            const __m256d rr = re[i], ri = im[i];
            re[i] = _mm256_add_pd(rr, re[i + 1]);
            im[i] = _mm256_add_pd(ri, im[i + 1]);
            re[i + 1] = _mm256_sub_pd(rr, re[i + 1]);
            im[i + 1] = _mm256_sub_pd(ri, im[i + 1]);
        }
    }
    for (const auto& stage : plan.stages) {
        const int len = stage.len;
        const int quarter = len / 4;
        for (int base = 0; base < n; base += len) {
            if (quarter == 1) {
                const __m256d ar = re[base], ai = im[base];
                const __m256d br = re[base + 1], bi = im[base + 1];
                const __m256d cr = re[base + 2], ci = im[base + 2];
                const __m256d dr = re[base + 3], di = im[base + 3];
                const __m256d sum01r = _mm256_add_pd(ar, br);
                const __m256d sum01i = _mm256_add_pd(ai, bi);
                const __m256d dif01r = _mm256_sub_pd(ar, br);
                const __m256d dif01i = _mm256_sub_pd(ai, bi);
                const __m256d sum23r = _mm256_add_pd(cr, dr);
                const __m256d sum23i = _mm256_add_pd(ci, di);
                const __m256d dif23r = _mm256_sub_pd(cr, dr);
                const __m256d dif23i = _mm256_sub_pd(ci, di);
                re[base] = _mm256_add_pd(sum01r, sum23r);
                im[base] = _mm256_add_pd(sum01i, sum23i);
                re[base + 1] = _mm256_add_pd(dif01r, dif23i);
                im[base + 1] = _mm256_sub_pd(dif01i, dif23r);
                re[base + 2] = _mm256_sub_pd(sum01r, sum23r);
                im[base + 2] = _mm256_sub_pd(sum01i, sum23i);
                re[base + 3] = _mm256_sub_pd(dif01r, dif23i);
                im[base + 3] = _mm256_add_pd(dif01i, dif23r);
                continue;
            }
            for (int j = 0; j < quarter; ++j) {
                const int p = base + j;
                const __m256d ar = re[p], ai = im[p];
                const __m256d br = re[p + quarter], bi = im[p + quarter];
                const __m256d cr = re[p + 2 * quarter], ci = im[p + 2 * quarter];
                const __m256d dr = re[p + 3 * quarter], di = im[p + 3 * quarter];
                const __m256d w1r = _mm256_set1_pd(stage.w1re[j]);
                const __m256d w1i = _mm256_set1_pd(stage.w1im[j]);
                const __m256d w2r = _mm256_set1_pd(stage.w2re[j]);
                const __m256d w2i = _mm256_set1_pd(stage.w2im[j]);
                const __m256d w3r = _mm256_set1_pd(stage.w3re[j]);
                const __m256d w3i = _mm256_set1_pd(stage.w3im[j]);
                const __m256d br2 = _mm256_fmsub_pd(br, w2r, _mm256_mul_pd(bi, w2i));
                const __m256d bi2 = _mm256_fmadd_pd(br, w2i, _mm256_mul_pd(bi, w2r));
                const __m256d cr1 = _mm256_fmsub_pd(cr, w1r, _mm256_mul_pd(ci, w1i));
                const __m256d ci1 = _mm256_fmadd_pd(cr, w1i, _mm256_mul_pd(ci, w1r));
                const __m256d dr3 = _mm256_fmsub_pd(dr, w3r, _mm256_mul_pd(di, w3i));
                const __m256d di3 = _mm256_fmadd_pd(dr, w3i, _mm256_mul_pd(di, w3r));
                const __m256d sum01r = _mm256_add_pd(ar, br2);
                const __m256d sum01i = _mm256_add_pd(ai, bi2);
                const __m256d dif01r = _mm256_sub_pd(ar, br2);
                const __m256d dif01i = _mm256_sub_pd(ai, bi2);
                const __m256d sum23r = _mm256_add_pd(cr1, dr3);
                const __m256d sum23i = _mm256_add_pd(ci1, di3);
                const __m256d dif23r = _mm256_sub_pd(cr1, dr3);
                const __m256d dif23i = _mm256_sub_pd(ci1, di3);
                re[p] = _mm256_add_pd(sum01r, sum23r);
                im[p] = _mm256_add_pd(sum01i, sum23i);
                re[p + quarter] = _mm256_add_pd(dif01r, dif23i);
                im[p + quarter] = _mm256_sub_pd(dif01i, dif23r);
                re[p + 2 * quarter] = _mm256_sub_pd(sum01r, sum23r);
                im[p + 2 * quarter] = _mm256_sub_pd(sum01i, sum23i);
                re[p + 3 * quarter] = _mm256_sub_pd(dif01r, dif23i);
                im[p + 3 * quarter] = _mm256_add_pd(dif01i, dif23r);
            }
        }
    }
}

static double validate_batch4(const FFTPlan& plan, const std::vector<double>& input) {
    const int n = static_cast<int>(input.size());
    ComplexBuffer single(n);
    Batch4Buffer batch(n);
    pack_batch4(plan, input, 0, batch);
    apply_batch4(plan, batch);
    double max_error = 0.0;
    double scale = 1.0;
    alignas(32) double lane_re[4], lane_im[4];
    for (int lane = 0; lane < 4; ++lane) {
        pack_single(plan, input, lane, single);
        plan.apply_permuted(single);
        for (int i = 0; i < n; ++i) {
            _mm256_store_pd(lane_re, batch.re[i]);
            _mm256_store_pd(lane_im, batch.im[i]);
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
#if !defined(__AVX2__) || !defined(__FMA__)
    (void)argc;
    (void)argv;
    std::cout << "AVX2 and FMA are required; batch4 benchmark was not built.\n";
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
    if (n < 1 || (n & (n - 1)) != 0 || frames < 4 || frames % 4 != 0 || repeats < 1) {
        std::cerr << "usage: fft_batch4_micro --n POW2 --frames MULTIPLE_OF_4 --repeats N\n";
        return 2;
    }

    FFTPlan plan(n);
    ComplexBuffer single(n);
    Batch4Buffer batch(n);
    std::vector<double> input(n);
    for (int i = 0; i < n; ++i)
        input[i] = 0.25 + 0.5 * std::sin(0.017 * i) + 0.1 * std::cos(0.071 * i);

    const double validation_error = validate_batch4(plan, input);
    std::cout << "validation scaled_max_error=" << std::setprecision(12)
              << validation_error << '\n';
    if (!(validation_error <= 1e-9)) {
        std::cerr << "batch4 numerical validation failed\n";
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
        for (int frame = 0; frame < frames; frame += 4) {
            auto begin = Clock::now();
            pack_batch4(plan, input, frame, batch);
            auto packed = Clock::now();
            apply_batch4(plan, batch);
            auto end = Clock::now();
            batch_pack += seconds(begin, packed);
            batch_fft += seconds(packed, end);
            for (int lane = 0; lane < 4; ++lane) {
                const int current = frame + lane;
                alignas(32) double re_lane[4], im_lane[4];
                _mm256_store_pd(re_lane, batch.re[current % n]);
                _mm256_store_pd(im_lane, batch.im[(current * 7) % n]);
                batch_checksum += re_lane[lane] + im_lane[lane];
            }
        }

        std::cout << "single," << n << ',' << rep << ',' << std::setprecision(12)
                  << single_pack * 1e9 / frames << ',' << single_fft * 1e9 / frames << ','
                  << (single_pack + single_fft) * 1e9 / frames << ','
                  << single_checksum << '\n';
        std::cout << "batch4," << n << ',' << rep << ',' << std::setprecision(12)
                  << batch_pack * 1e9 / frames << ',' << batch_fft * 1e9 / frames << ','
                  << (batch_pack + batch_fft) * 1e9 / frames << ','
                  << batch_checksum << '\n';
    }
    return 0;
#endif
}
