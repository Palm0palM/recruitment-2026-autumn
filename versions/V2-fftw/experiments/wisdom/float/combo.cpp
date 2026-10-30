// Independent fftwf R2C precision experiment with optional private wisdom.
// Link with the protected src/main.cpp, but NOT with src/melspec.cpp.

#include "melspec.h"
#include "float_wisdom.h"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

// 1: convert each float FFT component to double, then compute power in double.
// 0: compute power in float, then convert it to double for the Mel reduction.
#ifndef MELSPEC_FFTWF_DOUBLE_POWER
#define MELSPEC_FFTWF_DOUBLE_POWER 1
#endif

#if MELSPEC_FFTWF_DOUBLE_POWER != 0 && MELSPEC_FFTWF_DOUBLE_POWER != 1
#error "MELSPEC_FFTWF_DOUBLE_POWER must be 0 or 1"
#endif

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct FftwfFree {
    void operator()(void* p) const { fftwf_free(p); }
};

template <class T>
class FftwfBuffer {
    std::unique_ptr<T, FftwfFree> data_;

public:
    explicit FftwfBuffer(std::size_t n)
        : data_(static_cast<T*>(fftwf_malloc(n * sizeof(T)))) {
        if (!data_) throw std::bad_alloc();
    }

    T* get() const { return data_.get(); }
    T& operator[](std::size_t i) const { return get()[i]; }
};

std::mutex& fftwf_planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

class FftwfR2CPlan {
    fftwf_plan plan_;

public:
    FftwfR2CPlan(int n, float* input, fftwf_complex* output) {
        // The only planner/destructor calls are outside the worker threads.
        // This mutex also makes concurrent API invocations safe with respect
        // to this translation unit's use of the global FFTW planner.
        std::lock_guard<std::mutex> lock(fftwf_planner_mutex());
        plan_ = melspec_fftwf::plan_r2c_with_float_wisdom(n, input, output);
        if (!plan_) throw std::runtime_error("FFTW float R2C plan creation failed");
    }

    FftwfR2CPlan(const FftwfR2CPlan&) = delete;
    FftwfR2CPlan& operator=(const FftwfR2CPlan&) = delete;

    ~FftwfR2CPlan() {
        std::lock_guard<std::mutex> lock(fftwf_planner_mutex());
        fftwf_destroy_plan(plan_);
    }

    void apply(float* input, fftwf_complex* output) const {
        // Shared read-only plan; every worker supplies its own aligned arrays.
        fftwf_execute_dft_r2c(plan_, input, output);
    }
};

inline double power_from_float_fft(const fftwf_complex& z) {
    const float re = z[0];
    const float im = z[1];
#if MELSPEC_FFTWF_DOUBLE_POWER
    const double re_d = static_cast<double>(re);
    const double im_d = static_cast<double>(im);
    return re_d * re_d + im_d * im_d;
#else
    // The named float result is intentional: do not promote the operands.
    const float p = re * re + im * im;
    return static_cast<double>(p);
#endif
}

inline double power_from_float_sample(float sample) {
#if MELSPEC_FFTWF_DOUBLE_POWER
    const double sample_d = static_cast<double>(sample);
    return sample_d * sample_d;
#else
    const float p = sample * sample;
    return static_cast<double>(p);
#endif
}

}  // namespace

double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

std::vector<std::vector<double>> build_mel_filterbank(int sr, int n_fft,
                                                       int n_mels, double f_min,
                                                       double f_max) {
    const int n_bins = n_fft / 2 + 1;

    std::vector<double> fft_freqs(n_bins);
    for (int k = 0; k < n_bins; ++k)
        fft_freqs[k] = static_cast<double>(k) * sr / static_cast<double>(n_fft);

    const double mel_min = hz_to_mel(f_min);
    const double mel_max = hz_to_mel(f_max);
    std::vector<double> hz_points(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; ++i)
        hz_points[i] = mel_to_hz(
            mel_min + i * (mel_max - mel_min) / (n_mels + 1));

    std::vector<std::vector<double>> weights(
        n_mels, std::vector<double>(n_bins, 0.0));
    for (int m = 0; m < n_mels; ++m) {
        const double f_l = hz_points[m];
        const double f_c = hz_points[m + 1];
        const double f_r = hz_points[m + 2];
        const double enorm = 2.0 / (f_r - f_l);
        for (int k = 0; k < n_bins; ++k) {
            const double rising = (fft_freqs[k] - f_l) / (f_c - f_l);
            const double falling = (f_r - fft_freqs[k]) / (f_r - f_c);
            const double val = std::max(0.0, std::min(rising, falling));
            weights[m][k] = val * enorm;
        }
    }
    return weights;
}

void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    if (n < 1 || (n & (n - 1)) != 0)
        throw std::runtime_error("FFT length must be a positive power of two");

    FftwfBuffer<float> input(n);
    FftwfBuffer<fftwf_complex> spectrum(n / 2 + 1);
    FftwfR2CPlan plan(n, input.get(), spectrum.get());
    // Initialize after planning, even when imported wisdom preserves input.
    for (int i = 0; i < n; ++i)
        input[i] = static_cast<float>(frame[i]);
    plan.apply(input.get(), spectrum.get());

    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k)
        out[k] = {static_cast<double>(spectrum[k][0]),
                  static_cast<double>(spectrum[k][1])};
}

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    const int n_samples = static_cast<int>(y.size());
    // Check before division so a short input cannot be rounded to one frame.
    if (n_fft < 1 || (n_fft & (n_fft - 1)) != 0 || hop_length < 1 ||
        n_mels < 1 || sr < 1 || n_samples < n_fft)
        throw std::runtime_error("Invalid parameters or audio too short for n_fft");

    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    const int n_bins = n_fft / 2 + 1;
    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    // Keep window construction and the input product in double. Only the
    // product is rounded to float for fftwf.
    std::vector<double> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] = 0.5 *
            (1.0 - std::cos(2.0 * kPi * i / static_cast<double>(n_fft)));

    const auto mel_fb = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    std::vector<int> first(n_mels), last(n_mels, n_bins);
    for (int m = 0; m < n_mels; ++m) {
        while (first[m] < n_bins && mel_fb[m][first[m]] == 0.0) ++first[m];
        while (last[m] > first[m] && mel_fb[m][last[m] - 1] == 0.0) --last[m];
    }

    output.resize(static_cast<std::size_t>(n_mels) * n_frames);

    const int n_threads = std::min({16,
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency())),
        std::max(1, n_frames / 256)});

    // All allocations precede thread launch. No worker allocates or plans.
    std::vector<FftwfBuffer<float>> input_buffers;
    std::vector<FftwfBuffer<fftwf_complex>> fft_buffers;
    input_buffers.reserve(n_threads);
    fft_buffers.reserve(n_threads);
    for (int id = 0; id < n_threads; ++id) {
        input_buffers.emplace_back(n_fft);
        fft_buffers.emplace_back(n_bins);
    }
    FftwfR2CPlan plan(n_fft, input_buffers[0].get(), fft_buffers[0].get());
    std::vector<std::vector<double>> power_buffers(
        n_threads, std::vector<double>(n_bins));

    const auto worker = [&](int id) {
        auto& input = input_buffers[id];
        auto& fft_out = fft_buffers[id];
        auto& power = power_buffers[id];
        const int begin = static_cast<long long>(n_frames) * id / n_threads;
        const int end = static_cast<long long>(n_frames) * (id + 1) / n_threads;

        for (int t = begin; t < end; ++t) {
            const int start = t * hop_length;
            if (n_fft == 1) {
                // The periodic one-point Hann coefficient is zero. Preserve
                // that A2/V1 boundary behavior rather than special-casing a
                // one-point window to one.
                const float sample =
                    static_cast<float>(y[start] * window[0]);
                power[0] = power_from_float_sample(sample);
            } else {
                for (int i = 0; i < n_fft; ++i)
                    input[i] = static_cast<float>(y[start + i] * window[i]);
                plan.apply(input.get(), fft_out.get());
                for (int k = 0; k < n_bins; ++k)
                    power[k] = power_from_float_fft(fft_out[k]);
            }

            // Both precision variants use double weights, double power
            // storage, and a double accumulator in the same bin order.
            for (int m = 0; m < n_mels; ++m) {
                double acc = 0.0;
                for (int k = first[m]; k < last[m]; ++k)
                    acc += mel_fb[m][k] * power[k];
                output[static_cast<size_t>(m) * n_frames + t] = acc;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    int launched = 1;
    try {
        for (; launched < n_threads; ++launched)
            threads.emplace_back(worker, launched);
    } catch (const std::system_error&) {
        // Finish every unlaunched partition locally if resources are limited.
    }
    worker(0);
    for (int id = launched; id < n_threads; ++id) worker(id);
    for (auto& thread : threads) thread.join();
}
