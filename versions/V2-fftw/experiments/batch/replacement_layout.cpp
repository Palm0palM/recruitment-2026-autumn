#include "melspec.h"
#include "fftw_support.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#ifndef MELSPEC_FFTW_BATCH
#define MELSPEC_FFTW_BATCH 8
#endif

#if MELSPEC_FFTW_BATCH != 4 && MELSPEC_FFTW_BATCH != 8 && \
    MELSPEC_FFTW_BATCH != 16
#error "MELSPEC_FFTW_BATCH must be one of 4, 8, 16 for the layout experiment"
#endif

// 0: padded rows; 1: frame-interleaved SIMD-friendly planes.
#ifndef MELSPEC_BATCH_LAYOUT
#define MELSPEC_BATCH_LAYOUT 0
#endif

#if MELSPEC_BATCH_LAYOUT != 0 && MELSPEC_BATCH_LAYOUT != 1
#error "MELSPEC_BATCH_LAYOUT must be 0 (padded-row) or 1 (interleaved)"
#endif

namespace {
constexpr int kBatch = MELSPEC_FFTW_BATCH;
constexpr int kLayout = MELSPEC_BATCH_LAYOUT;

constexpr int output_distance(int n_bins) {
    return kLayout == 0 ? (n_bins + 3) & ~3 : 1;
}
}  // namespace

// HTK mel scale: keep the public helpers and double arithmetic unchanged.
double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

using melspec_fftw::Buffer;
using melspec_fftw::Plan;

void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    if (n < 1 || (n & (n - 1)) != 0)
        throw std::runtime_error("FFT length must be a positive power of two");

    Buffer<double> input(n);
    Buffer<fftw_complex> spectrum(n / 2 + 1);
    Plan plan(n, input.get(), spectrum.get());
    for (int i = 0; i < n; ++i) input[i] = frame[i];
    plan.apply(input.get(), spectrum.get());

    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k)
        out[k] = {spectrum[k][0], spectrum[k][1]};
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
        hz_points[i] =
            mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1));

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

namespace {

// A single serially-created new-array plan is shared by all outer workers.
// Layout 0: input[b*N+i], output[b*Kpad+k], with Kpad rounded up to
//           four complexes (64-byte distance between output rows).
// Layout 1: input[i*B+b], output[k*B+b], making b contiguous at each i/k.
class BatchPlan {
    fftw_plan plan_ = nullptr;

public:
    BatchPlan(int n_fft, int n_bins, double* input, fftw_complex* output) {
        std::lock_guard<std::mutex> lock(melspec_fftw::planner_mutex());
        FFTW_START(plan_start);
        const int n[1] = {n_fft};
        if constexpr (kLayout == 0) {
            plan_ = fftw_plan_many_dft_r2c(
                1, n, kBatch, input, nullptr, 1, n_fft,
                output, nullptr, 1, output_distance(n_bins),
                melspec_fftw::planner_flags());
        } else {
            plan_ = fftw_plan_many_dft_r2c(
                1, n, kBatch, input, nullptr, kBatch, 1,
                output, nullptr, kBatch, 1,
                melspec_fftw::planner_flags());
        }
        if (!plan_)
            throw std::runtime_error("FFTW layout plan-many R2C creation failed");
        FFTW_TIME("r2c_layout_many_plan_s", plan_start);
    }

    BatchPlan(const BatchPlan&) = delete;
    BatchPlan& operator=(const BatchPlan&) = delete;

    ~BatchPlan() {
        std::lock_guard<std::mutex> lock(melspec_fftw::planner_mutex());
        fftw_destroy_plan(plan_);
    }

    void apply(double* input, fftw_complex* output) const {
        fftw_execute_dft_r2c(plan_, input, output);
    }
};

}  // namespace

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    if (y.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        n_fft < 1 || (n_fft & (n_fft - 1)) != 0 || hop_length < 1 ||
        n_mels < 1 || sr < 1 ||
        y.size() < static_cast<std::size_t>(n_fft))
        throw std::runtime_error("Invalid parameters or audio too short for n_fft");

    const int n_samples = static_cast<int>(y.size());
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    const int n_bins = n_fft / 2 + 1;
    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    FFTW_START(setup_start);

    std::vector<double> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] =
            0.5 * (1.0 - std::cos(2.0 * M_PI * i / static_cast<double>(n_fft)));

    auto mel_fb = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    std::vector<int> first(n_mels), last(n_mels, n_bins);
    for (int m = 0; m < n_mels; ++m) {
        while (first[m] < n_bins && mel_fb[m][first[m]] == 0.0) ++first[m];
        while (last[m] > first[m] && mel_fb[m][last[m] - 1] == 0.0) --last[m];
    }

    output.resize(static_cast<std::size_t>(n_mels) * n_frames);

    const int n_threads = std::min({16,
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency())),
        std::max(1, n_frames / 256)});
    const int out_dist = output_distance(n_bins);
    const std::size_t fft_buffer_elements =
        kLayout == 0 ? static_cast<std::size_t>(kBatch) * out_dist
                     : static_cast<std::size_t>(kBatch) * n_bins;

    // All workers use the same sizes, fftw_malloc alignment, placement and
    // strides. Scratch allocation and planning complete before thread launch.
    std::vector<Buffer<double>> input_buffers;
    std::vector<Buffer<fftw_complex>> fft_buffers;
    input_buffers.reserve(n_threads);
    fft_buffers.reserve(n_threads);
    for (int id = 0; id < n_threads; ++id) {
        input_buffers.emplace_back(static_cast<std::size_t>(kBatch) * n_fft);
        fft_buffers.emplace_back(fft_buffer_elements);
    }
    std::vector<std::vector<double>> power_buffers(
        n_threads,
        std::vector<double>(static_cast<std::size_t>(kBatch) * n_bins));

    std::unique_ptr<BatchPlan> plan;
    if (n_fft > 1)
        plan = std::make_unique<BatchPlan>(n_fft, n_bins,
                                           input_buffers[0].get(),
                                           fft_buffers[0].get());
    FFTW_TIME("compute_setup_s", setup_start);

    auto worker = [&](int id) {
        FFTW_COUNTERS(counters);
        auto& input = input_buffers[id];
        auto& fft_out = fft_buffers[id];
        auto& power = power_buffers[id];

        const int begin = static_cast<long long>(n_frames) * id / n_threads;
        const int end = static_cast<long long>(n_frames) * (id + 1) / n_threads;

        // Increment by valid, not B: the final iteration reaches end without
        // overflowing even when n_frames is close to INT_MAX.
        for (int t0 = begin; t0 < end;) {
            const int valid = std::min(kBatch, end - t0);

            if (n_fft == 1) {
                // Periodic Hann's only coefficient is exactly zero. K=1 makes
                // power[b] the correct entry in either power layout.
                for (int b = 0; b < valid; ++b) {
                    const std::size_t start =
                        static_cast<std::size_t>(t0 + b) * hop_length;
                    const double sample = y[start] * window[0];
                    power[b] = sample * sample;
                }
            } else {
                FFTW_START(pack_start);
                if constexpr (kLayout == 0) {
                    for (int b = 0; b < kBatch; ++b) {
                        double* dst = input.get() +
                                      static_cast<std::size_t>(b) * n_fft;
                        if (b < valid) {
                            const std::size_t start =
                                static_cast<std::size_t>(t0 + b) * hop_length;
                            for (int i = 0; i < n_fft; ++i)
                                dst[i] = y[start + i] * window[i];
                        } else {
                            std::fill(dst, dst + n_fft, 0.0);
                        }
                    }
                } else {
                    // Contiguous b writes at each sample index. Unused frames
                    // are explicitly zeroed for the full shared-plan execute.
                    for (int i = 0; i < n_fft; ++i) {
                        double* dst = input.get() +
                                      static_cast<std::size_t>(i) * kBatch;
                        for (int b = 0; b < valid; ++b) {
                            const std::size_t start =
                                static_cast<std::size_t>(t0 + b) * hop_length;
                            dst[b] = y[start + i] * window[i];
                        }
                        for (int b = valid; b < kBatch; ++b) dst[b] = 0.0;
                    }
                }
                FFTW_ADD(counters, 0, pack_start);

                FFTW_START(fft_start);
                plan->apply(input.get(), fft_out.get());
                FFTW_ADD(counters, 1, fft_start);

                FFTW_START(power_start);
                if constexpr (kLayout == 0) {
                    // Padded output rows are contiguous per frame. Padding is
                    // never read; power uses unpadded per-frame rows.
                    for (int b = 0; b < valid; ++b) {
                        const fftw_complex* spectrum = fft_out.get() +
                            static_cast<std::size_t>(b) * out_dist;
                        double* power_frame = power.data() +
                            static_cast<std::size_t>(b) * n_bins;
                        for (int k = 0; k < n_bins; ++k)
                            power_frame[k] = spectrum[k][0] * spectrum[k][0] +
                                             spectrum[k][1] * spectrum[k][1];
                    }
                } else {
                    // Both spectrum and power are contiguous in b. This loop
                    // can SIMD across independent frames without changing any
                    // frame's arithmetic or the subsequent Mel reduction order.
                    for (int k = 0; k < n_bins; ++k) {
                        const fftw_complex* spectrum = fft_out.get() +
                            static_cast<std::size_t>(k) * kBatch;
                        double* power_bin = power.data() +
                            static_cast<std::size_t>(k) * kBatch;
                        for (int b = 0; b < valid; ++b)
                            power_bin[b] = spectrum[b][0] * spectrum[b][0] +
                                           spectrum[b][1] * spectrum[b][1];
                    }
                }
                FFTW_ADD(counters, 2, power_start);
            }

            FFTW_START(mel_start);
            for (int b = 0; b < valid; ++b) {
                const int t = t0 + b;
                for (int m = 0; m < n_mels; ++m) {
                    double acc = 0.0;
                    if constexpr (kLayout == 0) {
                        const double* power_frame = power.data() +
                            static_cast<std::size_t>(b) * n_bins;
                        for (int k = first[m]; k < last[m]; ++k)
                            acc += mel_fb[m][k] * power_frame[k];
                    } else {
                        // Strided power reads are the layout tradeoff. Keep k
                        // increasing and one scalar double accumulator per frame.
                        for (int k = first[m]; k < last[m]; ++k)
                            acc += mel_fb[m][k] *
                                   power[static_cast<std::size_t>(k) * kBatch + b];
                    }
                    output[static_cast<std::size_t>(m) * n_frames + t] = acc;
                }
            }
            FFTW_ADD(counters, 3, mel_start);
            t0 += valid;
        }
        FFTW_PRINT(counters, id);
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    int launched = 1;
    try {
        for (; launched < n_threads; ++launched)
            threads.emplace_back(worker, launched);
    } catch (const std::system_error&) {
    }
    worker(0);
    for (int id = launched; id < n_threads; ++id) worker(id);
    for (auto& thread : threads) thread.join();
}
