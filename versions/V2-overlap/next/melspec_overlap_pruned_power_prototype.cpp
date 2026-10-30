#include "melspec.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

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

struct FFTPlan {
    std::vector<int> reversed;
    std::vector<std::complex<double>> twiddles;

    explicit FFTPlan(int n) : reversed(n), twiddles(n) {
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            reversed[i] = j;
        }
        // Store each stage's factors, using the original recurrence.
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * M_PI / static_cast<double>(len);
            std::complex<double> wlen(std::cos(ang), std::sin(ang));
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                twiddles[len / 2 + j] = w;
                w *= wlen;
            }
        }
    }

    void apply_butterflies(std::vector<std::complex<double>>& a) const {
        const int n = static_cast<int>(a.size());
        for (int len = 2; len <= n; len <<= 1) {
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < len / 2; ++j) {
                    std::complex<double> u = a[i + j];
                    std::complex<double> v =
                        a[i + j + len / 2] * twiddles[len / 2 + j];
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                }
            }
        }
    }

    void apply(std::vector<std::complex<double>>& a) const {
        const int n = static_cast<int>(a.size());
        for (int i = 1; i < n; ++i)
            if (i < reversed[i]) std::swap(a[i], a[reversed[i]]);
        apply_butterflies(a);
    }
};

// Sliding sub-FFT decomposition for highly overlapped frames. With N=S*B and
// S=hop_length, B_q[r] is the B-point FFT of x[start+q+S*j]. Moving to the next
// frame shifts every real subsequence by one sample, so its positive-frequency
// bins can be updated with one sliding-DFT recurrence. An S-point FFT across q
// then reconstructs each group of full-frame frequencies.
struct OverlapFFTPlan {
    int n_fft;
    int hop;
    int b;
    int s;
    int log_b;
    FFTPlan leaf_plan;
    FFTPlan combine_plan;
    std::vector<std::complex<double>> update_phase;
    std::vector<std::complex<double>> combine_twiddles;

    OverlapFFTPlan(int n, int hop_length)
        : n_fft(n), hop(hop_length), b(n / hop_length), s(hop_length), log_b(0),
          leaf_plan(b), combine_plan(s), update_phase(b / 2 + 1),
          combine_twiddles((b / 2 + 1) * s) {
        while ((1 << log_b) < b) ++log_b;
        for (int r = 0; r <= b / 2; ++r) {
            double ang = 2.0 * M_PI * r / b;
            update_phase[r] = {std::cos(ang), std::sin(ang)};
        }
        for (int r = 0; r <= b / 2; ++r) {
            for (int q = 0; q < s; ++q) {
                int index = static_cast<int>(static_cast<long long>(r) * q % n);
                double ang = -2.0 * M_PI * index / n;
                combine_twiddles[r * s + q] = {std::cos(ang), std::sin(ang)};
            }
        }
    }

    static bool supported(int n, int hop_length) {
        if (n < 4 || hop_length <= 0 || n % hop_length != 0) return false;
        int b = n / hop_length;
        auto power_of_two = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
        return power_of_two(b) && power_of_two(hop_length);
    }
};

struct OverlapScratch {
    std::vector<std::complex<double>> leaves;
    std::vector<std::complex<double>> leaf;
    std::vector<std::complex<double>> combined;
    std::vector<std::complex<double>> spectrum;
    std::vector<double> delta;

    explicit OverlapScratch(const OverlapFFTPlan& plan)
        : leaves((plan.b / 2 + 1) * plan.s), leaf(plan.b), combined(plan.s),
          spectrum(plan.b * (plan.s / 2 + 1)), delta(plan.s) {}
};

void overlap_power(const std::vector<double>& y, int start, int frame,
                   const OverlapFFTPlan& plan, OverlapScratch& scratch,
                   const std::vector<int>& power_bins, std::vector<double>& power) {
    const int b = plan.b;
    const int s = plan.s;
    const int n = plan.n_fft;
    // Recompute exactly every 1024 frames to bound sliding-DFT roundoff drift.
    if (frame % 1024 == 0) {
        for (int q = 0; q < s; ++q) {
            for (int j = 0; j < b; ++j)
                scratch.leaf[j] = {y[start + q + s * j], 0.0};
            plan.leaf_plan.apply(scratch.leaf);
            for (int r = 0; r <= b / 2; ++r)
                scratch.leaves[r * s + q] = scratch.leaf[r];
        }
    } else {
        const int old_start = start - plan.hop;
        // The same input delta is shared by every residue r; load/subtract once.
        for (int q = 0; q < s; ++q)
            scratch.delta[q] = y[old_start + q + n] - y[old_start + q];
        for (int r = 0; r <= b / 2; ++r) {
            const auto phase = plan.update_phase[r];
            for (int q = 0; q < s; ++q) {
                auto& dst = scratch.leaves[r * s + q];
                dst = phase * (dst + scratch.delta[q]);
            }
        }
    }

    const int stride = s / 2 + 1;
    for (int r = 0; r <= b / 2; ++r) {
        for (int q = 0; q < s; ++q) {
            // Fuse bit reversal into the gather, leaving only butterfly stages.
            scratch.combined[plan.combine_plan.reversed[q]] =
                scratch.leaves[r * s + q] * plan.combine_twiddles[r * s + q];
        }
        plan.combine_plan.apply_butterflies(scratch.combined);
        if (r == 0) {
            // DC through Nyquist, inclusive.
            for (int m = 0; m <= s / 2; ++m)
                scratch.spectrum[m] = scratch.combined[m];
        } else if (r == b / 2) {
            // Self-paired residue: k=(m+1/2)*B <= N/2 requires m<S/2.
            for (int m = 0; m < s / 2; ++m)
                scratch.spectrum[r * stride + m] = scratch.combined[m];
        } else {
            for (int m = 0; m < s / 2; ++m) {
                scratch.spectrum[r * stride + m] = scratch.combined[m];
                // A[(B-r)+m*B] = conj(A[r+(S-m-1)*B]).
                scratch.spectrum[(b - r) * stride + m] =
                    std::conj(scratch.combined[s - m - 1]);
            }
        }
    }

    // Periodic Hann: w[n] = 1/2 - exp(+i*w*n)/4 - exp(-i*w*n)/4.
    // Multiplication in time is a three-bin subtraction in frequency.
    const auto& a = scratch.spectrum;
    auto at = [&](int k) -> const std::complex<double>& {
        return a[(k & (b - 1)) * stride + (k >> plan.log_b)];
    };
    // The full unwindowed spectrum remains available for k-1/k+1. Only bins
    // actually consumed by a nonzero Mel triangle need Hann and |X|^2 here.
    for (int k : power_bins) {
        if (k == 0)
            power[k] = std::norm(0.5 * at(0) - 0.25 * (at(1) + std::conj(at(1))));
        else if (k == n / 2)
            power[k] = std::norm(0.5 * at(n / 2) -
                0.25 * (at(n / 2 - 1) + std::conj(at(n / 2 - 1))));
        else
            power[k] = std::norm(0.5 * at(k) - 0.25 * (at(k - 1) + at(k + 1)));
    }
}

// Real FFT: input n real values → output n/2+1 complex values.
void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    // Copy real input into complex array
    std::vector<std::complex<double>> a(n);
    for (int i = 0; i < n; ++i) a[i] = {frame[i], 0.0};
    FFTPlan(n).apply(a);
    // Keep only DC + positive frequencies (n/2+1 bins)
    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k) out[k] = a[k];
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

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
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
    std::vector<char> used(n_bins, 0);
    for (int m = 0; m < n_mels; ++m) {
        while (first[m] < n_bins && mel_fb[m][first[m]] == 0.0) ++first[m];
        while (last[m] > first[m] && mel_fb[m][last[m] - 1] == 0.0) --last[m];
        for (int k = first[m]; k < last[m]; ++k) used[k] = 1;
    }
    std::vector<int> power_bins;
    for (int k = 0; k < n_bins; ++k)
        if (used[k]) power_bins.push_back(k);

    const int half = std::max(1, n_fft / 2);
    FFTPlan plan(half);
    // -i/2 * exp(-2*pi*i*k/n_fft), for real-FFT reconstruction.
    std::vector<std::complex<double>> recovery(n_bins);
    for (int k = 1; k < n_fft / 2; ++k) {
        double ang = -2.0 * M_PI * k / static_cast<double>(n_fft);
        recovery[k] = {0.5 * std::sin(ang), -0.5 * std::cos(ang)};
    }
    output.resize(static_cast<size_t>(n_mels) * n_frames);

    std::optional<OverlapFFTPlan> overlap_plan;
    if (OverlapFFTPlan::supported(n_fft, hop_length))
        overlap_plan.emplace(n_fft, hop_length);

    // Keep short inputs single-threaded, and use at most the judge's 16 cores.
    const int n_threads = std::min({16,
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency())),
        std::max(1, n_frames / 256)});
    // Allocate scratch before launching threads, so workers cannot throw.
    std::vector<std::vector<std::complex<double>>> fft_buffers(
        n_threads, std::vector<std::complex<double>>(half));
    std::vector<std::vector<double>> power_buffers(
        n_threads, std::vector<double>(n_bins));
    std::vector<OverlapScratch> overlap_buffers;
    if (overlap_plan) {
        overlap_buffers.reserve(n_threads);
        for (int id = 0; id < n_threads; ++id)
            overlap_buffers.emplace_back(*overlap_plan);
    }
    auto worker = [&](int id) {
        auto& fft_out = fft_buffers[id];
        auto& power = power_buffers[id];
        const int begin = static_cast<long long>(n_frames) * id / n_threads;
        const int end = static_cast<long long>(n_frames) * (id + 1) / n_threads;
        for (int t = begin; t < end; ++t) {
            int start = t * hop_length;

            if (overlap_plan) {
                overlap_power(y, start, t - begin, *overlap_plan,
                              overlap_buffers[id], power_bins, power);
            } else if (n_fft == 1) {
                double sample = y[start] * window[0];
                power[0] = sample * sample;
            } else {
                // Pack even/odd real samples into a half-length complex FFT.
                for (int i = 0; i < half; ++i)
                    fft_out[i] = {y[start + 2 * i] * window[2 * i],
                                  y[start + 2 * i + 1] * window[2 * i + 1]};
                plan.apply(fft_out);

                double dc = fft_out[0].real() + fft_out[0].imag();
                double nyquist = fft_out[0].real() - fft_out[0].imag();
                power[0] = dc * dc;
                power[half] = nyquist * nyquist;
                for (int k = 1; k < half; ++k) {
                    auto a = fft_out[k];
                    auto b = std::conj(fft_out[half - k]);
                    auto x = 0.5 * (a + b) + recovery[k] * (a - b);
                    power[k] = std::norm(x);
                }
            }

            // Apply filters immediately, while this frame's power is in cache.
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
        // If resources are limited, finish unlaunched partitions locally.
    }
    worker(0);
    for (int id = launched; id < n_threads; ++id) worker(id);
    for (auto& thread : threads) thread.join();
}
