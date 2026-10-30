// Library-first C++ reference: batched FFTW + OpenBLAS dense / Eigen sparse.
// This is a standalone CLI; protected src/main.cpp is not modified or reused.
#include <fftw3.h>
#include <cblas.h>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/resource.h>
#include <string>
#include <thread>
#include <vector>

#if defined(__has_include)
#if __has_include(<openblas_config.h>)
#include <openblas_config.h>
#define HAVE_OPENBLAS_CONFIG 1
#endif
#endif

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

struct Args {
    std::string input, output;
    int sr = 44100, n_fft = 2048, hop_length = 512, n_mels = 128;
    double f_min = 0.0, f_max = 22050.0;
    int block_size = 64, threads = 16, blas_threads = 1;
    std::string mode = "sparse";  // sparse | dense
};

Args parse_args(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("usage: library_melspec <input> <output> [options]");
    Args a;
    a.input = argv[1];
    a.output = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        std::string val = argv[++i];
        if (key == "--sr") a.sr = std::stoi(val);
        else if (key == "--n_fft") a.n_fft = std::stoi(val);
        else if (key == "--hop_length") a.hop_length = std::stoi(val);
        else if (key == "--n_mels") a.n_mels = std::stoi(val);
        else if (key == "--f_min") a.f_min = std::stod(val);
        else if (key == "--f_max") a.f_max = std::stod(val);
        else if (key == "--block-size") a.block_size = std::stoi(val);
        else if (key == "--threads") a.threads = std::stoi(val);
        else if (key == "--blas-threads") a.blas_threads = std::stoi(val);
        else if (key == "--mode") a.mode = val;
        else throw std::runtime_error("unknown argument: " + key);
    }
    if (a.mode != "sparse" && a.mode != "dense")
        throw std::runtime_error("mode must be sparse or dense");
    if (a.block_size < 1 || a.threads < 1 || a.blas_threads < 1)
        throw std::runtime_error("block size and thread counts must be positive");
    if (a.n_fft < 1 || (a.n_fft & (a.n_fft - 1)) || a.hop_length < 1 ||
        a.n_mels < 1 || a.sr < 1)
        throw std::runtime_error("invalid parameters");
    return a;
}

std::vector<double> read_pcm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input: " + path);
    f.seekg(0, std::ios::end);
    auto bytes = f.tellg();
    if (bytes < 0 || bytes % 2) throw std::runtime_error("invalid PCM byte length");
    f.seekg(0, std::ios::beg);
    std::vector<int16_t> pcm(static_cast<size_t>(bytes) / 2);
    if (!f.read(reinterpret_cast<char*>(pcm.data()), bytes))
        throw std::runtime_error("failed to read PCM");
    std::vector<double> y(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) y[i] = static_cast<double>(pcm[i]) / 32768.0;
    return y;
}

void write_output(const std::string& path, const std::vector<float>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output: " + path);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("failed to write output");
}

double hz_to_mel(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
double mel_to_hz(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

struct Filterbank {
    int n_mels, n_bins;
    std::vector<float> dense;  // row-major n_mels x n_bins
    Eigen::SparseMatrix<float, Eigen::RowMajor> sparse;
};

Filterbank build_filterbank(int sr, int n_fft, int n_mels, double f_min, double f_max) {
    const int n_bins = n_fft / 2 + 1;
    Filterbank fb{n_mels, n_bins, std::vector<float>(static_cast<size_t>(n_mels) * n_bins),
                  Eigen::SparseMatrix<float, Eigen::RowMajor>(n_mels, n_bins)};
    std::vector<double> freqs(n_bins), hz(n_mels + 2);
    for (int k = 0; k < n_bins; ++k) freqs[k] = static_cast<double>(k) * sr / n_fft;
    const double mel_min = hz_to_mel(f_min), mel_max = hz_to_mel(f_max);
    for (int i = 0; i <= n_mels + 1; ++i)
        hz[i] = mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1));
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(static_cast<size_t>(n_mels) * 32);
    for (int m = 0; m < n_mels; ++m) {
        const double f_l = hz[m], f_c = hz[m + 1], f_r = hz[m + 2];
        const double enorm = 2.0 / (f_r - f_l);
        for (int k = 0; k < n_bins; ++k) {
            double rising = (freqs[k] - f_l) / (f_c - f_l);
            double falling = (f_r - freqs[k]) / (f_r - f_c);
            float value = static_cast<float>(std::max(0.0, std::min(rising, falling)) * enorm);
            fb.dense[static_cast<size_t>(m) * n_bins + k] = value;
            if (value != 0.0f) triplets.emplace_back(m, k, value);
        }
    }
    fb.sparse.setFromTriplets(triplets.begin(), triplets.end());
    fb.sparse.makeCompressed();
    return fb;
}

struct Plan {
    fftwf_plan plan = nullptr;
    Plan(int n_fft, int block, float* in, fftwf_complex* out) {
        int n[1] = {n_fft};
        plan = fftwf_plan_many_dft_r2c(1, n, block, in, nullptr, 1, n_fft,
                                       out, nullptr, 1, n_fft / 2 + 1, FFTW_ESTIMATE);
        if (!plan) throw std::runtime_error("FFTWf batched R2C plan failed");
    }
    Plan(int n_fft, float* in, fftwf_complex* out) {
        plan = fftwf_plan_dft_r2c_1d(n_fft, in, out, FFTW_ESTIMATE);
        if (!plan) throw std::runtime_error("FFTWf R2C plan failed");
    }
    ~Plan() { fftwf_destroy_plan(plan); }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
};

struct Scratch {
    float* input = nullptr;
    fftwf_complex* spectrum = nullptr;
    std::vector<float> power, block_out;
    Scratch(int block, int n_fft, int n_bins, int n_mels)
        : power(static_cast<size_t>(block) * n_bins),
          block_out(static_cast<size_t>(block) * n_mels) {
        input = static_cast<float*>(fftwf_malloc(sizeof(float) * block * n_fft));
        spectrum = static_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * block * n_bins));
        if (!input || !spectrum) throw std::bad_alloc();
    }
    ~Scratch() { fftwf_free(input); fftwf_free(spectrum); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto total_start = Clock::now();
        Args args = parse_args(argc, argv);
#ifdef HAVE_OPENBLAS_CONFIG
        openblas_set_num_threads(args.blas_threads);
#endif
        auto t = Clock::now();
        std::vector<double> y = read_pcm(args.input);
        const double read_s = elapsed(t);
        if (static_cast<int>(y.size()) < args.n_fft)
            throw std::runtime_error("audio too short for n_fft");
        const int n_frames = 1 + (static_cast<int>(y.size()) - args.n_fft) / args.hop_length;
        const int n_bins = args.n_fft / 2 + 1;
        const int block = args.block_size;
        const int n_blocks = (n_frames + block - 1) / block;

        t = Clock::now();
        std::vector<float> window(args.n_fft);
        for (int i = 0; i < args.n_fft; ++i)
            window[i] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / args.n_fft)));
        Filterbank fb = build_filterbank(args.sr, args.n_fft, args.n_mels, args.f_min, args.f_max);
        const int n_threads = std::min({args.threads, n_blocks, 32});
        std::vector<std::unique_ptr<Scratch>> scratch;
        scratch.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            scratch.emplace_back(new Scratch(block, args.n_fft, n_bins, args.n_mels));
        }
        // A single shared plan is required for good FFTWf new-array scaling;
        // one plan per worker degrades badly on this platform. For block=1 use
        // the basic 1-D planner rather than plan_many(howmany=1).
        std::unique_ptr<Plan> plan;
        if (block == 1) {
            plan.reset(new Plan(args.n_fft, scratch[0]->input, scratch[0]->spectrum));
        } else {
            plan.reset(new Plan(args.n_fft, block, scratch[0]->input, scratch[0]->spectrum));
        }
        const double setup_s = elapsed(t);

        std::vector<float> output(static_cast<size_t>(args.n_mels) * n_frames);
        std::vector<double> worker_seconds(n_threads, 0.0);
#ifdef MELSPEC_LIBRARY_PROFILE
        std::vector<std::array<double, 5>> phase_seconds(n_threads);
#endif
        auto worker = [&](int tid) {
            Scratch& s = *scratch[tid];

            double work = 0.0;
#ifdef MELSPEC_LIBRARY_PROFILE
            auto& phases = phase_seconds[tid];
            phases.fill(0.0);
            auto mark = Clock::now();
#endif
            const int block_begin = static_cast<long long>(n_blocks) * tid / n_threads;
            const int block_end = static_cast<long long>(n_blocks) * (tid + 1) / n_threads;
            for (int bidx = block_begin; bidx < block_end; ++bidx) {
                const auto block_start = Clock::now();
#ifdef MELSPEC_LIBRARY_PROFILE
                mark = Clock::now();
#endif
                const int start = bidx * block;
                const int count = std::min(block, n_frames - start);
                for (int b = 0; b < block; ++b) {
                    float* dst = s.input + static_cast<size_t>(b) * args.n_fft;
                    if (b < count) {
                        const int frame_start = (start + b) * args.hop_length;
                        for (int i = 0; i < args.n_fft; ++i)
                            dst[i] = static_cast<float>(y[frame_start + i]) * window[i];
                    } else {
                        std::memset(dst, 0, sizeof(float) * args.n_fft);
                    }
                }
#ifdef MELSPEC_LIBRARY_PROFILE
                phases[0] += elapsed(mark); mark = Clock::now();
#endif
                fftwf_execute_dft_r2c(plan->plan, s.input, s.spectrum);
#ifdef MELSPEC_LIBRARY_PROFILE
                phases[1] += elapsed(mark); mark = Clock::now();
#endif

                for (int b = 0; b < block; ++b) {
                    const fftwf_complex* src = s.spectrum + static_cast<size_t>(b) * n_bins;
                    float* dst = s.power.data() + static_cast<size_t>(b) * n_bins;
                    for (int k = 0; k < n_bins; ++k)
                        dst[k] = src[k][0] * src[k][0] + src[k][1] * src[k][1];
                }

#ifdef MELSPEC_LIBRARY_PROFILE
                phases[2] += elapsed(mark); mark = Clock::now();
#endif
                if (args.mode == "dense") {
                    // power is block x bins; CblasTrans forms bins x block without a copy.
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                args.n_mels, block, n_bins, 1.0f,
                                fb.dense.data(), n_bins, s.power.data(), n_bins,
                                0.0f, s.block_out.data(), block);
                } else {
                    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                                  Eigen::RowMajor>>
                        power_map(s.power.data(), block, n_bins);
                    Eigen::MatrixXf power_t = power_map.transpose();
                    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                             Eigen::RowMajor>>
                        out_map(s.block_out.data(), args.n_mels, block);
                    out_map.noalias() = fb.sparse * power_t;
                }
#ifdef MELSPEC_LIBRARY_PROFILE
                phases[3] += elapsed(mark); mark = Clock::now();
#endif
                for (int m = 0; m < args.n_mels; ++m) {
                    std::memcpy(output.data() + static_cast<size_t>(m) * n_frames + start,
                                s.block_out.data() + static_cast<size_t>(m) * block,
                                sizeof(float) * count);
                }
#ifdef MELSPEC_LIBRARY_PROFILE
                phases[4] += elapsed(mark);
#endif
                work += elapsed(block_start);
            }
            worker_seconds[tid] = work;
        };

        t = Clock::now();
        std::vector<std::thread> threads;
        threads.reserve(n_threads - 1);
        for (int i = 1; i < n_threads; ++i) threads.emplace_back(worker, i);
        worker(0);
        for (auto& th : threads) th.join();
        const double compute_s = elapsed(t);

        t = Clock::now();
        write_output(args.output, output);
        const double write_s = elapsed(t);
        if (std::getenv("MELSPEC_STATS")) {
            std::ofstream stats(std::getenv("MELSPEC_STATS"));
            stats << "{\n";
            stats << "  \"implementation\": \"cpp-library\",\n";
            stats << "  \"mode\": \"" << args.mode << "\",\n";
            stats << "  \"n_fft\": " << args.n_fft << ", \"n_mels\": " << args.n_mels
                  << ", \"block_size\": " << block << ", \"threads\": " << n_threads << ",\n";
            stats << "  \"blas_threads\": " << args.blas_threads << ",\n";
            stats << "  \"read_s\": " << read_s << ", \"setup_s\": " << setup_s
                  << ", \"compute_s\": " << compute_s << ", \"write_s\": " << write_s << ",\n";
            stats << "  \"internal_s\": " << elapsed(total_start) << ",\n";
            stats << "  \"worker_seconds_sum\": "
                  << std::accumulate(worker_seconds.begin(), worker_seconds.end(), 0.0) << ",\n";
#ifdef MELSPEC_LIBRARY_PROFILE
            std::array<double, 5> phase_total{};
            for (const auto& p : phase_seconds)
                for (size_t i = 0; i < phase_total.size(); ++i) phase_total[i] += p[i];
            stats << "  \"window_s\": " << phase_total[0]
                  << ", \"fft_s\": " << phase_total[1]
                  << ", \"power_s\": " << phase_total[2]
                  << ", \"mel_s\": " << phase_total[3]
                  << ", \"store_s\": " << phase_total[4] << ",\n";
#endif
            stats << "  \"nnz\": " << fb.sparse.nonZeros() << ",\n";
            struct rusage usage {};
            getrusage(RUSAGE_SELF, &usage);
            stats << "  \"maxrss_kib\": " << usage.ru_maxrss << "\n}\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
