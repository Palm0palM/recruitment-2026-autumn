#include "melspec.h"
#include "melspec_backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Backend { Cpu, Cuda, Auto };

Backend requested_backend() {
    const char* value = std::getenv("MELSPEC_BACKEND");
    if (value == nullptr || *value == '\0') return Backend::Auto;
    const std::string name(value);
    if (name == "cpu") return Backend::Cpu;
    if (name == "cuda") return Backend::Cuda;
    if (name == "auto") return Backend::Auto;
    throw std::runtime_error("MELSPEC_BACKEND must be cpu, cuda, or auto");
}

// Full-process cost model fitted on the target node (Xeon Gold 6338, 16-CPU
// cpuset, AVX-512 FP32 Batch16 + L40, CUDA 13.4). Both formulas are linear and
// monotone in three measurable workload terms:
//
//   work    = n_frames * n_fft * log2(n_fft) / 1e8   (FFT volume)
//   input   = samples * sizeof(int16)           / 1e6 (input payload, MB)
//   output  = n_mels * n_frames * sizeof(float) / 1e6 (output payload, MB)
//
// The constant terms are the measured per-process fixed cost, which is what
// makes small inputs prefer CPU: CUDA pays context/cuFFT/transfer setup, the
// SIMD path only pays process startup and PCM decoding.
//
//   T_cpu  = 0.037173 + 0.0016168 * work + 0.0027815 * output
//   T_cuda = 0.245811 + 8.7131e-5 * work + 0.0036889 * input + 0.0021713 * output
//
// Coefficients come from scripts/fit_hybrid_model.py over 38 measured points
// (see versions/V3-hybrid/DISPATCH_MODEL.md and raw/h1). They are build-time
// overridable so another hardware profile can be recalibrated without code
// changes.
#ifndef MELSPEC_MODEL_CPU_CONST
#define MELSPEC_MODEL_CPU_CONST 0.0371727216
#endif
#ifndef MELSPEC_MODEL_CPU_WORK
#define MELSPEC_MODEL_CPU_WORK 0.00161676324
#endif
#ifndef MELSPEC_MODEL_CPU_INPUT
#define MELSPEC_MODEL_CPU_INPUT 0.0
#endif
#ifndef MELSPEC_MODEL_CPU_OUTPUT
#define MELSPEC_MODEL_CPU_OUTPUT 0.00278147181
#endif
#ifndef MELSPEC_MODEL_CUDA_CONST
#define MELSPEC_MODEL_CUDA_CONST 0.245810965
#endif
#ifndef MELSPEC_MODEL_CUDA_WORK
#define MELSPEC_MODEL_CUDA_WORK 8.71311565e-05
#endif
#ifndef MELSPEC_MODEL_CUDA_INPUT
#define MELSPEC_MODEL_CUDA_INPUT 0.0036888979
#endif
#ifndef MELSPEC_MODEL_CUDA_OUTPUT
#define MELSPEC_MODEL_CUDA_OUTPUT 0.00217130597
#endif

struct Workload {
    double work;
    double input_mb;
    double output_mb;
    int n_frames;
};

Workload describe(const std::vector<double>& y, int n_fft, int hop_length,
                  int n_mels) {
    const long long samples = static_cast<long long>(y.size());
    const long long frames =
        samples >= n_fft ? 1 + (samples - n_fft) / hop_length : 0;
    Workload w{0.0, 0.0, 0.0, static_cast<int>(frames)};
    if (frames <= 0 || n_fft <= 1) return w;
    w.work = static_cast<double>(frames) * n_fft * std::log2(static_cast<double>(n_fft)) / 1e8;
    w.input_mb = static_cast<double>(samples) * 2.0 / 1e6;
    w.output_mb = static_cast<double>(n_mels) * static_cast<double>(frames) * 4.0 / 1e6;
    return w;
}

double predict_cpu(const Workload& w) {
    return MELSPEC_MODEL_CPU_CONST + MELSPEC_MODEL_CPU_WORK * w.work +
           MELSPEC_MODEL_CPU_INPUT * w.input_mb +
           MELSPEC_MODEL_CPU_OUTPUT * w.output_mb;
}

double predict_cuda(const Workload& w) {
    return MELSPEC_MODEL_CUDA_CONST + MELSPEC_MODEL_CUDA_WORK * w.work +
           MELSPEC_MODEL_CUDA_INPUT * w.input_mb +
           MELSPEC_MODEL_CUDA_OUTPUT * w.output_mb;
}

#ifdef MELSPEC_WITH_CUDA
// Pure arithmetic: no CUDA API call, no trial run, no allocation.
bool model_prefers_cuda(const Workload& w) { return predict_cuda(w) < predict_cpu(w); }
#endif

void log_decision(const char* chosen, const Workload& w) {
    const char* flag = std::getenv("MELSPEC_DISPATCH_LOG");
    if (flag == nullptr || *flag == '\0' || *flag == '0') return;
    std::fprintf(stderr,
                 "dispatch: backend=%s frames=%d work=%.3f input_mb=%.3f "
                 "output_mb=%.3f t_cpu=%.4f t_cuda=%.4f\n",
                 chosen, w.n_frames, w.work, w.input_mb, w.output_mb,
                 predict_cpu(w), predict_cuda(w));
}

}  // namespace

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    const Backend backend = requested_backend();
    if (backend == Backend::Cpu) {
        log_decision("cpu (forced)", describe(y, n_fft, hop_length, n_mels));
        return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels,
                                          f_min, f_max, output);
    }

#ifdef MELSPEC_WITH_CUDA
    if (backend == Backend::Cuda) {
        if (!cuda_backend_available())
            throw std::runtime_error("MELSPEC_BACKEND=cuda requested but no CUDA device is available");
        log_decision("cuda (forced)", describe(y, n_fft, hop_length, n_mels));
        // A real CUDA compute error is propagated and never hidden as a CPU fallback.
        return compute_melspectrogram_cuda(y, sr, n_fft, hop_length, n_mels,
                                           f_min, f_max, output);
    }

    const Workload w = describe(y, n_fft, hop_length, n_mels);
    if (model_prefers_cuda(w)) {
        // Only the auto path probes the device, and only when the model already
        // predicted a GPU win. Availability is an environment fact, not a
        // compute error, so falling back here is safe.
        if (cuda_backend_available()) {
            log_decision("cuda (auto)", w);
            return compute_melspectrogram_cuda(y, sr, n_fft, hop_length, n_mels,
                                               f_min, f_max, output);
        }
        log_decision("cpu (auto, no CUDA device)", w);
    } else {
        log_decision("cpu (auto)", w);
    }
    return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels, f_min,
                                      f_max, output);
#else
    if (backend == Backend::Cuda)
        throw std::runtime_error("MELSPEC_BACKEND=cuda requested but this binary was built without CUDA");
    log_decision("cpu (auto)", describe(y, n_fft, hop_length, n_mels));
    return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels, f_min,
                                      f_max, output);
#endif
}
