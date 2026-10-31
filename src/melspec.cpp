#include "melspec.h"
#include "melspec_backend.h"

#if defined(MELSPEC_WITH_HIP)
#include "melspec_model_hip.h"
#else
#include "melspec_model_cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(MELSPEC_WITH_CUDA) && defined(MELSPEC_WITH_HIP)
#error "Build CPU+CUDA or CPU+HIP, not both GPU runtimes"
#endif

namespace {
enum class Backend { Cpu, Cuda, Hip, Auto };

Backend requested_backend() {
    const char* value = std::getenv("MELSPEC_BACKEND");
    if (!value || !*value || std::strcmp(value, "auto") == 0) return Backend::Auto;
    if (std::strcmp(value, "cpu") == 0) return Backend::Cpu;
    if (std::strcmp(value, "cuda") == 0) return Backend::Cuda;
    if (std::strcmp(value, "hip") == 0) return Backend::Hip;
    throw std::runtime_error("MELSPEC_BACKEND must be cpu, cuda, hip, or auto");
}

// Platform coefficients are generated from training sessions only. The
// footprint hinge is an empirical large-B16-working-set pressure term, not a
// change to FFT arithmetic. See versions/V5-final/F2_CUDA.md and F3_HIP.md.
struct Workload {
    double work, input_mb, output_mb, cache_work;
    int frames;
};

Workload describe(const std::vector<double>& y, int n_fft, int hop, int mels) {
    const int frames = 1 + static_cast<int>((y.size() - n_fft) / hop);
    const double work = static_cast<double>(frames) * n_fft * std::log2(n_fft) / 1e8;
    return {work, y.size() * 2.0 / 1e6, static_cast<double>(mels) * frames * 4.0 / 1e6,
            work * std::fmax(0.0, n_fft / 16384.0 - 1.0), frames};
}

double predict_cpu(const Workload& w) {
    return MELSPEC_MODEL_CPU_CONST + MELSPEC_MODEL_CPU_WORK * w.work +
           MELSPEC_MODEL_CPU_INPUT * w.input_mb + MELSPEC_MODEL_CPU_OUTPUT * w.output_mb +
           MELSPEC_MODEL_CPU_CACHE * w.cache_work;
}

#if defined(MELSPEC_WITH_HIP)
constexpr Backend compiled_gpu = Backend::Hip;
constexpr const char* gpu_name = "hip";
double predict_gpu(const Workload& w) {
    return MELSPEC_MODEL_HIP_CONST + MELSPEC_MODEL_HIP_WORK * w.work +
           MELSPEC_MODEL_HIP_INPUT * w.input_mb + MELSPEC_MODEL_HIP_OUTPUT * w.output_mb +
           MELSPEC_MODEL_HIP_CACHE * w.cache_work;
}
#else
constexpr const char* gpu_name = "cuda";
#if defined(MELSPEC_WITH_CUDA)
constexpr Backend compiled_gpu = Backend::Cuda;
#endif
double predict_gpu(const Workload& w) {
    return MELSPEC_MODEL_CUDA_CONST + MELSPEC_MODEL_CUDA_WORK * w.work +
           MELSPEC_MODEL_CUDA_INPUT * w.input_mb + MELSPEC_MODEL_CUDA_OUTPUT * w.output_mb +
           MELSPEC_MODEL_CUDA_CACHE * w.cache_work;
}
#endif

void log_decision(const char* chosen, const Workload& w) {
    const char* flag = std::getenv("MELSPEC_DISPATCH_LOG");
    if (!flag || !*flag || *flag == '0') return;
    std::fprintf(stderr, "dispatch: backend=%s frames=%d work=%.3f input_mb=%.3f "
                 "output_mb=%.3f cache_work=%.3f t_cpu=%.5f t_%s=%.5f\n",
                 chosen, w.frames, w.work, w.input_mb, w.output_mb, w.cache_work,
                 predict_cpu(w), gpu_name, predict_gpu(w));
}

#if defined(MELSPEC_WITH_CUDA) || defined(MELSPEC_WITH_HIP)
void compute_gpu(const std::vector<double>& y, int sr, int n_fft, int hop, int mels,
                 double f_min, double f_max, std::vector<double>& output) {
    // The frozen n_fft=1 mathematical shortcut never uses a device. Forced
    // GPU mode must still fail clearly if no device exists. Real FFTs do NOT
    // pre-probe: the existing overlapped init reports typed unavailability.
#if defined(MELSPEC_WITH_HIP)
    if (n_fft == 1 && !hip_backend_available())
        throw BackendUnavailable("HIP device unavailable");
    compute_melspectrogram_hip(y, sr, n_fft, hop, mels, f_min, f_max, output);
#else
    if (n_fft == 1 && !cuda_backend_available())
        throw BackendUnavailable("CUDA device unavailable");
    compute_melspectrogram_cuda(y, sr, n_fft, hop, mels, f_min, f_max, output);
#endif
}
#endif
} // namespace

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    const Backend backend = requested_backend();
    // Before selector frame-count division and frozen int-sized indexing.
    if (sr < 1 || n_fft < 1 || (n_fft & (n_fft - 1)) != 0 || hop_length < 1 ||
        n_mels < 1 || y.size() < static_cast<size_t>(n_fft) ||
        y.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Invalid parameters or audio too short/large for n_fft");
    const auto w = describe(y, n_fft, hop_length, n_mels);

#if defined(MELSPEC_HIP_ONLY) && MELSPEC_HIP_ONLY
    if (backend == Backend::Cpu || backend == Backend::Cuda)
        throw std::runtime_error("AMD dedicated build requires hip or auto; use CPU+HIP for CPU dispatch");
    log_decision("hip", w);
    return compute_gpu(y, sr, n_fft, hop_length, n_mels, f_min, f_max, output);
#else
    if (backend == Backend::Cuda || backend == Backend::Hip) {
#if defined(MELSPEC_WITH_CUDA) || defined(MELSPEC_WITH_HIP)
        if (backend == compiled_gpu) {
            log_decision(gpu_name, w);
            // No catch in forced mode: both unavailable and real errors report.
            return compute_gpu(y, sr, n_fft, hop_length, n_mels, f_min, f_max, output);
        }
#endif
        throw std::runtime_error("Requested GPU backend is not compiled into this binary");
    }

#if defined(MELSPEC_WITH_CUDA) || defined(MELSPEC_WITH_HIP)
    bool calibrated = true;
#if defined(MELSPEC_WITH_HIP) && !MELSPEC_HIP_MODEL_CALIBRATED
    calibrated = false;
#endif
    if (backend == Backend::Auto && n_fft > 1 && calibrated && predict_gpu(w) < predict_cpu(w)) {
        try {
            compute_gpu(y, sr, n_fft, hop_length, n_mels, f_min, f_max, output);
            log_decision(gpu_name, w);
            return;
        } catch (const BackendUnavailable&) {
            // This type is reserved for no-device/insufficient-driver at init,
            // before numerical kernels. No other compute error is hidden.
        }
    }
#endif
    log_decision("cpu", w);
    return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels,
                                      f_min, f_max, output);
#endif
}
