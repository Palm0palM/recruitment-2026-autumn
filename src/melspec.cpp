#include "melspec.h"
#include "melspec_backend.h"

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

// The cross-over model will be added after backend preparation. For now auto
// selects CUDA only when a device is available; CPU remains the fallback.
bool auto_should_use_cuda(const std::vector<double>&, int, int, int, double) {
    return true;
}

}  // namespace

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    const Backend backend = requested_backend();
    if (backend == Backend::Cpu)
        return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels,
                                          f_min, f_max, output);

#ifdef MELSPEC_WITH_CUDA
    if (!cuda_backend_available()) {
        if (backend == Backend::Cuda)
            throw std::runtime_error("MELSPEC_BACKEND=cuda requested but no CUDA device is available");
        return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels,
                                          f_min, f_max, output);
    }
    if (backend == Backend::Cuda ||
        (backend == Backend::Auto && auto_should_use_cuda(y, n_fft, n_mels, hop_length, f_min))) {
        // A real CUDA compute error is propagated and never hidden as a CPU fallback.
        return compute_melspectrogram_cuda(y, sr, n_fft, hop_length, n_mels,
                                           f_min, f_max, output);
    }
    return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels,
                                      f_min, f_max, output);
#else
    if (backend == Backend::Cuda)
        throw std::runtime_error("MELSPEC_BACKEND=cuda requested but this binary was built without CUDA");
    return compute_melspectrogram_cpu(y, sr, n_fft, hop_length, n_mels,
                                      f_min, f_max, output);
#endif
}
