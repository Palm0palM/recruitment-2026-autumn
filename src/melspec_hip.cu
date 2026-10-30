#include "melspec_hip.h"
#include "melspec.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

// Initialization scaffold only. The HIP pipeline will be ported from the frozen
// CUDA design after the private ROCm/hipFFT environment is verified on gfx1201.
void compute_melspectrogram_hip(const std::vector<double>& y, int sr, int n_fft,
                                int hop_length, int n_mels, double f_min,
                                double f_max, std::vector<double>& output) {
    (void)y;
    (void)sr;
    (void)n_fft;
    (void)hop_length;
    (void)n_mels;
    (void)f_min;
    (void)f_max;
    (void)output;
    int devices = 0;
    const hipError_t status = hipGetDeviceCount(&devices);
    if (status != hipSuccess)
        throw std::runtime_error(std::string("HIP runtime: ") +
                                 hipGetErrorString(status));
    throw std::runtime_error("HIP backend scaffold is built, but the melspec "
                             "pipeline has not been implemented yet");
}
