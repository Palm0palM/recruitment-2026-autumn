#pragma once

#include <vector>
#include <stdexcept>

// Only thrown at the GPU runtime initialization boundary, before any kernel.
// Allocation, FFT, launch and transfer errors must NOT use this type.
class BackendUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// CPU backend entry point. Uses the frozen SIMD implementation from v2/simd.
void compute_melspectrogram_cpu(const std::vector<double>& y, int sr, int n_fft,
                                int hop_length, int n_mels, double f_min,
                                double f_max, std::vector<double>& output);

#if defined(MELSPEC_WITH_CUDA)
// CUDA backend entry point, frozen from V3 CUDA commit 3e11194.
void compute_melspectrogram_cuda(const std::vector<double>& y, int sr, int n_fft,
                                 int hop_length, int n_mels, double f_min,
                                 double f_max, std::vector<double>& output);

// Only probes device availability. Does not hide errors from actual compute.
bool cuda_backend_available();
#endif
