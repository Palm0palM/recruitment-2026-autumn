#pragma once

#include <vector>

// Internal CUDA backend entry point; the public melspec.h interface is unchanged.
void compute_melspectrogram_cuda(const std::vector<double>& y, int sr, int n_fft,
                                 int hop_length, int n_mels, double f_min,
                                 double f_max, std::vector<double>& output);
