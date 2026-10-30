#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>

namespace v3_cuda_pipeline {

// cuFFT's R2C/D2Z output layout is [batch][n_fft / 2 + 1].
template <typename T>
struct FftTraits;

template <>
struct FftTraits<float> {
    using Complex = cufftComplex;
};

template <>
struct FftTraits<double> {
    using Complex = cufftDoubleComplex;
};

__device__ __forceinline__ float squared_norm(const cufftComplex& z) {
    return z.x * z.x + z.y * z.y;
}

__device__ __forceinline__ double squared_norm(const cufftDoubleComplex& z) {
    return z.x * z.x + z.y * z.y;
}

// Gather directly from the complete device waveform. No CPU frame matrix exists.
// fft_input has cuFFT-compatible compact real layout [batch][n_fft].
template <typename T>
__global__ void gather_window_kernel(const T* __restrict__ y,
                                     const T* __restrict__ window,
                                     T* __restrict__ fft_input,
                                     int batch, std::size_t frame_begin,
                                     int n_fft, int hop_length) {
    const std::size_t count = static_cast<std::size_t>(batch) * n_fft;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
         i < count; i += stride) {
        const std::size_t local_frame = i / static_cast<unsigned>(n_fft);
        const std::size_t n = i - local_frame * n_fft;
        const std::size_t source =
            (frame_begin + local_frame) * static_cast<unsigned>(hop_length) + n;
        fft_input[i] = y[source] * window[n];
    }
}

// Materialize power as [batch][n_bins], preserving the frame-major FFT layout.
template <typename T>
__global__ void power_kernel(
    const typename FftTraits<T>::Complex* __restrict__ fft_output,
    T* __restrict__ power, int batch, int n_bins) {
    const std::size_t count = static_cast<std::size_t>(batch) * n_bins;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
         i < count; i += stride) {
        power[i] = squared_norm(fft_output[i]);
    }
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ double warp_sum(double value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// One warp computes one (local frame, Mel row) dot product. Lanes traverse a
// contiguous support range, so power/FFT and weight loads coalesce across lanes.
// output is the complete row-major [n_mels, n_frames] matrix, not batch-major.
template <typename T>
__global__ void mel_from_power_kernel(const T* __restrict__ power,
                                      const T* __restrict__ mel_weights,
                                      const int* __restrict__ first,
                                      const int* __restrict__ last,
                                      T* __restrict__ output, int batch,
                                      int n_bins, int n_mels,
                                      std::size_t n_frames,
                                      std::size_t frame_begin) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warps_per_block = blockDim.x >> 5;
    const std::size_t count = static_cast<std::size_t>(batch) * n_mels;
    const std::size_t warp_stride =
        static_cast<std::size_t>(gridDim.x) * warps_per_block;

    for (std::size_t work =
             static_cast<std::size_t>(blockIdx.x) * warps_per_block +
             (threadIdx.x >> 5);
         work < count; work += warp_stride) {
        const int local_frame = static_cast<int>(work % batch);
        const int m = static_cast<int>(work / batch);
        T acc = static_cast<T>(0);
        for (int k = first[m] + lane; k < last[m]; k += warpSize) {
            const std::size_t power_index =
                static_cast<std::size_t>(local_frame) * n_bins + k;
            const std::size_t weight_index =
                static_cast<std::size_t>(m) * n_bins + k;
            acc += mel_weights[weight_index] * power[power_index];
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            output[static_cast<std::size_t>(m) * n_frames + frame_begin +
                   local_frame] = acc;
        }
    }
}

// Fused alternative: no power matrix is stored. Each Mel tap reads the complex
// FFT bin and computes |X|^2. Bins shared by filters are therefore recomputed.
template <typename T>
__global__ void fused_power_mel_kernel(
    const typename FftTraits<T>::Complex* __restrict__ fft_output,
    const T* __restrict__ mel_weights, const int* __restrict__ first,
    const int* __restrict__ last, T* __restrict__ output, int batch,
    int n_bins, int n_mels, std::size_t n_frames, std::size_t frame_begin) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warps_per_block = blockDim.x >> 5;
    const std::size_t count = static_cast<std::size_t>(batch) * n_mels;
    const std::size_t warp_stride =
        static_cast<std::size_t>(gridDim.x) * warps_per_block;

    for (std::size_t work =
             static_cast<std::size_t>(blockIdx.x) * warps_per_block +
             (threadIdx.x >> 5);
         work < count; work += warp_stride) {
        const int local_frame = static_cast<int>(work % batch);
        const int m = static_cast<int>(work / batch);
        T acc = static_cast<T>(0);
        for (int k = first[m] + lane; k < last[m]; k += warpSize) {
            const std::size_t fft_index =
                static_cast<std::size_t>(local_frame) * n_bins + k;
            const std::size_t weight_index =
                static_cast<std::size_t>(m) * n_bins + k;
            acc += mel_weights[weight_index] * squared_norm(fft_output[fft_index]);
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            output[static_cast<std::size_t>(m) * n_frames + frame_begin +
                   local_frame] = acc;
        }
    }
}

// A capped grid plus grid-stride loops avoids dimension limits and 32-bit flat
// index overflow. The host may choose another cap without changing the kernels.
inline unsigned int blocks_for(std::size_t work_items,
                               unsigned int threads = 256,
                               unsigned int cap = 65535) {
    const std::size_t needed =
        (work_items + threads - 1) / static_cast<std::size_t>(threads);
    const std::size_t blocks = needed < cap ? needed : cap;
    return static_cast<unsigned int>(blocks == 0 ? 1 : blocks);
}

inline unsigned int projection_blocks(std::size_t output_items,
                                      unsigned int threads = 256,
                                      unsigned int cap = 65535) {
    const unsigned int warps = threads / 32;
    const std::size_t warps_needed =
        (output_items + warps - 1) / static_cast<std::size_t>(warps);
    const std::size_t blocks = warps_needed < cap ? warps_needed : cap;
    return static_cast<unsigned int>(blocks == 0 ? 1 : blocks);
}

}  // namespace v3_cuda_pipeline
