#include "melspec_cuda.h"
#include "melspec.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef MELSPEC_CUDA_PRECISION
#define MELSPEC_CUDA_PRECISION 32
#endif
#ifndef MELSPEC_CUDA_BATCH
#define MELSPEC_CUDA_BATCH 256
#endif
#ifndef MELSPEC_CUDA_FUSED
#define MELSPEC_CUDA_FUSED 1
#endif
#ifndef MELSPEC_CUDA_PINNED
#define MELSPEC_CUDA_PINNED 0
#endif

#ifdef MELSPEC_CUDA_PROFILE
#include <chrono>
#include <cstdio>
using ProfileClock = std::chrono::steady_clock;
double profile_elapsed(ProfileClock::time_point start) {
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}
#define PROFILE_POINT(name) auto name = ProfileClock::now()
#define PROFILE_ADD(variable, point) variable += profile_elapsed(point); point = ProfileClock::now()
#define PROFILE_ADD_SYNC(variable, point) \
    cudaDeviceSynchronize(); PROFILE_ADD(variable, point)
#define PROFILE_PRINT(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define PROFILE_POINT(name)
#define PROFILE_ADD(variable, point)
#define PROFILE_ADD_SYNC(variable, point)
#define PROFILE_PRINT(...)
#endif

namespace {

using Real = std::conditional_t<MELSPEC_CUDA_PRECISION == 32, float, double>;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void cufft_check(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT error " +
                                 std::to_string(static_cast<int>(status)));
}

template <typename T> struct FftTraits;
template <> struct FftTraits<float> {
    using Complex = cufftComplex;
    static constexpr cufftType type = CUFFT_R2C;
    static cufftResult execute(cufftHandle plan, float* input, Complex* output) {
        return cufftExecR2C(plan, input, output);
    }
};
template <> struct FftTraits<double> {
    using Complex = cufftDoubleComplex;
    static constexpr cufftType type = CUFFT_D2Z;
    static cufftResult execute(cufftHandle plan, double* input, Complex* output) {
        return cufftExecD2Z(plan, input, output);
    }
};
using Complex = typename FftTraits<Real>::Complex;

__device__ __forceinline__ float squared_norm(const cufftComplex& z) {
    return z.x * z.x + z.y * z.y;
}
__device__ __forceinline__ double squared_norm(const cufftDoubleComplex& z) {
    return z.x * z.x + z.y * z.y;
}

template <typename T>
__global__ void gather_window_kernel(const T* __restrict__ y,
                                     const T* __restrict__ window,
                                     T* __restrict__ fft_input, int batch,
                                     size_t frame_begin, int n_fft, int hop_length) {
    const size_t count = static_cast<size_t>(batch) * n_fft;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const size_t frame = i / static_cast<unsigned>(n_fft);
        const size_t n = i - frame * n_fft;
        fft_input[i] = y[(frame_begin + frame) * hop_length + n] * window[n];
    }
}

template <typename T>
__global__ void power_kernel(const typename FftTraits<T>::Complex* __restrict__ fft_output,
                             T* __restrict__ power, int batch, int n_bins) {
    const size_t count = static_cast<size_t>(batch) * n_bins;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride)
        power[i] = squared_norm(fft_output[i]);
}

template <typename T> __device__ __forceinline__ T warp_sum(T value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

template <typename T>
__global__ void mel_kernel(const typename FftTraits<T>::Complex* __restrict__ fft_output,
                           const T* __restrict__ power,
                           const T* __restrict__ mel_weights,
                           const int* __restrict__ first,
                           const int* __restrict__ last, T* __restrict__ output,
                           int batch, int n_bins, int n_mels,
                           size_t n_frames, size_t frame_begin) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warps_per_block = blockDim.x >> 5;
    const size_t count = static_cast<size_t>(batch) * n_mels;
    const size_t warp_stride = static_cast<size_t>(gridDim.x) * warps_per_block;
    for (size_t work = static_cast<size_t>(blockIdx.x) * warps_per_block +
                       (threadIdx.x >> 5);
         work < count; work += warp_stride) {
        const int frame = static_cast<int>(work % batch);
        const int m = static_cast<int>(work / batch);
        T acc = static_cast<T>(0);
        for (int k = first[m] + lane; k < last[m]; k += warpSize) {
            const size_t bin = static_cast<size_t>(frame) * n_bins + k;
            const T value = power ? power[bin] : squared_norm(fft_output[bin]);
            acc += mel_weights[static_cast<size_t>(m) * n_bins + k] * value;
        }
        acc = warp_sum(acc);
        if (lane == 0)
            output[static_cast<size_t>(m) * n_frames + frame_begin + frame] = acc;
    }
}

unsigned blocks_for(size_t items, unsigned threads = 256) {
    return static_cast<unsigned>(std::max<size_t>(1,
        std::min<size_t>((items + threads - 1) / threads, 65535)));
}

unsigned projection_blocks(size_t items) {
    return blocks_for(items, 8);  // eight warps per 256-thread block
}

template <typename T> class DeviceBuffer {
    T* pointer_ = nullptr;
public:
    explicit DeviceBuffer(size_t count = 0) {
        if (count)
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
                       "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
    operator bool() const { return pointer_ != nullptr; }
};

#if MELSPEC_CUDA_PINNED
template <typename T> class PinnedBuffer {
    T* pointer_ = nullptr;
public:
    explicit PinnedBuffer(size_t count) {
        if (count)
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&pointer_),
                                     count * sizeof(T), cudaHostAllocDefault),
                       "cudaHostAlloc");
    }
    ~PinnedBuffer() { cudaFreeHost(pointer_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    T* get() const { return pointer_; }
    T& operator[](size_t i) const { return pointer_[i]; }
};
#endif

class CufftPlan {
    cufftHandle handle_ = 0;
public:
    CufftPlan(int n_fft, int batch) {
        const int bins = n_fft / 2 + 1;
        cufft_check(cufftPlanMany(&handle_, 1, &n_fft, nullptr, 1, n_fft,
                                  nullptr, 1, bins, FftTraits<Real>::type, batch),
                    "cufftPlanMany");
    }
    ~CufftPlan() { cufftDestroy(handle_); }
    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;
    cufftHandle get() const { return handle_; }
    size_t workspace() const {
        size_t bytes = 0;
        cufft_check(cufftGetSize(handle_, &bytes), "cufftGetSize");
        return bytes;
    }
};

void copy_to_device(void* destination, const void* source, size_t bytes) {
    if (bytes)
        cuda_check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
}

}  // namespace

void compute_melspectrogram_cuda(const std::vector<double>& y, int sr, int n_fft,
                                 int hop_length, int n_mels, double f_min,
                                 double f_max, std::vector<double>& output) {
#ifdef MELSPEC_CUDA_PROFILE
    double host_setup_ms = 0, device_alloc_ms = 0, h2d_ms = 0, batch_alloc_ms = 0;
    double plan_ms = 0, gather_ms = 0, fft_ms = 0, mel_ms = 0, pipeline_sync_ms = 0;
    double d2h_ms = 0, host_convert_ms = 0;
#endif
    PROFILE_POINT(total_start);
    PROFILE_POINT(phase);
    const int n_samples = static_cast<int>(y.size());
    if (n_fft < 1 || (n_fft & (n_fft - 1)) != 0 || hop_length < 1 ||
        n_mels < 1 || sr < 1 || n_samples < n_fft)
        throw std::runtime_error("Invalid parameters or audio too short for n_fft");
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    const int n_bins = n_fft / 2 + 1;
    output.resize(static_cast<size_t>(n_mels) * n_frames);
    if (n_fft == 1) {
        std::fill(output.begin(), output.end(), 0.0);
        return;
    }

#if MELSPEC_CUDA_PINNED
    PinnedBuffer<Real> host_y(y.size());
    std::copy(y.begin(), y.end(), host_y.get());
#else
    std::vector<Real> host_y(y.begin(), y.end());
#endif
    std::vector<Real> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] = static_cast<Real>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / n_fft)));
    const auto mel_fb = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    std::vector<Real> weights(static_cast<size_t>(n_mels) * n_bins);
    for (int m = 0; m < n_mels; ++m)
        for (int k = 0; k < n_bins; ++k)
            weights[static_cast<size_t>(m) * n_bins + k] = static_cast<Real>(mel_fb[m][k]);
    std::vector<int> first(n_mels), last(n_mels, n_bins);
    for (int m = 0; m < n_mels; ++m) {
        while (first[m] < n_bins && mel_fb[m][first[m]] == 0.0) ++first[m];
        while (last[m] > first[m] && mel_fb[m][last[m] - 1] == 0.0) --last[m];
    }
    PROFILE_ADD(host_setup_ms, phase);

    DeviceBuffer<Real> device_y(y.size());
    DeviceBuffer<Real> device_window(window.size());
    DeviceBuffer<Real> device_weights(weights.size());
    DeviceBuffer<int> device_first(first.size());
    DeviceBuffer<int> device_last(last.size());
    DeviceBuffer<Real> device_output(output.size());
    PROFILE_ADD(device_alloc_ms, phase);
#if MELSPEC_CUDA_PINNED
    copy_to_device(device_y.get(), host_y.get(), y.size() * sizeof(Real));
#else
    copy_to_device(device_y.get(), host_y.data(), y.size() * sizeof(Real));
#endif
    copy_to_device(device_window.get(), window.data(), window.size() * sizeof(Real));
    copy_to_device(device_weights.get(), weights.data(), weights.size() * sizeof(Real));
    copy_to_device(device_first.get(), first.data(), first.size() * sizeof(int));
    copy_to_device(device_last.get(), last.data(), last.size() * sizeof(int));
    PROFILE_ADD(h2d_ms, phase);

    const int capacity = std::min(MELSPEC_CUDA_BATCH, n_frames);
    DeviceBuffer<Real> fft_input(static_cast<size_t>(capacity) * n_fft);
    DeviceBuffer<Complex> fft_output(static_cast<size_t>(capacity) * n_bins);
#if !MELSPEC_CUDA_FUSED
    DeviceBuffer<Real> power(static_cast<size_t>(capacity) * n_bins);
#endif
    PROFILE_ADD(batch_alloc_ms, phase);
    CufftPlan full_plan(n_fft, capacity);
    const int tail = n_frames % capacity;
    CufftPlan tail_plan(n_fft, tail ? tail : capacity);
    PROFILE_ADD(plan_ms, phase);
    PROFILE_PRINT("cufft_workspace full=%zu tail=%zu bytes\n",
                  full_plan.workspace(), tail_plan.workspace());
#ifdef MELSPEC_CUDA_PROFILE
    size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    PROFILE_PRINT("cuda_memory free=%zu total=%zu used=%zu bytes\n",
                  free_bytes, total_bytes, total_bytes - free_bytes);
#endif

    for (int frame_begin = 0; frame_begin < n_frames; frame_begin += capacity) {
        const int batch = std::min(capacity, n_frames - frame_begin);
        const int items = batch * n_fft;
        PROFILE_POINT(phase);
        gather_window_kernel<<<blocks_for(items), 256>>>(
            device_y.get(), device_window.get(), fft_input.get(), batch,
            static_cast<size_t>(frame_begin), n_fft, hop_length);
        PROFILE_ADD_SYNC(gather_ms, phase);
        auto& plan = batch == capacity ? full_plan : tail_plan;
        cufft_check(FftTraits<Real>::execute(plan.get(), fft_input.get(), fft_output.get()),
                    "cuFFT execute");
        PROFILE_ADD_SYNC(fft_ms, phase);
#if MELSPEC_CUDA_FUSED
        const Real* power_pointer = nullptr;
#else
        power_kernel<<<blocks_for(batch * n_bins), 256>>>(
            fft_output.get(), power.get(), batch, n_bins);
        const Real* power_pointer = power.get();
#endif
        mel_kernel<<<projection_blocks(static_cast<size_t>(batch) * n_mels), 256>>>(
            fft_output.get(), power_pointer, device_weights.get(), device_first.get(),
            device_last.get(), device_output.get(), batch, n_bins, n_mels,
            static_cast<size_t>(n_frames), static_cast<size_t>(frame_begin));
        PROFILE_ADD_SYNC(mel_ms, phase);
    }
    cuda_check(cudaGetLastError(), "CUDA kernel launch");
    cuda_check(cudaDeviceSynchronize(), "CUDA pipeline synchronize");
    PROFILE_ADD_SYNC(pipeline_sync_ms, phase);

#if MELSPEC_CUDA_PINNED
    PinnedBuffer<Real> host_output(output.size());
    cuda_check(cudaMemcpy(host_output.get(), device_output.get(),
                          output.size() * sizeof(Real), cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H output");
    PROFILE_ADD(d2h_ms, phase);
    std::copy(host_output.get(), host_output.get() + output.size(), output.begin());
    PROFILE_ADD(host_convert_ms, phase);
#else
    if constexpr (std::is_same_v<Real, double>) {
        cuda_check(cudaMemcpy(output.data(), device_output.get(),
                              output.size() * sizeof(Real), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H output");
        PROFILE_ADD(d2h_ms, phase);
    } else {
        std::vector<Real> host_output(output.size());
        cuda_check(cudaMemcpy(host_output.data(), device_output.get(),
                              output.size() * sizeof(Real), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H output");
        PROFILE_ADD(d2h_ms, phase);
        std::copy(host_output.begin(), host_output.end(), output.begin());
        PROFILE_ADD(host_convert_ms, phase);
    }
#endif
    PROFILE_PRINT("cuda_profile precision=%d batch=%d fused=%d host_setup_ms=%.3f device_alloc_ms=%.3f h2d_ms=%.3f batch_alloc_ms=%.3f plan_ms=%.3f gather_ms=%.3f fft_ms=%.3f mel_ms=%.3f pipeline_sync_ms=%.3f d2h_ms=%.3f host_convert_ms=%.3f total_ms=%.3f\n",
                  MELSPEC_CUDA_PRECISION, MELSPEC_CUDA_BATCH, MELSPEC_CUDA_FUSED,
                  host_setup_ms, device_alloc_ms, h2d_ms, batch_alloc_ms, plan_ms,
                  gather_ms, fft_ms, mel_ms, pipeline_sync_ms, d2h_ms, host_convert_ms,
                  profile_elapsed(total_start));
}
