#include "melspec_cuda.h"
#include "melspec.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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
#ifndef MELSPEC_CUDA_OVERLAP_INIT
#define MELSPEC_CUDA_OVERLAP_INIT 0
#endif
#ifndef MELSPEC_CUDA_DEFER_RESIZE
#define MELSPEC_CUDA_DEFER_RESIZE 0
#endif
#ifndef MELSPEC_CUDA_ASYNC_OUTPUT
#define MELSPEC_CUDA_ASYNC_OUTPUT 1
#endif
#ifndef MELSPEC_CUDA_RESIZE_THREAD
#define MELSPEC_CUDA_RESIZE_THREAD 1
#endif
#ifndef MELSPEC_CUDA_BACKGROUND_INIT
#define MELSPEC_CUDA_BACKGROUND_INIT 0
#endif
#ifndef MELSPEC_CUDA_GPU_INPUT_CONVERT
#define MELSPEC_CUDA_GPU_INPUT_CONVERT 0
#endif
#ifndef MELSPEC_CUDA_STAGING_ROWS
#define MELSPEC_CUDA_STAGING_ROWS 16
#endif

#if MELSPEC_CUDA_OVERLAP_INIT || MELSPEC_CUDA_RESIZE_THREAD || MELSPEC_CUDA_BACKGROUND_INIT
#include <exception>
#include <thread>
#endif
#if MELSPEC_CUDA_RESIZE_THREAD
struct OutputResize {
    std::exception_ptr error;
    std::thread thread;
    OutputResize(std::vector<double>& output, size_t elements) : thread([this, &output, elements] {
        try {
            output.resize(elements);
        } catch (...) {
            error = std::current_exception();
        }
    }) {}
    ~OutputResize() {
        if (thread.joinable()) thread.join();
    }
    void finish() {
        if (thread.joinable()) thread.join();
        if (error) std::rethrow_exception(error);
    }
};
#endif
#if MELSPEC_CUDA_BACKGROUND_INIT
class BackgroundTask {
    std::exception_ptr error_;
    std::thread thread_;
public:
    template <class Function>
    explicit BackgroundTask(Function function)
        : thread_([this, function = std::move(function)]() mutable {
              try {
                  function();
              } catch (...) {
                  error_ = std::current_exception();
              }
          }) {}
    ~BackgroundTask() {
        if (thread_.joinable()) thread_.join();
    }
    BackgroundTask(const BackgroundTask&) = delete;
    BackgroundTask& operator=(const BackgroundTask&) = delete;
    void finish() {
        if (thread_.joinable()) thread_.join();
        if (error_) std::rethrow_exception(error_);
    }
};
#endif
#if MELSPEC_CUDA_OVERLAP_INIT
struct ContextInit {
    std::exception_ptr error;  // initialize before the worker may write it
    std::thread thread;
    ContextInit() : thread([this] {
        try {
            const cudaError_t status = cudaFree(nullptr);
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("CUDA context initialization: ") +
                                         cudaGetErrorString(status));
        } catch (...) {
            error = std::current_exception();
        }
    }) {}
    ~ContextInit() {
        if (thread.joinable()) thread.join();
    }
    void finish() {
        if (thread.joinable()) thread.join();
        if (error) std::rethrow_exception(error);
    }
};
#endif

#ifdef MELSPEC_CUDA_PROFILE
#include <chrono>
#include <cstdio>
using ProfileClock = std::chrono::steady_clock;
double profile_elapsed(ProfileClock::time_point start) {
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}
#define PROFILE_POINT(name) name = ProfileClock::now()
#define PROFILE_ADD(variable, point) variable += profile_elapsed(point); point = ProfileClock::now()
#define PROFILE_ADD_SYNC(variable, point) \
    cuda_check(cudaDeviceSynchronize(), "CUDA profile synchronize"); PROFILE_ADD(variable, point)
#define PROFILE_PRINT(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define PROFILE_POINT(name)
#define PROFILE_ADD(variable, point)
#define PROFILE_ADD_SYNC(variable, point)
#define PROFILE_PRINT(...)
#endif

namespace {

using Real = std::conditional_t<MELSPEC_CUDA_PRECISION == 32, float, double>;
static_assert(MELSPEC_CUDA_BATCH > 0, "CUDA_BATCH must be positive");
static_assert(MELSPEC_CUDA_STAGING_ROWS > 0, "CUDA_STAGING_ROWS must be positive");
static_assert(!(MELSPEC_CUDA_PINNED && MELSPEC_CUDA_ASYNC_OUTPUT),
              "Use either full pinned output or async chunked output, not both");
static_assert(!(MELSPEC_CUDA_RESIZE_THREAD && MELSPEC_CUDA_DEFER_RESIZE),
              "Use either resize thread or deferred resize, not both");
static_assert(!MELSPEC_CUDA_GPU_INPUT_CONVERT || MELSPEC_CUDA_PRECISION == 32,
              "GPU input conversion experiment is FP32-only");

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

#if MELSPEC_CUDA_GPU_INPUT_CONVERT
template <typename In, typename Out>
__global__ void convert_input_kernel(const In* __restrict__ input,
                                     Out* __restrict__ output, size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride)
        output[i] = static_cast<Out>(input[i]);
}
#endif

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

#if MELSPEC_CUDA_PINNED || MELSPEC_CUDA_ASYNC_OUTPUT
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

#if MELSPEC_CUDA_ASYNC_OUTPUT
class CudaStream {
    cudaStream_t stream_ = nullptr;
public:
    CudaStream() {
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    }
    ~CudaStream() { cudaStreamDestroy(stream_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }
};

class CudaEvent {
    cudaEvent_t event_ = nullptr;
public:
    CudaEvent() { cuda_check(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() { cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }
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

#if MELSPEC_CUDA_BACKGROUND_INIT
struct DeviceState {
#if MELSPEC_CUDA_GPU_INPUT_CONVERT
    DeviceBuffer<double> y_input;
#endif
    DeviceBuffer<Real> y;
    DeviceBuffer<Real> window;
    DeviceBuffer<Real> weights;
    DeviceBuffer<int> first;
    DeviceBuffer<int> last;
    DeviceBuffer<Real> output;
    DeviceBuffer<Real> fft_input;
    DeviceBuffer<Complex> fft_output;
#if !MELSPEC_CUDA_FUSED
    DeviceBuffer<Real> power;
#endif
    CufftPlan full_plan;
    CufftPlan tail_plan;

    DeviceState(size_t n_samples, int n_fft, size_t weight_count, size_t mel_count,
                size_t output_elements, int capacity, int n_bins, int tail)
        :
#if MELSPEC_CUDA_GPU_INPUT_CONVERT
          y_input(n_samples),
#endif
          y(n_samples),
          window(n_fft),
          weights(weight_count),
          first(mel_count),
          last(mel_count),
          output(output_elements),
          fft_input(static_cast<size_t>(capacity) * n_fft),
          fft_output(static_cast<size_t>(capacity) * n_bins),
#if !MELSPEC_CUDA_FUSED
          power(static_cast<size_t>(capacity) * n_bins),
#endif
          full_plan(n_fft, capacity),
          tail_plan(n_fft, tail ? tail : capacity) {}
};
#endif

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
    ProfileClock::time_point total_start, phase, async_start;
    double device_alloc_ms = 0, h2d_ms = 0, batch_alloc_ms = 0, device_alloc_wait_ms = 0;
    double plan_ms = 0, gather_ms = 0, fft_ms = 0, mel_ms = 0, pipeline_sync_ms = 0;
    double d2h_ms = 0, host_convert_ms = 0;
    double y_convert_ms = 0, window_fb_ms = 0, output_resize_ms = 0, staging_alloc_ms = 0;
    double gpu_input_convert_ms = 0;
    double async_output_total_ms = 0, resize_wait_ms = 0;
#endif
    PROFILE_POINT(total_start);
    PROFILE_POINT(phase);
    if (y.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Input exceeds the protected CLI's int size limit");
    const int n_samples = static_cast<int>(y.size());
    if (n_fft < 1 || (n_fft & (n_fft - 1)) != 0 || hop_length < 1 ||
        n_mels < 1 || sr < 1 || n_samples < n_fft)
        throw std::runtime_error("Invalid parameters or audio too short for n_fft");
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    const int n_bins = n_fft / 2 + 1;
    const int capacity = std::min(MELSPEC_CUDA_BATCH, n_frames);
    const int tail = n_frames % capacity;
    const size_t output_elements = static_cast<size_t>(n_mels) * n_frames;
    if (n_fft == 1) {
        output.resize(output_elements);
        std::fill(output.begin(), output.end(), 0.0);
        return;
    }
#if MELSPEC_CUDA_RESIZE_THREAD
    OutputResize output_resize(output, output_elements);
#elif !MELSPEC_CUDA_DEFER_RESIZE
    PROFILE_POINT(phase);
    output.resize(output_elements);
    PROFILE_ADD(output_resize_ms, phase);
#endif
#if MELSPEC_CUDA_BACKGROUND_INIT
    std::unique_ptr<DeviceState> device_state;
    BackgroundTask device_init([&] {
        device_state = std::make_unique<DeviceState>(
            y.size(), n_fft, static_cast<size_t>(n_mels) * n_bins, n_mels,
            output_elements, capacity, n_bins, tail);
    });
#elif MELSPEC_CUDA_OVERLAP_INIT
    ContextInit context_init;
#endif

    PROFILE_POINT(phase);
#if !MELSPEC_CUDA_GPU_INPUT_CONVERT
#if MELSPEC_CUDA_PINNED
    PinnedBuffer<Real> host_y(y.size());
    std::copy(y.begin(), y.end(), host_y.get());
#else
    std::vector<Real> host_y(y.begin(), y.end());
#endif
#endif
    PROFILE_ADD(y_convert_ms, phase);
    PROFILE_POINT(phase);
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
    PROFILE_ADD(window_fb_ms, phase);
#if MELSPEC_CUDA_BACKGROUND_INIT
    PROFILE_POINT(phase);
    device_init.finish();
    PROFILE_ADD(device_alloc_wait_ms, phase);
#if MELSPEC_CUDA_GPU_INPUT_CONVERT
    auto& device_y_input = device_state->y_input;
#endif
    auto& device_y = device_state->y;
    auto& device_window = device_state->window;
    auto& device_weights = device_state->weights;
    auto& device_first = device_state->first;
    auto& device_last = device_state->last;
    auto& device_output = device_state->output;
    auto& fft_input = device_state->fft_input;
    auto& fft_output = device_state->fft_output;
#if !MELSPEC_CUDA_FUSED
    auto& power = device_state->power;
#endif
    auto& full_plan = device_state->full_plan;
    auto& tail_plan = device_state->tail_plan;
#else
#if MELSPEC_CUDA_OVERLAP_INIT
    context_init.finish();
#endif
    DeviceBuffer<Real> device_y(y.size());
    DeviceBuffer<Real> device_window(window.size());
    DeviceBuffer<Real> device_weights(weights.size());
    DeviceBuffer<int> device_first(first.size());
    DeviceBuffer<int> device_last(last.size());
    DeviceBuffer<Real> device_output(output_elements);
#if MELSPEC_CUDA_GPU_INPUT_CONVERT
    DeviceBuffer<double> device_y_input(y.size());
#endif
    PROFILE_ADD(device_alloc_ms, phase);
    DeviceBuffer<Real> fft_input(static_cast<size_t>(capacity) * n_fft);
    DeviceBuffer<Complex> fft_output(static_cast<size_t>(capacity) * n_bins);
#if !MELSPEC_CUDA_FUSED
    DeviceBuffer<Real> power(static_cast<size_t>(capacity) * n_bins);
#endif
    PROFILE_ADD(batch_alloc_ms, phase);
    CufftPlan full_plan(n_fft, capacity);
    CufftPlan tail_plan(n_fft, tail ? tail : capacity);
    PROFILE_ADD(plan_ms, phase);
#endif
#if MELSPEC_CUDA_GPU_INPUT_CONVERT
    copy_to_device(device_y_input.get(), y.data(),
                   y.size() * sizeof(double));
    PROFILE_POINT(phase);
    convert_input_kernel<<<blocks_for(y.size()), 256>>>(
        device_y_input.get(), device_y.get(), y.size());
    PROFILE_ADD_SYNC(gpu_input_convert_ms, phase);
#elif MELSPEC_CUDA_PINNED
    copy_to_device(device_y.get(), host_y.get(), y.size() * sizeof(Real));
#else
    copy_to_device(device_y.get(), host_y.data(), y.size() * sizeof(Real));
#endif
    copy_to_device(device_window.get(), window.data(), window.size() * sizeof(Real));
    copy_to_device(device_weights.get(), weights.data(), weights.size() * sizeof(Real));
    copy_to_device(device_first.get(), first.data(), first.size() * sizeof(int));
    copy_to_device(device_last.get(), last.data(), last.size() * sizeof(int));
    PROFILE_ADD(h2d_ms, phase);
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
        const size_t items = static_cast<size_t>(batch) * n_fft;
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
        power_kernel<<<blocks_for(static_cast<size_t>(batch) * n_bins), 256>>>(
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

#if MELSPEC_CUDA_ASYNC_OUTPUT
    PROFILE_POINT(async_start);
    CudaEvent compute_done;
    cuda_check(cudaEventRecord(compute_done.get()), "cudaEventRecord compute");
    const int staging_rows = std::min(MELSPEC_CUDA_STAGING_ROWS, n_mels);
    const int chunks = (n_mels + staging_rows - 1) / staging_rows;
    const size_t staging_values = static_cast<size_t>(staging_rows) * n_frames;
    PROFILE_POINT(phase);
    PinnedBuffer<Real> staging0(staging_values);
    PinnedBuffer<Real> staging1(staging_values);
    CudaStream streams[2];
    CudaEvent copies[2];
    PROFILE_ADD(staging_alloc_ms, phase);
    cudaStreamWaitEvent(streams[0].get(), compute_done.get());
    cudaStreamWaitEvent(streams[1].get(), compute_done.get());
    auto enqueue = [&](int chunk) {
        const int p = chunk & 1;
        const int row_begin = chunk * staging_rows;
        const size_t rows = std::min(staging_rows, n_mels - row_begin);
        const size_t bytes = rows * n_frames * sizeof(Real);
        cuda_check(cudaMemcpyAsync((p == 0 ? staging0.get() : staging1.get()),
                                   device_output.get() + static_cast<size_t>(row_begin) * n_frames,
                                   bytes, cudaMemcpyDeviceToHost, streams[p].get()),
                   "cudaMemcpyAsync D2H output");
        cuda_check(cudaEventRecord(copies[p].get(), streams[p].get()),
                   "cudaEventRecord D2H");
    };
    for (int i = 0; i < std::min(2, chunks); ++i) enqueue(i);
#if MELSPEC_CUDA_RESIZE_THREAD
    PROFILE_POINT(phase);
    output_resize.finish();
    PROFILE_ADD(resize_wait_ms, phase);
#elif MELSPEC_CUDA_DEFER_RESIZE
    PROFILE_POINT(phase);
    output.resize(output_elements);
    PROFILE_ADD(output_resize_ms, phase);
#endif
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int p = chunk & 1;
        const int row_begin = chunk * staging_rows;
        const size_t rows = std::min(staging_rows, n_mels - row_begin);
        const size_t values = rows * n_frames;
        PROFILE_POINT(phase);
        cuda_check(cudaEventSynchronize(copies[p].get()), "cudaEventSynchronize D2H");
        PROFILE_ADD(d2h_ms, phase);
        PROFILE_POINT(phase);
        const Real* source = p == 0 ? staging0.get() : staging1.get();
        double* destination = output.data() + static_cast<size_t>(row_begin) * n_frames;
        if constexpr (std::is_same_v<Real, double>) {
            std::copy(source, source + values, destination);
        } else {
            for (size_t i = 0; i < values; ++i)
                destination[i] = static_cast<double>(source[i]);
        }
        PROFILE_ADD(host_convert_ms, phase);
        if (chunk + 2 < chunks) enqueue(chunk + 2);
    }
    cuda_check(cudaStreamSynchronize(streams[0].get()), "cudaStreamSynchronize output");
    cuda_check(cudaStreamSynchronize(streams[1].get()), "cudaStreamSynchronize output");
    PROFILE_ADD(async_output_total_ms, async_start);
#else
#if MELSPEC_CUDA_RESIZE_THREAD
    PROFILE_POINT(phase);
    output_resize.finish();
    PROFILE_ADD(resize_wait_ms, phase);
#elif MELSPEC_CUDA_DEFER_RESIZE
    PROFILE_POINT(phase);
    output.resize(output_elements);
    PROFILE_ADD(output_resize_ms, phase);
#endif
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
#endif
    PROFILE_PRINT("cuda_profile precision=%d batch=%d fused=%d defer_resize=%d async_output=%d resize_thread=%d staging_rows=%d y_convert_ms=%.3f window_fb_ms=%.3f output_resize_ms=%.3f resize_wait_ms=%.3f device_alloc_ms=%.3f device_alloc_wait_ms=%.3f h2d_ms=%.3f gpu_input_convert_ms=%.3f batch_alloc_ms=%.3f plan_ms=%.3f gather_ms=%.3f fft_ms=%.3f mel_ms=%.3f pipeline_sync_ms=%.3f staging_alloc_ms=%.3f d2h_ms=%.3f host_convert_ms=%.3f async_output_total_ms=%.3f compute_total_ms=%.3f\n",
                  MELSPEC_CUDA_PRECISION, MELSPEC_CUDA_BATCH, MELSPEC_CUDA_FUSED,
                  MELSPEC_CUDA_DEFER_RESIZE, MELSPEC_CUDA_ASYNC_OUTPUT,
                  MELSPEC_CUDA_RESIZE_THREAD, MELSPEC_CUDA_STAGING_ROWS,
                  y_convert_ms, window_fb_ms, output_resize_ms, resize_wait_ms,
                  device_alloc_ms, device_alloc_wait_ms, h2d_ms, gpu_input_convert_ms,
                  batch_alloc_ms, plan_ms, gather_ms, fft_ms,
                  mel_ms, pipeline_sync_ms, staging_alloc_ms, d2h_ms, host_convert_ms,
                  async_output_total_ms, profile_elapsed(total_start));
}
