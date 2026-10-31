#include "melspec_hip.h"
#include "melspec.h"
#include "melspec_backend.h"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef MELSPEC_HIP_PRECISION
#define MELSPEC_HIP_PRECISION 32
#endif
#ifndef MELSPEC_HIP_BATCH
#define MELSPEC_HIP_BATCH 512
#endif
#ifndef MELSPEC_HIP_BATCH_LARGE
#define MELSPEC_HIP_BATCH_LARGE 256
#endif
#ifndef MELSPEC_HIP_FUSED
#define MELSPEC_HIP_FUSED 1
#endif
#ifndef MELSPEC_HIP_PINNED
#define MELSPEC_HIP_PINNED 0
#endif
#ifndef MELSPEC_HIP_PINNED_INPUT
#define MELSPEC_HIP_PINNED_INPUT 0
#endif
#ifndef MELSPEC_HIP_ASYNC_H2D
#define MELSPEC_HIP_ASYNC_H2D 1
#endif
#ifndef MELSPEC_HIP_PREWARM
#define MELSPEC_HIP_PREWARM 1
#endif
#ifndef MELSPEC_HIP_EAGER_PREWARM
#define MELSPEC_HIP_EAGER_PREWARM 0
#endif
#ifndef MELSPEC_HIP_DIRECT_OUTPUT
#define MELSPEC_HIP_DIRECT_OUTPUT 1
#endif
#ifndef MELSPEC_HIP_HUGEPAGE
#define MELSPEC_HIP_HUGEPAGE 1
#endif
#ifndef MELSPEC_HIP_OVERLAP_INIT
#define MELSPEC_HIP_OVERLAP_INIT 0
#endif
#ifndef MELSPEC_HIP_DEFER_RESIZE
#define MELSPEC_HIP_DEFER_RESIZE 0
#endif
#ifndef MELSPEC_HIP_ASYNC_OUTPUT
#define MELSPEC_HIP_ASYNC_OUTPUT 1
#endif
#ifndef MELSPEC_HIP_RESIZE_THREAD
#define MELSPEC_HIP_RESIZE_THREAD 1
#endif
#ifndef MELSPEC_HIP_STAGING_ROWS
#define MELSPEC_HIP_STAGING_ROWS 16
#endif

#if MELSPEC_HIP_OVERLAP_INIT || MELSPEC_HIP_RESIZE_THREAD || MELSPEC_HIP_PREWARM
#include <exception>
#include <thread>
#endif
#if MELSPEC_HIP_HUGEPAGE && defined(__linux__)
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#endif

// Optional page-allocation hint only: every output double is still produced by
// the normal resize. Align the range so allocator metadata is never advised.
static void advise_huge_pages(void* data, size_t bytes) {
#if MELSPEC_HIP_HUGEPAGE && defined(__linux__) && defined(MADV_HUGEPAGE)
    const long system_page = sysconf(_SC_PAGESIZE);
    if (data == nullptr || system_page <= 0) return;
    const uintptr_t page = static_cast<uintptr_t>(system_page);
    const uintptr_t base = reinterpret_cast<uintptr_t>(data);
    const uintptr_t begin = base + (page - base % page) % page;
    const uintptr_t end_address = base + bytes;
    const uintptr_t end = end_address - end_address % page;
    if (end > begin)
        (void)madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
#else
    (void)data;
    (void)bytes;
#endif
}

static void resize_output_with_hint(std::vector<double>& output, size_t elements) {
    output.reserve(elements);
    advise_huge_pages(output.data(), elements * sizeof(double));
    output.resize(elements);
}

#if MELSPEC_HIP_RESIZE_THREAD
struct OutputResize {
    std::exception_ptr error;
    std::thread thread;
    OutputResize(std::vector<double>& output, size_t elements) : thread([this, &output, elements] {
        try {
            output.reserve(elements);
            advise_huge_pages(output.data(), elements * sizeof(double));
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
static void check_hip_initialization(hipError_t status) {
    if (status == hipErrorNoDevice || status == hipErrorInsufficientDriver)
        throw BackendUnavailable(std::string("HIP device unavailable: ") + hipGetErrorString(status));
    if (status != hipSuccess)
        throw std::runtime_error(std::string("HIP initialization: ") + hipGetErrorString(status));
}

#if MELSPEC_HIP_OVERLAP_INIT
struct ContextInit {
    std::exception_ptr error;  // initialize before the worker may write it
    std::thread thread;
    ContextInit() : thread([this] {
        try {
            const hipError_t status = hipInit(0);
            check_hip_initialization(status);
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

#ifdef MELSPEC_HIP_PROFILE
#include <chrono>
#include <cstdio>
using ProfileClock = std::chrono::steady_clock;
double profile_elapsed(ProfileClock::time_point start) {
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}
#define PROFILE_POINT(name) name = ProfileClock::now()
#define PROFILE_ADD(variable, point) variable += profile_elapsed(point); point = ProfileClock::now()
#define PROFILE_ADD_SYNC(variable, point) \
    hip_check(hipDeviceSynchronize(), "HIP profile synchronize"); PROFILE_ADD(variable, point)
#define PROFILE_PRINT(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define PROFILE_POINT(name)
#define PROFILE_ADD(variable, point)
#define PROFILE_ADD_SYNC(variable, point)
#define PROFILE_PRINT(...)
#endif

namespace {

using Real = std::conditional_t<MELSPEC_HIP_PRECISION == 32, float, double>;
static_assert(MELSPEC_HIP_BATCH > 0, "HIP_BATCH must be positive");
static_assert(MELSPEC_HIP_STAGING_ROWS > 0, "HIP_STAGING_ROWS must be positive");
static_assert(!(MELSPEC_HIP_PINNED && MELSPEC_HIP_ASYNC_OUTPUT),
              "Use either full pinned output or async chunked output, not both");
static_assert(!(MELSPEC_HIP_DIRECT_OUTPUT && MELSPEC_HIP_PINNED),
              "Direct double output already streams into the output vector");
static_assert(!(MELSPEC_HIP_RESIZE_THREAD && MELSPEC_HIP_DEFER_RESIZE),
              "Use either resize thread or deferred resize, not both");

void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

void hipfft_check(hipfftResult status, const char* operation) {
    if (status != HIPFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": hipFFT error " +
                                 std::to_string(static_cast<int>(status)));
}

template <typename T> struct FftTraits;
template <> struct FftTraits<float> {
    using Complex = hipfftComplex;
    static constexpr hipfftType type = HIPFFT_R2C;
    static hipfftResult execute(hipfftHandle plan, float* input, Complex* output) {
        return hipfftExecR2C(plan, input, output);
    }
};
template <> struct FftTraits<double> {
    using Complex = hipfftDoubleComplex;
    static constexpr hipfftType type = HIPFFT_D2Z;
    static hipfftResult execute(hipfftHandle plan, double* input, Complex* output) {
        return hipfftExecD2Z(plan, input, output);
    }
};
using Complex = typename FftTraits<Real>::Complex;
#if MELSPEC_HIP_DIRECT_OUTPUT
using OutReal = double;
#else
using OutReal = Real;
#endif

__device__ __forceinline__ float squared_norm(const hipfftComplex& z) {
    return z.x * z.x + z.y * z.y;
}
__device__ __forceinline__ double squared_norm(const hipfftDoubleComplex& z) {
    return z.x * z.x + z.y * z.y;
}

template <typename T>
__global__ void gather_window_kernel(const T* __restrict__ y,
                                     const T* __restrict__ window,
                                     T* __restrict__ fft_input, int batch,
                                     int valid, size_t frame_begin, int n_fft,
                                     int hop_length) {
    const size_t count = static_cast<size_t>(batch) * n_fft;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const size_t frame = i / static_cast<unsigned>(n_fft);
        const size_t n = i - frame * n_fft;
        fft_input[i] = frame < static_cast<size_t>(valid)
            ? y[(frame_begin + frame) * hop_length + n] * window[n]
            : static_cast<T>(0);
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
        value += __shfl_down(value, offset, warpSize);
    return value;
}

template <typename T, typename Out>
__global__ void mel_kernel(const typename FftTraits<T>::Complex* __restrict__ fft_output,
                           const T* __restrict__ power,
                           const T* __restrict__ mel_weights,
                           const int* __restrict__ first,
                           const int* __restrict__ last, Out* __restrict__ output,
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
            output[static_cast<size_t>(m) * n_frames + frame_begin + frame] =
                static_cast<Out>(acc);
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
            hip_check(hipMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
                       "hipMalloc");
    }
    ~DeviceBuffer() { (void)hipFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
    operator bool() const { return pointer_ != nullptr; }
};

#if MELSPEC_HIP_PINNED || MELSPEC_HIP_PINNED_INPUT || MELSPEC_HIP_ASYNC_OUTPUT
template <typename T> class PinnedBuffer {
    T* pointer_ = nullptr;
public:
    explicit PinnedBuffer(size_t count) {
        if (count)
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&pointer_),
                                     count * sizeof(T), hipHostMallocDefault),
                       "hipHostMalloc");
    }
    ~PinnedBuffer() { (void)hipHostFree(pointer_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    T* get() const { return pointer_; }
    T& operator[](size_t i) const { return pointer_[i]; }
};
#endif

#if MELSPEC_HIP_ASYNC_OUTPUT
class HipStream {
    hipStream_t stream_ = nullptr;
public:
    HipStream() {
        hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                   "hipStreamCreateWithFlags");
    }
    ~HipStream() { (void)hipStreamDestroy(stream_); }
    HipStream(const HipStream&) = delete;
    HipStream& operator=(const HipStream&) = delete;
    hipStream_t get() const { return stream_; }
};

class HipEvent {
    hipEvent_t event_ = nullptr;
public:
    HipEvent() { hip_check(hipEventCreate(&event_), "hipEventCreate"); }
    ~HipEvent() { (void)hipEventDestroy(event_); }
    HipEvent(const HipEvent&) = delete;
    HipEvent& operator=(const HipEvent&) = delete;
    hipEvent_t get() const { return event_; }
};
#endif

class HipfftPlan {
    hipfftHandle handle_ = 0;
public:
    HipfftPlan(int n_fft, int batch) {
        const int bins = n_fft / 2 + 1;
        hipfft_check(hipfftPlanMany(&handle_, 1, &n_fft, nullptr, 1, n_fft,
                                  nullptr, 1, bins, FftTraits<Real>::type, batch),
                    "hipfftPlanMany");
    }
    ~HipfftPlan() { hipfftDestroy(handle_); }
    HipfftPlan(const HipfftPlan&) = delete;
    HipfftPlan& operator=(const HipfftPlan&) = delete;
    hipfftHandle get() const { return handle_; }
    size_t workspace() const {
        size_t bytes = 0;
        hipfft_check(hipfftGetSize(handle_, &bytes), "hipfftGetSize");
        return bytes;
    }
};

hipStream_t& input_stream() {
    static hipStream_t stream = [] {
        hipStream_t created = nullptr;
        hip_check(hipStreamCreateWithFlags(&created, hipStreamNonBlocking),
                  "hipStreamCreate H2D");
        return created;
    }();
    return stream;
}

// The synchronous hipMemcpy/hipMemset path pays a one-time ~190 ms runtime cost
// on this rootless ROCm install; the async path does not. Keep transfers async
// and synchronize once after the whole group.
void copy_to_device(void* destination, const void* source, size_t bytes) {
    if (!bytes) return;
#if MELSPEC_HIP_ASYNC_H2D
    hip_check(hipMemcpyAsync(destination, source, bytes, hipMemcpyHostToDevice,
                             input_stream()),
              "hipMemcpyAsync H2D");
#else
    hip_check(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice),
              "hipMemcpy H2D");
#endif
}

void input_stream_synchronize() {
#if MELSPEC_HIP_ASYNC_H2D
    hip_check(hipStreamSynchronize(input_stream()), "hipStreamSynchronize H2D");
#endif
}

#if MELSPEC_HIP_PREWARM
// The first HIP queue creation in a process costs ~190-230 ms on this install.
// Dedicated builds start at load time to overlap PCM read/normalisation.
// CPU+HIP starts only after selecting HIP, overlapping backend host setup.
struct HipPrewarm {
    std::exception_ptr error; // initialize before the worker can publish it
    std::thread thread;
    HipPrewarm() : thread([this] {
        try {
            check_hip_initialization(hipInit(0));
            (void)input_stream();
            DeviceBuffer<unsigned char> scratch(4096);
            hip_check(hipMemsetAsync(scratch.get(), 0, 4096, input_stream()), "HIP prewarm memset");
            hip_check(hipStreamSynchronize(input_stream()), "HIP prewarm synchronize");
        } catch (...) {
            error = std::current_exception();
        }
    }) {}
    ~HipPrewarm() {
        if (thread.joinable()) thread.join();
    }
    void finish() {
        if (thread.joinable()) thread.join();
        if (error) std::rethrow_exception(error);
    }
};
HipPrewarm& hip_prewarm() {
    static HipPrewarm prewarm;
    return prewarm;
}
#if MELSPEC_HIP_EAGER_PREWARM
// Only AMD-dedicated builds may prewarm before main. Mixed CPU+HIP builds
// instead start this SAME queue warmup after the dispatcher selected HIP.
struct EagerHipPrewarm {
    EagerHipPrewarm() {
        const char* mode = std::getenv("MELSPEC_BACKEND");
        if (!mode || !*mode || std::strcmp(mode, "auto") == 0 || std::strcmp(mode, "hip") == 0)
            (void)hip_prewarm();
    }
} eager_hip_prewarm;
#endif
#endif

}  // namespace

void compute_melspectrogram_hip(const std::vector<double>& y, int sr, int n_fft,
                                 int hop_length, int n_mels, double f_min,
                                 double f_max, std::vector<double>& output) {
#ifdef MELSPEC_HIP_PROFILE
    ProfileClock::time_point total_start, phase, async_start;
    double device_alloc_ms = 0, h2d_ms = 0, batch_alloc_ms = 0;
    double h2d_y_ms = 0, h2d_aux_ms = 0, prewarm_wait_ms = 0;
    double plan_ms = 0, gather_ms = 0, fft_ms = 0, mel_ms = 0, pipeline_sync_ms = 0;
    double d2h_ms = 0, host_convert_ms = 0;
    double y_convert_ms = 0, window_fb_ms = 0, output_resize_ms = 0, staging_alloc_ms = 0;
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
    const size_t output_elements = static_cast<size_t>(n_mels) * n_frames;
    if (n_fft == 1) {
        resize_output_with_hint(output, output_elements);
        std::fill(output.begin(), output.end(), 0.0);
        return;
    }
#if MELSPEC_HIP_PREWARM
    (void)hip_prewarm();
#endif
#if MELSPEC_HIP_RESIZE_THREAD
    OutputResize output_resize(output, output_elements);
#elif !MELSPEC_HIP_DEFER_RESIZE
    PROFILE_POINT(phase);
    resize_output_with_hint(output, output_elements);
    PROFILE_ADD(output_resize_ms, phase);
#endif
#if MELSPEC_HIP_OVERLAP_INIT
    ContextInit context_init;
#endif

    PROFILE_POINT(phase);
#if MELSPEC_HIP_PINNED || MELSPEC_HIP_PINNED_INPUT
    PinnedBuffer<Real> host_y(y.size());
    std::copy(y.begin(), y.end(), host_y.get());
#else
    std::vector<Real> host_y;
    host_y.reserve(y.size());
    advise_huge_pages(host_y.data(), y.size() * sizeof(Real));
    host_y.resize(y.size());
    std::transform(y.begin(), y.end(), host_y.begin(),
                   [](double v) { return static_cast<Real>(v); });
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
#if MELSPEC_HIP_OVERLAP_INIT
    context_init.finish();
#endif

    DeviceBuffer<Real> device_y(y.size());
    DeviceBuffer<Real> device_window(window.size());
    DeviceBuffer<Real> device_weights(weights.size());
    DeviceBuffer<int> device_first(first.size());
    DeviceBuffer<int> device_last(last.size());
    DeviceBuffer<OutReal> device_output(output_elements);
    PROFILE_ADD(device_alloc_ms, phase);
#if MELSPEC_HIP_PREWARM
    hip_prewarm().finish();
    PROFILE_ADD(prewarm_wait_ms, phase);
#endif
#if MELSPEC_HIP_PINNED || MELSPEC_HIP_PINNED_INPUT
    copy_to_device(device_y.get(), host_y.get(), y.size() * sizeof(Real));
#else
    copy_to_device(device_y.get(), host_y.data(), y.size() * sizeof(Real));
#endif
    PROFILE_ADD(h2d_y_ms, phase);
    copy_to_device(device_window.get(), window.data(), window.size() * sizeof(Real));
    copy_to_device(device_weights.get(), weights.data(), weights.size() * sizeof(Real));
    copy_to_device(device_first.get(), first.data(), first.size() * sizeof(int));
    copy_to_device(device_last.get(), last.data(), last.size() * sizeof(int));
    input_stream_synchronize();
    PROFILE_ADD(h2d_aux_ms, phase);
#ifdef MELSPEC_HIP_PROFILE
    h2d_ms = h2d_y_ms + h2d_aux_ms;
#endif

    const int batch_limit = n_fft >= 16384
        ? std::min(MELSPEC_HIP_BATCH, MELSPEC_HIP_BATCH_LARGE)
        : MELSPEC_HIP_BATCH;
    const int capacity = std::min(batch_limit, n_frames);
    DeviceBuffer<Real> fft_input(static_cast<size_t>(capacity) * n_fft);
    DeviceBuffer<Complex> fft_output(static_cast<size_t>(capacity) * n_bins);
#if !MELSPEC_HIP_FUSED
    DeviceBuffer<Real> power(static_cast<size_t>(capacity) * n_bins);
#endif
    PROFILE_ADD(batch_alloc_ms, phase);
    HipfftPlan full_plan(n_fft, capacity);
    PROFILE_ADD(plan_ms, phase);
    PROFILE_PRINT("hipfft_workspace full=%zu bytes\n", full_plan.workspace());
#ifdef MELSPEC_HIP_PROFILE
    size_t free_bytes = 0, total_bytes = 0;
    hip_check(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");
    PROFILE_PRINT("hip_memory free=%zu total=%zu used=%zu bytes\n",
                  free_bytes, total_bytes, total_bytes - free_bytes);
#endif

    for (int frame_begin = 0; frame_begin < n_frames; frame_begin += capacity) {
        const int batch = std::min(capacity, n_frames - frame_begin);
        PROFILE_POINT(phase);
        gather_window_kernel<<<blocks_for(static_cast<size_t>(capacity) * n_fft), 256>>>(
            device_y.get(), device_window.get(), fft_input.get(), capacity, batch,
            static_cast<size_t>(frame_begin), n_fft, hop_length);
        PROFILE_ADD_SYNC(gather_ms, phase);
        hipfft_check(FftTraits<Real>::execute(full_plan.get(), fft_input.get(),
                                              fft_output.get()),
                    "hipFFT execute");
        PROFILE_ADD_SYNC(fft_ms, phase);
#if MELSPEC_HIP_FUSED
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
    hip_check(hipGetLastError(), "HIP kernel launch");

#if MELSPEC_HIP_DIRECT_OUTPUT
    PROFILE_POINT(async_start);
#if MELSPEC_HIP_RESIZE_THREAD
    PROFILE_POINT(phase);
    output_resize.finish();
    PROFILE_ADD(resize_wait_ms, phase);
#elif MELSPEC_HIP_DEFER_RESIZE
    PROFILE_POINT(phase);
    resize_output_with_hint(output, output_elements);
    PROFILE_ADD(output_resize_ms, phase);
#endif
    PROFILE_POINT(phase);
    hip_check(hipMemcpy(output.data(), device_output.get(),
                        output_elements * sizeof(OutReal), hipMemcpyDeviceToHost),
              "hipMemcpy D2H direct output");
    PROFILE_ADD(d2h_ms, phase);
    PROFILE_ADD(async_output_total_ms, async_start);
#elif MELSPEC_HIP_ASYNC_OUTPUT
    PROFILE_POINT(async_start);
    HipEvent compute_done;
    hip_check(hipEventRecord(compute_done.get()), "hipEventRecord compute");
    const int staging_rows = std::min(MELSPEC_HIP_STAGING_ROWS, n_mels);
    const int chunks = (n_mels + staging_rows - 1) / staging_rows;
    const size_t staging_values = static_cast<size_t>(staging_rows) * n_frames;
    PROFILE_POINT(phase);
    PinnedBuffer<Real> staging0(staging_values);
    PinnedBuffer<Real> staging1(staging_values);
    HipStream streams[2];
    HipEvent copies[2];
    PROFILE_ADD(staging_alloc_ms, phase);
    hip_check(hipStreamWaitEvent(streams[0].get(), compute_done.get()),
              "hipStreamWaitEvent compute");
    hip_check(hipStreamWaitEvent(streams[1].get(), compute_done.get()),
              "hipStreamWaitEvent compute");
    auto enqueue = [&](int chunk) {
        const int p = chunk & 1;
        const int row_begin = chunk * staging_rows;
        const size_t rows = std::min(staging_rows, n_mels - row_begin);
        const size_t bytes = rows * n_frames * sizeof(Real);
        hip_check(hipMemcpyAsync((p == 0 ? staging0.get() : staging1.get()),
                                   device_output.get() + static_cast<size_t>(row_begin) * n_frames,
                                   bytes, hipMemcpyDeviceToHost, streams[p].get()),
                   "hipMemcpyAsync D2H output");
        hip_check(hipEventRecord(copies[p].get(), streams[p].get()),
                   "hipEventRecord D2H");
    };
    for (int i = 0; i < std::min(2, chunks); ++i) enqueue(i);
#if MELSPEC_HIP_RESIZE_THREAD
    PROFILE_POINT(phase);
    output_resize.finish();
    PROFILE_ADD(resize_wait_ms, phase);
#elif MELSPEC_HIP_DEFER_RESIZE
    PROFILE_POINT(phase);
    resize_output_with_hint(output, output_elements);
    PROFILE_ADD(output_resize_ms, phase);
#endif
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int p = chunk & 1;
        const int row_begin = chunk * staging_rows;
        const size_t rows = std::min(staging_rows, n_mels - row_begin);
        const size_t values = rows * n_frames;
        PROFILE_POINT(phase);
        hip_check(hipEventSynchronize(copies[p].get()), "hipEventSynchronize D2H");
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
    hip_check(hipStreamSynchronize(streams[0].get()), "hipStreamSynchronize output");
    hip_check(hipStreamSynchronize(streams[1].get()), "hipStreamSynchronize output");
    PROFILE_ADD(async_output_total_ms, async_start);
#else
#if MELSPEC_HIP_RESIZE_THREAD
    PROFILE_POINT(phase);
    output_resize.finish();
    PROFILE_ADD(resize_wait_ms, phase);
#elif MELSPEC_HIP_DEFER_RESIZE
    PROFILE_POINT(phase);
    resize_output_with_hint(output, output_elements);
    PROFILE_ADD(output_resize_ms, phase);
#endif
    hip_check(hipDeviceSynchronize(), "HIP pipeline synchronize");
    PROFILE_ADD_SYNC(pipeline_sync_ms, phase);
#if MELSPEC_HIP_PINNED
    PinnedBuffer<Real> host_output(output.size());
    hip_check(hipMemcpy(host_output.get(), device_output.get(),
                          output.size() * sizeof(Real), hipMemcpyDeviceToHost),
               "hipMemcpy D2H output");
    PROFILE_ADD(d2h_ms, phase);
    std::copy(host_output.get(), host_output.get() + output.size(), output.begin());
    PROFILE_ADD(host_convert_ms, phase);
#else
    if constexpr (std::is_same_v<Real, double>) {
        hip_check(hipMemcpy(output.data(), device_output.get(),
                              output.size() * sizeof(Real), hipMemcpyDeviceToHost),
                   "hipMemcpy D2H output");
        PROFILE_ADD(d2h_ms, phase);
    } else {
        std::vector<Real> host_output(output.size());
        hip_check(hipMemcpy(host_output.data(), device_output.get(),
                              output.size() * sizeof(Real), hipMemcpyDeviceToHost),
                   "hipMemcpy D2H output");
        PROFILE_ADD(d2h_ms, phase);
        std::copy(host_output.begin(), host_output.end(), output.begin());
        PROFILE_ADD(host_convert_ms, phase);
    }
#endif
#endif
    PROFILE_PRINT("hip_profile precision=%d batch=%d fused=%d defer_resize=%d async_output=%d resize_thread=%d async_h2d=%d prewarm_wait_ms=%.3f staging_rows=%d y_convert_ms=%.3f h2d_y_ms=%.3f h2d_aux_ms=%.3f window_fb_ms=%.3f output_resize_ms=%.3f resize_wait_ms=%.3f device_alloc_ms=%.3f h2d_ms=%.3f batch_alloc_ms=%.3f plan_ms=%.3f gather_ms=%.3f fft_ms=%.3f mel_ms=%.3f pipeline_sync_ms=%.3f staging_alloc_ms=%.3f d2h_ms=%.3f host_convert_ms=%.3f async_output_total_ms=%.3f compute_total_ms=%.3f\n",
                  MELSPEC_HIP_PRECISION, MELSPEC_HIP_BATCH, MELSPEC_HIP_FUSED,
                  MELSPEC_HIP_DEFER_RESIZE, MELSPEC_HIP_ASYNC_OUTPUT,
                  MELSPEC_HIP_RESIZE_THREAD, MELSPEC_HIP_STAGING_ROWS,
                  MELSPEC_HIP_ASYNC_H2D, prewarm_wait_ms, y_convert_ms, h2d_y_ms, h2d_aux_ms, window_fb_ms, output_resize_ms, resize_wait_ms,
                  device_alloc_ms, h2d_ms, batch_alloc_ms, plan_ms, gather_ms, fft_ms,
                  mel_ms, pipeline_sync_ms, staging_alloc_ms, d2h_ms, host_convert_ms,
                  async_output_total_ms, profile_elapsed(total_start));
}

bool hip_backend_available() {
    int count = 0;
    const hipError_t status = hipGetDeviceCount(&count);
    if (status == hipSuccess) return count > 0;
    if (status == hipErrorNoDevice || status == hipErrorInsufficientDriver) return false;
    throw std::runtime_error(std::string("hipGetDeviceCount: ") + hipGetErrorString(status));
}
