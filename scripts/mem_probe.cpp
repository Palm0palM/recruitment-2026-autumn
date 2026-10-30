// G3 memory-path probe: pageable vs pinned H2D, allocation and conversion cost.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t err_ = (expr);                                              \
        if (err_ != hipSuccess) {                                              \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                    \
                         hipGetErrorString(err_));                             \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static const char* ptr_kind(const void* p) {
    hipPointerAttribute_t attr{};
    if (hipPointerGetAttributes(&attr, p) != hipSuccess) return "unknown";
    switch (attr.type) {
        case hipMemoryTypeHost: return "host(pinned)";
        case hipMemoryTypeDevice: return "device";
        case hipMemoryTypeManaged: return "managed";
        default: return "other";
    }
}

static void bench_copy(const char* label, void* src, size_t bytes,
                       void* dst_device, int reps) {
    std::vector<double> times;
    for (int i = 0; i < reps; ++i) {
        auto t = Clock::now();
        HIP_CHECK(hipMemcpy(dst_device, src, bytes, hipMemcpyHostToDevice));
        times.push_back(ms_since(t));
    }
    std::sort(times.begin(), times.end());
    const double best = times.front();
    std::printf("%-22s bytes=%zu min_ms=%.3f median_ms=%.3f GB/s=%.2f\n",
                label, bytes, best, times[times.size() / 2],
                bytes / best / 1e6);
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s wavefront=%d\n", prop.name, prop.gcnArchName,
                prop.warpSize);

    const size_t samples = 6158720;  // official audio.pcm as double
    const size_t bytes_f32 = samples * sizeof(float);
    std::printf("samples=%zu float_bytes=%zu\n", samples, bytes_f32);

    // Source vector mimics main.cpp: vector<double> already in heap.
    auto t = Clock::now();
    std::vector<double> y(samples);
    for (size_t i = 0; i < samples; ++i) y[i] = (i % 1000) * 1e-4;
    std::printf("host_fill_double_ms=%.3f\n", ms_since(t));

    t = Clock::now();
    std::vector<float> pageable(samples);
    std::transform(y.begin(), y.end(), pageable.begin(),
                   [](double v) { return static_cast<float>(v); });
    std::printf("convert_pageable_ms=%.3f (%.2f GB/s effective)\n", ms_since(t),
                2.0 * bytes_f32 / ms_since(t) / 1e6);

    t = Clock::now();
    float* pinned = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&pinned), bytes_f32,
                            hipHostMallocDefault));
    const double alloc_ms = ms_since(t);
    std::printf("hipHostMalloc_ms=%.3f kind=%s\n", alloc_ms, ptr_kind(pinned));

    t = Clock::now();
    std::transform(y.begin(), y.end(), pinned,
                   [](double v) { return static_cast<float>(v); });
    std::printf("convert_pinned_ms=%.3f\n", ms_since(t));

    t = Clock::now();
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&pinned), bytes_f32,
                            hipHostMallocDefault));
    std::printf("hipHostMalloc_second_ms=%.3f\n", ms_since(t));

    std::vector<double> convert_times;
    for (int i = 0; i < 5; ++i) {
        t = Clock::now();
        std::transform(y.begin(), y.end(), pinned,
                       [](double v) { return static_cast<float>(v); });
        convert_times.push_back(ms_since(t));
    }
    std::sort(convert_times.begin(), convert_times.end());
    std::printf("convert_pinned_repeat_min_ms=%.3f median_ms=%.3f\n",
                convert_times.front(), convert_times[convert_times.size() / 2]);

    float* device = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device), bytes_f32));
    std::printf("pageable_kind=%s\n", ptr_kind(pageable.data()));
    bench_copy("h2d_pageable", pageable.data(), bytes_f32, device, 7);
    bench_copy("h2d_pinned", pinned, bytes_f32, device, 7);

    // Async on a non-default stream, pageable vs pinned.
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto async_copy = [&](const char* label, void* src) {
        std::vector<double> times;
        for (int i = 0; i < 7; ++i) {
            auto s = Clock::now();
            HIP_CHECK(hipMemcpyAsync(device, src, bytes_f32,
                                     hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            times.push_back(ms_since(s));
        }
        std::sort(times.begin(), times.end());
        std::printf("%-22s min_ms=%.3f GB/s=%.2f\n", label, times.front(),
                    bytes_f32 / times.front() / 1e6);
    };
    async_copy("h2d_async_pageable", pageable.data());
    async_copy("h2d_async_pinned", pinned);

    // Allocation cost of the real pipeline's device buffers.
    const size_t device_total = bytes_f32 + 16384 * sizeof(float) +
                                256 * 4097 * sizeof(float) + 256 * sizeof(float);
    t = Clock::now();
    float* big = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&big), device_total));
    std::printf("hipMalloc_pipeline_ms=%.3f bytes=%zu\n", ms_since(t), device_total);
    t = Clock::now();
    HIP_CHECK(hipMemset(big, 0, device_total));
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("hipMemset_first_touch_ms=%.3f\n", ms_since(t));

    HIP_CHECK(hipFree(big));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(device));
    HIP_CHECK(hipHostFree(pinned));
    return 0;
}
