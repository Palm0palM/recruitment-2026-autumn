// Follow-up: why does the 98.5 MB (float) input copy take ~190 ms in the pipeline?
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

static void report(const char* label, size_t bytes, std::vector<double> times) {
    std::sort(times.begin(), times.end());
    std::printf("%-34s bytes=%9zu min_ms=%7.3f first_ms=%7.3f GB/s=%.2f\n",
                label, bytes, times.front(), times[0],
                bytes / times.front() / 1e6);
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s\n", prop.name, prop.gcnArchName);

    const double sizes_mb[] = {24.6, 49.2, 98.5, 197.0};
    for (double mb : sizes_mb) {
        const size_t bytes = static_cast<size_t>(mb * 1e6);
        const size_t floats = bytes / sizeof(float);
        std::vector<float> host(floats, 1.0f);

        // Fresh device allocation for every copy (pipeline behaviour).
        std::vector<double> fresh;
        for (int i = 0; i < 3; ++i) {
            float* dst = nullptr;
            HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dst), bytes));
            auto t = Clock::now();
            HIP_CHECK(hipMemcpy(dst, host.data(), bytes, hipMemcpyHostToDevice));
            fresh.push_back(ms_since(t));
            HIP_CHECK(hipFree(dst));
        }
        char label[64];
        std::snprintf(label, sizeof label, "fresh_dst_pageable_%.0fMB", mb);
        report(label, bytes, fresh);

        // Reused destination.
        float* dst = nullptr;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dst), bytes));
        std::vector<double> reused;
        for (int i = 0; i < 5; ++i) {
            auto t = Clock::now();
            HIP_CHECK(hipMemcpy(dst, host.data(), bytes, hipMemcpyHostToDevice));
            reused.push_back(ms_since(t));
        }
        std::snprintf(label, sizeof label, "reused_dst_pageable_%.0fMB", mb);
        report(label, bytes, reused);

        float* pinned = nullptr;
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&pinned), bytes,
                                hipHostMallocDefault));
        std::copy(host.begin(), host.end(), pinned);
        std::vector<double> pin_times;
        for (int i = 0; i < 5; ++i) {
            auto t = Clock::now();
            HIP_CHECK(hipMemcpy(dst, pinned, bytes, hipMemcpyHostToDevice));
            pin_times.push_back(ms_since(t));
        }
        std::snprintf(label, sizeof label, "reused_dst_pinned_%.0fMB", mb);
        report(label, bytes, pin_times);

        HIP_CHECK(hipHostFree(pinned));
        HIP_CHECK(hipFree(dst));
    }

    // Device-side copy bandwidth for the largest size.
    {
        const size_t bytes = static_cast<size_t>(98.5e6);
        float* a = nullptr;
        float* b = nullptr;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&a), bytes));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&b), bytes));
        HIP_CHECK(hipMemset(a, 0, bytes));
        auto t = Clock::now();
        HIP_CHECK(hipMemcpy(b, a, bytes, hipMemcpyDeviceToDevice));
        std::printf("d2d_98MB_ms=%.3f GB/s=%.2f\n", ms_since(t),
                    bytes / ms_since(t) / 1e6);
        HIP_CHECK(hipFree(b));
        HIP_CHECK(hipFree(a));
    }
    return 0;
}
