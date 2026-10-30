// FAILED PROBE (kept as a record): case 8 segfaults inside the ROCm runtime
// before the first printf, so the async pre-touch hypothesis was verified by
// probe4 case 7 and by the real backend instead. Do not reuse as-is.
// Does an async pre-touch remove the ~190 ms first synchronous H2D cost?
#include <hip/hip_runtime.h>

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

static const size_t kIn = 24634880;
static const size_t kOut = 256ull * 192397;

int main(int argc, char** argv) {
    const int c = argc > 1 ? std::atoi(argv[1]) : 8;
    std::vector<float> host(kIn, 1.0f);
    float* a = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&a), kIn * sizeof(float)));
    float* b = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&b), kOut * sizeof(float)));
    hipStream_t s{};
    HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    if (c == 8) {  // tiny async memset pre-touch, then sync copy
        auto t = Clock::now();
        HIP_CHECK(hipMemsetAsync(a, 0, sizeof(float), s));
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("case=8 tiny_async_pretouch_ms=%.3f\n", ms_since(t));
        t = Clock::now();
        HIP_CHECK(hipMemcpy(a, host.data(), kIn * sizeof(float), hipMemcpyHostToDevice));
        std::printf("case=8 sync_copy_y_ms=%.3f\n", ms_since(t));
        t = Clock::now();
        HIP_CHECK(hipMemcpy(b, host.data(), kOut * sizeof(float), hipMemcpyHostToDevice));
        std::printf("case=8 sync_copy_out_ms=%.3f\n", ms_since(t));
    } else if (c == 9) {  // async copy y, then sync
        auto t = Clock::now();
        HIP_CHECK(hipMemcpyAsync(a, host.data(), kIn * sizeof(float),
                                 hipMemcpyHostToDevice, s));
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("case=9 async_copy_y_plus_sync_ms=%.3f\n", ms_since(t));
        t = Clock::now();
        HIP_CHECK(hipMemcpy(a, host.data(), kIn * sizeof(float), hipMemcpyHostToDevice));
        std::printf("case=9 second_sync_copy_ms=%.3f\n", ms_since(t));
    } else if (c == 10) {  // sync copy first (baseline), then async out
        auto t = Clock::now();
        HIP_CHECK(hipMemcpy(a, host.data(), kIn * sizeof(float), hipMemcpyHostToDevice));
        std::printf("case=10 sync_copy_y_ms=%.3f\n", ms_since(t));
        t = Clock::now();
        HIP_CHECK(hipMemcpyAsync(b, host.data(), kOut * sizeof(float),
                                 hipMemcpyHostToDevice, s));
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("case=10 async_copy_out_plus_sync_ms=%.3f\n", ms_since(t));
    } else if (c == 11) {  // async copy y on a *blocking* stream, then sync
        hipStream_t b2{};
        HIP_CHECK(hipStreamCreate(&b2));
        auto t = Clock::now();
        HIP_CHECK(hipMemcpyAsync(a, host.data(), kIn * sizeof(float),
                                 hipMemcpyHostToDevice, b2));
        HIP_CHECK(hipStreamSynchronize(b2));
        std::printf("case=11 blocking_stream_async_ms=%.3f\n", ms_since(t));
        t = Clock::now();
        HIP_CHECK(hipMemcpy(a, host.data(), kIn * sizeof(float), hipMemcpyHostToDevice));
        std::printf("case=11 sync_copy_y_ms=%.3f\n", ms_since(t));
    }
    return 0;
}
