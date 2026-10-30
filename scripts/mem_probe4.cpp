// Bisect the one-time first-touch cost seen after the pipeline's allocations.
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

static const size_t kIn = 24634880;   // float count of y
static const size_t kOut = 256ull * 192397;  // output floats
static const size_t kWeights = 256ull * 8193;

int main(int argc, char** argv) {
    const int c = argc > 1 ? std::atoi(argv[1]) : 1;
    std::vector<float> host(kIn > kOut ? kIn : kOut, 1.0f);

    auto alloc = [](size_t n) {
        float* p = nullptr;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(float)));
        return p;
    };
    auto copy = [&](const char* tag, float* dst, size_t n) {
        auto t = Clock::now();
        HIP_CHECK(hipMemcpy(dst, host.data(), n * sizeof(float), hipMemcpyHostToDevice));
        std::printf("case=%d %-26s n=%zu ms=%.3f\n", c, tag, n, ms_since(t));
    };

    if (c == 1) {                       // small alloc only
        float* a = alloc(kIn);
        copy("y_only", a, kIn);
    } else if (c == 2) {                // y + big output allocated
        float* a = alloc(kIn);
        float* b = alloc(kOut);
        copy("y_after_output_alloc", a, kIn);
        copy("output", b, kOut);
    } else if (c == 3) {                // big output only
        float* b = alloc(kOut);
        copy("output_only", b, kOut);
    } else if (c == 4) {                // full pipeline set
        float* a = alloc(kIn);
        float* w = alloc(16384);
        float* m = alloc(kWeights);
        int* f = nullptr;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&f), 256 * sizeof(int)));
        float* b = alloc(kOut);
        (void)w; (void)m; (void)f;
        copy("y_full_set", a, kIn);
    } else if (c == 5) {                // pre-touch y by memset before copy
        float* a = alloc(kIn);
        auto t = Clock::now();
        HIP_CHECK(hipMemset(a, 0, kIn * sizeof(float)));
        std::printf("case=5 memset_y_ms=%.3f\n", ms_since(t));
        copy("y_after_memset", a, kIn);
    } else if (c == 6) {                // pre-touch big output by memset, then copy y
        float* a = alloc(kIn);
        float* b = alloc(kOut);
        auto t = Clock::now();
        HIP_CHECK(hipMemset(b, 0, kOut * sizeof(float)));
        HIP_CHECK(hipDeviceSynchronize());
        std::printf("case=6 memset_output_ms=%.3f\n", ms_since(t));
        copy("y_after_output_memset", a, kIn);
    } else if (c == 7) {                // async memset on stream, overlap with host work
        float* a = alloc(kIn);
        float* b = alloc(kOut);
        hipStream_t s{};
        HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        auto t = Clock::now();
        HIP_CHECK(hipMemsetAsync(b, 0, kOut * sizeof(float), s));
        std::printf("case=7 memset_async_enqueue_ms=%.3f\n", ms_since(t));
        // simulate host work overlapping the mapping
        auto host_start = Clock::now();
        std::vector<float> tmp(kIn);
        for (size_t i = 0; i < kIn; ++i) tmp[i] = host[i];
        std::printf("case=7 host_work_ms=%.3f\n", ms_since(host_start));
        t = Clock::now();
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("case=7 stream_sync_ms=%.3f\n", ms_since(t));
        copy("y_after_async_pretouch", a, kIn);
    }
    return 0;
}
