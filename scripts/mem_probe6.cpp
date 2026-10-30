// Is the ~190 ms first H2D a per-process/power-state wake-up that can be hidden?
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
    const int mode = argc > 1 ? std::atoi(argv[1]) : 0;
    auto t_start = Clock::now();
    hipStream_t s{};
    HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    float* a = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&a), kIn * sizeof(float)));
    float* b = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&b), kOut * sizeof(float)));
    std::printf("mode=%d setup_ms=%.3f\n", mode, ms_since(t_start));

    if (mode == 1 || mode == 2) {
        // Wake the GPU asynchronously, then do host work like the CLI does.
        auto t = Clock::now();
        HIP_CHECK(hipMemsetAsync(a, 0, sizeof(float), s));
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("mode=%d wake_sync_ms=%.3f\n", mode, ms_since(t));
    }

    // Host work similar to reading/parsing the PCM file (do not optimize away).
    auto t = Clock::now();
    std::vector<double> y(kIn, 0.25);
    volatile double sink = 0;
    for (size_t i = 0; i < kIn; ++i) sink += y[i];
    std::printf("host_work_ms=%.3f sink=%.3f\n", ms_since(t), sink);

    if (mode == 2) {
        // Extra sleep-equivalent host work: touch another large buffer.
        t = Clock::now();
        std::vector<double> pad(kOut, 0.5);
        for (size_t i = 0; i < kOut; i += 4096) pad[i] += 1.0;
        std::printf("extra_host_work_ms=%.3f\n", ms_since(t));
    }

    std::vector<float> host(kIn, 1.0f);
    t = Clock::now();
    HIP_CHECK(hipMemcpyAsync(a, host.data(), kIn * sizeof(float),
                             hipMemcpyHostToDevice, s));
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("mode=%d first_h2d_ms=%.3f\n", mode, ms_since(t));

    t = Clock::now();
    HIP_CHECK(hipMemcpyAsync(a, host.data(), kIn * sizeof(float),
                             hipMemcpyHostToDevice, s));
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("mode=%d second_h2d_ms=%.3f\n", mode, ms_since(t));
    std::printf("mode=%d total_ms=%.3f\n", mode, ms_since(t_start));
    return 0;
}
