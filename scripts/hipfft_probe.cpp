// G1 feasibility probe for batched R2C on the Radeon AI PRO R9700 (gfx1201).
// Measures plan creation, first execution, and warm execution separately.
#include <hip/hip_runtime.h>
#include <hipfft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

#define HIPFFT_CHECK(expr)                                                     \
    do {                                                                       \
        hipfftResult err_ = (expr);                                            \
        if (err_ != HIPFFT_SUCCESS) {                                          \
            std::fprintf(stderr, "%s failed: hipFFT %d\n", #expr,             \
                         static_cast<int>(err_));                              \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static bool direct_r2c_check() {
    constexpr int n = 8;
    constexpr int bins = n / 2 + 1;
    const float input[n] = {1, 2, 3, 4, 5, 6, 7, 8};
    float* din = nullptr;
    hipfftComplex* dout = nullptr;
    HIP_CHECK(hipMalloc(&din, sizeof(input)));
    HIP_CHECK(hipMalloc(&dout, bins * sizeof(*dout)));
    HIP_CHECK(hipMemcpy(din, input, sizeof(input), hipMemcpyHostToDevice));
    hipfftHandle plan{};
    HIPFFT_CHECK(hipfftPlan1d(&plan, n, HIPFFT_R2C, 1));
    HIPFFT_CHECK(hipfftExecR2C(plan, din, dout));
    hipfftComplex got[bins];
    HIP_CHECK(hipMemcpy(got, dout, sizeof(got), hipMemcpyDeviceToHost));
    for (int k = 0; k < bins; ++k) {
        double re = 0, im = 0;
        for (int j = 0; j < n; ++j) {
            const double angle = -2.0 * std::acos(-1.0) * k * j / n;
            re += input[j] * std::cos(angle);
            im += input[j] * std::sin(angle);
        }
        if (std::abs(got[k].x - re) > 1e-4 ||
            std::abs(got[k].y - im) > 1e-4) {
            std::fprintf(stderr, "R2C self-test failed bin %d\n", k);
            return false;
        }
    }
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(dout));
    HIP_CHECK(hipFree(din));
    std::puts("r2c_self_test=PASS n=8");
    return true;
}

static void run_case(int n_fft, int batch, bool fp64, int warmups, int iterations) {
    const int bins = n_fft / 2 + 1;
    const size_t real_values =
        static_cast<size_t>(batch) * static_cast<size_t>(n_fft);
    const size_t complex_values =
        static_cast<size_t>(batch) * static_cast<size_t>(bins);
    const size_t real_bytes = real_values * (fp64 ? sizeof(double) : sizeof(float));
    const size_t complex_bytes = complex_values * (fp64 ? sizeof(hipfftDoubleComplex)
                                                        : sizeof(hipfftComplex));
    void* input = nullptr;
    void* output = nullptr;
    HIP_CHECK(hipMalloc(&input, real_bytes));
    HIP_CHECK(hipMalloc(&output, complex_bytes));
    hipfftHandle plan{};
    auto plan_start = Clock::now();
    if (fp64) {
        HIPFFT_CHECK(hipfftPlanMany(&plan, 1, &n_fft, nullptr, 1, n_fft,
                                    nullptr, 1, bins, HIPFFT_D2Z, batch));
    } else {
        HIPFFT_CHECK(hipfftPlanMany(&plan, 1, &n_fft, nullptr, 1, n_fft,
                                    nullptr, 1, bins, HIPFFT_R2C, batch));
    }
    HIP_CHECK(hipDeviceSynchronize());
    const double plan_ms = milliseconds(plan_start);
    size_t work_bytes = 0;
    const hipfftResult size_result = hipfftGetSize(plan, &work_bytes);
    if (size_result != HIPFFT_SUCCESS) work_bytes = 0;

    auto exec = [&]() {
        if (fp64) {
            HIPFFT_CHECK(hipfftExecD2Z(plan, static_cast<double*>(input),
                                       static_cast<hipfftDoubleComplex*>(output)));
        } else {
            HIPFFT_CHECK(hipfftExecR2C(plan, static_cast<float*>(input),
                                       static_cast<hipfftComplex*>(output)));
        }
    };
    HIP_CHECK(hipDeviceSynchronize());
    auto first_start = Clock::now();
    exec();
    HIP_CHECK(hipDeviceSynchronize());
    const double first_ms = milliseconds(first_start);
    for (int i = 0; i < warmups; ++i) exec();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        hipEvent_t begin{}, end{};
        HIP_CHECK(hipEventCreate(&begin));
        HIP_CHECK(hipEventCreate(&end));
        HIP_CHECK(hipEventRecord(begin));
        exec();
        HIP_CHECK(hipEventRecord(end));
        HIP_CHECK(hipEventSynchronize(end));
        float elapsed = 0;
        HIP_CHECK(hipEventElapsedTime(&elapsed, begin, end));
        times.push_back(elapsed);
        HIP_CHECK(hipEventDestroy(end));
        HIP_CHECK(hipEventDestroy(begin));
    }
    std::sort(times.begin(), times.end());
    const double min_ms = times.front();
    const double median_ms = times[times.size() / 2];
    const double max_ms = times.back();
    std::printf("precision=%s n_fft=%d batch=%d plan_ms=%.6f first_exec_ms=%.6f "
                "exec_min_ms=%.6f exec_median_ms=%.6f exec_max_ms=%.6f "
                "workspace_bytes=%zu real_bytes=%zu complex_bytes=%zu "
                "ns_per_fft=%.3f\n",
                fp64 ? "fp64" : "fp32", n_fft, batch, plan_ms, first_ms,
                min_ms, median_ms, max_ms, work_bytes, real_bytes, complex_bytes,
                min_ms * 1e6 / batch);
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(input));
}

int main() {
    int devices = 0;
    HIP_CHECK(hipGetDeviceCount(&devices));
    if (devices < 1) return 3;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s mem=%zu CUs=%d wavefront=%d\n",
                prop.name, prop.gcnArchName, prop.totalGlobalMem,
                prop.multiProcessorCount, prop.warpSize);
    if (!direct_r2c_check()) return 4;
    for (int n_fft : {8192, 16384}) {
        for (int batch : {1, 16, 64, 128, 256, 512, 1024}) {
            run_case(n_fft, batch, false, 3, 7);
        }
        run_case(n_fft, 256, true, 3, 5);
    }
    return 0;
}
