// Environment probe for the Radeon AI PRO R9700. Not part of the submission.
#include <hip/hip_runtime.h>
#include <hipfft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t err_ = (expr);                                              \
        if (err_ != hipSuccess) {                                              \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                    \
                         hipGetErrorString(err_));                             \
            return 2;                                                          \
        }                                                                      \
    } while (0)

#define HIPFFT_CHECK(expr)                                                     \
    do {                                                                       \
        hipfftResult err_ = (expr);                                            \
        if (err_ != HIPFFT_SUCCESS) {                                          \
            std::fprintf(stderr, "%s failed: hipFFT result %d\n", #expr,      \
                         static_cast<int>(err_));                             \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__global__ void add_one(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1.0f;
}

int main() {
    int count = 0, runtime = 0, driver = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    HIP_CHECK(hipRuntimeGetVersion(&runtime));
    HIP_CHECK(hipDriverGetVersion(&driver));
    std::printf("hip_devices=%d runtime=%d driver=%d\n", count, runtime, driver);
    if (count < 1) return 3;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("name=%s arch=%s mem=%zu CUs=%d wavefront=%d runtime_driver=%d\n",
                prop.name, prop.gcnArchName, prop.totalGlobalMem,
                prop.multiProcessorCount, prop.warpSize, driver);

    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);
    float* host = static_cast<float*>(std::malloc(bytes));
    if (!host) return 4;
    for (int i = 0; i < n; ++i) host[i] = static_cast<float>(i % 1000);
    float *din = nullptr, *dout = nullptr;
    HIP_CHECK(hipMalloc(&din, bytes));
    HIP_CHECK(hipMalloc(&dout, bytes));
    HIP_CHECK(hipMemcpy(din, host, bytes, hipMemcpyHostToDevice));
    add_one<<<(n + 255) / 256, 256>>>(din, dout, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host, dout, bytes, hipMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (host[i] != static_cast<float>(i % 1000) + 1.0f) {
            std::fprintf(stderr, "vector mismatch at %d: %f\n", i, host[i]);
            return 5;
        }
    }
    HIP_CHECK(hipFree(din));
    HIP_CHECK(hipFree(dout));
    std::free(host);
    std::puts("HIP vector-add: PASS");

    // Real-to-complex hipFFT, direct-DFT checked.
    constexpr int fft_n = 8;
    constexpr int bins = fft_n / 2 + 1;
    const float input[fft_n] = {1, 2, 3, 4, 5, 6, 7, 8};
    float* fft_in = nullptr;
    hipfftComplex* fft_out = nullptr;
    HIP_CHECK(hipMalloc(&fft_in, sizeof(input)));
    HIP_CHECK(hipMalloc(&fft_out, bins * sizeof(*fft_out)));
    HIP_CHECK(hipMemcpy(fft_in, input, sizeof(input), hipMemcpyHostToDevice));
    hipfftHandle plan{};
    HIPFFT_CHECK(hipfftPlan1d(&plan, fft_n, HIPFFT_R2C, 1));
    HIPFFT_CHECK(hipfftExecR2C(plan, fft_in, fft_out));
    hipfftComplex got[bins];
    HIP_CHECK(hipMemcpy(got, fft_out, sizeof(got), hipMemcpyDeviceToHost));
    for (int k = 0; k < bins; ++k) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < fft_n; ++j) {
            const double angle = -2.0 * std::acos(-1.0) * k * j / fft_n;
            re += input[j] * std::cos(angle);
            im += input[j] * std::sin(angle);
        }
        if (std::abs(got[k].x - re) > 1e-4 || std::abs(got[k].y - im) > 1e-4) {
            std::fprintf(stderr, "hipFFT mismatch bin %d: %.6g%+.6gi vs %.6g%+.6gi\n",
                         k, got[k].x, got[k].y, re, im);
            return 6;
        }
    }
    HIPFFT_CHECK(hipfftDestroy(plan));
    HIP_CHECK(hipFree(fft_out));
    HIP_CHECK(hipFree(fft_in));
    std::puts("hipFFT R2C: PASS");
    return 0;
}
