// Minimal Radeon AI PRO R9700 HIP probe. Not part of the melspec submission.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t err = (expr);                                                \
        if (err != hipSuccess) {                                                \
            std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(err)); \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__global__ void add_one(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1.0f;
}

int main() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    std::printf("hip_devices=%d\n", count);
    if (count < 1) return 3;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("name=%s gcnArchName=%s global_mem=%zu multiprocessors=%d\n",
                prop.name, prop.gcnArchName, prop.totalGlobalMem, prop.multiProcessorCount);
    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);
    float *host = static_cast<float*>(std::malloc(bytes));
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
            std::fprintf(stderr, "mismatch at %d: %f\n", i, host[i]);
            return 5;
        }
    }
    HIP_CHECK(hipFree(din));
    HIP_CHECK(hipFree(dout));
    std::free(host);
    std::printf("HIP vector-add: PASS (%d elements)\n", n);
    return 0;
}
