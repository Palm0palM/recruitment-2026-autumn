#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "CUDA smoke failure: %s\n", what);
        std::exit(1);
    }
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    check(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice");
    check(cudaGetDeviceProperties(&prop, device) == cudaSuccess, "cudaGetDeviceProperties");
    std::printf("device=%d name=%s sm=%d.%d memory=%zu MiB\n", device, prop.name,
                prop.major, prop.minor, prop.totalGlobalMem >> 20);

    constexpr int n = 8;
    const float input[n] = {1, 2, 3, 4, 5, 6, 7, 8};
    float* d_input = nullptr;
    cufftComplex* d_output = nullptr;
    check(cudaMalloc(&d_input, sizeof(input)) == cudaSuccess, "cudaMalloc input");
    check(cudaMalloc(&d_output, (n / 2 + 1) * sizeof(*d_output)) == cudaSuccess, "cudaMalloc output");
    check(cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice) == cudaSuccess, "H2D");
    cufftHandle plan;
    check(cufftPlan1d(&plan, n, CUFFT_R2C, 1) == CUFFT_SUCCESS, "cufftPlan1d R2C");
    check(cufftExecR2C(plan, d_input, d_output) == CUFFT_SUCCESS, "cufftExecR2C");
    cufftComplex output[n / 2 + 1];
    check(cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost) == cudaSuccess, "D2H");
    for (int k = 0; k <= n / 2; ++k) {
        double re = 0, im = 0;
        for (int j = 0; j < n; ++j) {
            const double angle = -2.0 * std::acos(-1.0) * k * j / n;
            re += input[j] * std::cos(angle);
            im += input[j] * std::sin(angle);
        }
        if (std::abs(output[k].x - re) > 1e-4 || std::abs(output[k].y - im) > 1e-4) {
            std::fprintf(stderr, "wrong bin %d: got %.7g%+.7gi expected %.7g%+.7gi\n",
                         k, output[k].x, output[k].y, re, im);
            return 1;
        }
    }
    check(cufftDestroy(plan) == CUFFT_SUCCESS, "cufftDestroy");
    check(cudaFree(d_output) == cudaSuccess, "cudaFree output");
    check(cudaFree(d_input) == cudaSuccess, "cudaFree input");
    std::puts("CUDA runtime + cuFFT R2C smoke: PASS");
    return 0;
}
