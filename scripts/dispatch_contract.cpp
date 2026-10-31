#include "melspec.h"
#include "melspec_backend.h"
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <vector>

static int cpu_calls = 0, gpu_calls = 0, probes = 0;
static bool available = true, fail_compute = false;

void compute_melspectrogram_cpu(const std::vector<double>&, int, int, int, int,
                                double, double, std::vector<double>& out) {
    ++cpu_calls;
    out.assign(1, 1.0);
}

#ifdef MELSPEC_WITH_HIP
#define GPU_ENTRY compute_melspectrogram_hip
#define GPU_AVAILABLE hip_backend_available
static const char* gpu_name = "hip";
#else
#define GPU_ENTRY compute_melspectrogram_cuda
#define GPU_AVAILABLE cuda_backend_available
static const char* gpu_name = "cuda";
#endif

bool GPU_AVAILABLE() { ++probes; return available; }
void GPU_ENTRY(const std::vector<double>&, int, int, int, int, double, double,
               std::vector<double>& out) {
    ++gpu_calls;
    if (fail_compute) throw std::runtime_error("injected real GPU error");
    out.assign(1, 2.0);
}

int main() {
    std::vector<double> small(64), large(1 << 20), out;
    auto call = [&](const auto& y, int n_fft) {
        compute_melspectrogram(y, 96000, n_fft, 1, 8, 0, 22050, out);
    };
    setenv("MELSPEC_BACKEND", "cpu", 1);
    call(large, 65536);
    assert(cpu_calls == 1 && probes == 0 && gpu_calls == 0);
    unsetenv("MELSPEC_BACKEND");
    call(small, 32);
    assert(cpu_calls == 2 && probes == 0);
    available = false;
    call(large, 65536);
    assert(cpu_calls == 3 && probes == 1);
    setenv("MELSPEC_BACKEND", gpu_name, 1);
    bool threw = false;
    try { call(small, 32); } catch (const std::runtime_error&) { threw = true; }
    assert(threw && cpu_calls == 3 && gpu_calls == 0);
    setenv("MELSPEC_BACKEND", "auto", 1);
    available = true;
    fail_compute = true;
    threw = false;
    try { call(large, 65536); } catch (const std::runtime_error&) { threw = true; }
    assert(threw && cpu_calls == 3 && gpu_calls == 1);
    setenv("MELSPEC_BACKEND", "invalid", 1);
    threw = false;
    try { call(small, 32); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    setenv("MELSPEC_BACKEND", "cpu", 1);
    threw = false;
    try {
        compute_melspectrogram(small, 96000, 32, 0, 8, 0, 22050, out);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
}
