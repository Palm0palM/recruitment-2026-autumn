// Replicate the pipeline allocation order and time the first big H2D copy.
#include <hip/hip_runtime.h>

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

static void meminfo(const char* tag) {
    FILE* f = std::fopen("/proc/self/status", "r");
    char line[256];
    while (f && std::fgets(line, sizeof line, f)) {
        if (!std::strncmp(line, "VmRSS", 5) || !std::strncmp(line, "VmHWM", 5) ||
            !std::strncmp(line, "VmPeak", 6) || !std::strncmp(line, "VmSize", 6))
            std::printf("%s %s", tag, line);
    }
    if (f) std::fclose(f);
    f = std::fopen("/sys/fs/cgroup/memory.max", "r");
    if (f) {
        if (std::fgets(line, sizeof line, f)) std::printf("%s cgroup_memory.max=%s", tag, line);
        std::fclose(f);
    }
    f = std::fopen("/sys/fs/cgroup/memory.current", "r");
    if (f) {
        if (std::fgets(line, sizeof line, f)) std::printf("%s cgroup_memory.current=%s", tag, line);
        std::fclose(f);
    }
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s\n", prop.name, prop.gcnArchName);
    meminfo("start");

    const size_t samples = 24634880;  // int16 count of official audio.pcm
    std::vector<double> y(samples, 0.5);
    std::vector<float> host_y(samples, 0.5f);
    meminfo("after_host_buffers");

    // Exact pipeline allocation order.
    float* device_y = nullptr;
    float* device_window = nullptr;
    float* device_weights = nullptr;
    int* device_first = nullptr;
    int* device_last = nullptr;
    float* device_output = nullptr;
    auto t = Clock::now();
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_y), samples * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_window), 16384 * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_weights), 256 * 8193 * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_first), 256 * sizeof(int)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_last), 256 * sizeof(int)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), 256ull * 192397 * sizeof(float)));
    std::printf("alloc_all_ms=%.3f\n", ms_since(t));
    meminfo("after_device_alloc");

    for (int i = 0; i < 3; ++i) {
        t = Clock::now();
        HIP_CHECK(hipMemcpy(device_y, host_y.data(), samples * sizeof(float),
                            hipMemcpyHostToDevice));
        std::printf("copy_y_%d_ms=%.3f\n", i, ms_since(t));
    }

    // Does pre-touching the output buffer remove a first-touch penalty?
    t = Clock::now();
    HIP_CHECK(hipMemset(device_output, 0, 256ull * 192397 * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("memset_output_ms=%.3f\n", ms_since(t));
    t = Clock::now();
    HIP_CHECK(hipMemcpy(device_y, host_y.data(), samples * sizeof(float),
                        hipMemcpyHostToDevice));
    std::printf("copy_y_after_memset_ms=%.3f\n", ms_since(t));

    // 394 MB host resize, as the real output vector.
    t = Clock::now();
    std::vector<double> output(256ull * 192397);
    std::printf("resize_output_ms=%.3f\n", ms_since(t));
    meminfo("end");
    return 0;
}
