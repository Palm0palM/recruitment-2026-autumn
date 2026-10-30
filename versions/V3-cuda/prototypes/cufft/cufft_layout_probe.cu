#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void cufftCheck(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 ": cuFFT error code " +
                                 std::to_string(static_cast<int>(status)));
    }
}

std::size_t checkedProduct(std::size_t a, std::size_t b,
                           const char* description) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(std::string("size overflow for ") + description);
    }
    return a * b;
}

class CufftPlan {
public:
    CufftPlan() = default;
    ~CufftPlan() {
        if (handle_ != 0) {
            (void)cufftDestroy(handle_);
        }
    }

    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;
    CufftPlan(CufftPlan&& other) noexcept : handle_(other.handle_) {
        other.handle_ = 0;
    }
    CufftPlan& operator=(CufftPlan&& other) noexcept {
        if (this != &other) {
            if (handle_ != 0) {
                (void)cufftDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = 0;
        }
        return *this;
    }

    cufftHandle get() const { return handle_; }
    cufftHandle* receive() { return &handle_; }

private:
    cufftHandle handle_ = 0;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count) {
        if (pointer_ != nullptr) {
            throw std::logic_error("DeviceBuffer allocated twice");
        }
        const std::size_t bytes = checkedProduct(count, sizeof(T), "device buffer");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes),
                  "cudaMalloc");
    }

    T* get() const { return pointer_; }

private:
    T* pointer_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { cudaCheck(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename Real>
struct FftTraits;

template <>
struct FftTraits<float> {
    using Complex = cufftComplex;
    static constexpr cufftType plan_type = CUFFT_R2C;
    static constexpr const char* name = "FP32_R2C";
    static constexpr double tolerance = 2.0e-4;

    static cufftResult execute(cufftHandle plan, float* input, Complex* output) {
        return cufftExecR2C(plan, input, output);
    }
};

template <>
struct FftTraits<double> {
    using Complex = cufftDoubleComplex;
    static constexpr cufftType plan_type = CUFFT_D2Z;
    static constexpr const char* name = "FP64_D2Z";
    static constexpr double tolerance = 2.0e-11;

    static cufftResult execute(cufftHandle plan, double* input, Complex* output) {
        return cufftExecD2Z(plan, input, output);
    }
};

struct Options {
    int warmup = 3;
    int iterations = 10;
    bool self_test_only = false;
};

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--warmup N] [--iters N] [--self-test-only]\n"
              << "Defaults: --warmup 3 --iters 10; full batch sweep.\n";
}

int parsePositive(const char* text, const char* option) {
    const std::string value(text);
    std::size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    if (consumed != value.size() || parsed <= 0) {
        throw std::invalid_argument(std::string(option) +
                                    " requires a positive integer");
    }
    return parsed;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--warmup" && i + 1 < argc) {
            options.warmup = parsePositive(argv[++i], "--warmup");
        } else if (argument == "--iters" && i + 1 < argc) {
            options.iterations = parsePositive(argv[++i], "--iters");
        } else if (argument == "--self-test-only") {
            options.self_test_only = true;
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
    }
    return options;
}

CufftPlan makePlan(int n_fft, int batch, cufftType type) {
    // Natural dense layout, with all distances expressed in elements:
    // input[b, t]  = input[b * n_fft + t]
    // output[b, k] = output[b * (n_fft / 2 + 1) + k]
    CufftPlan plan;
    const int output_size = n_fft / 2 + 1;
    cufftCheck(cufftPlanMany(plan.receive(), 1, &n_fft,
                             nullptr, 1, n_fft,
                             nullptr, 1, output_size,
                             type, batch),
               "cufftPlanMany");
    return plan;
}

struct SelfTestResult {
    double max_absolute = 0.0;
    double max_scaled = 0.0;
};

template <typename Real>
double deterministicSample(int sequence, int t) {
    switch (sequence % 4) {
        case 0:
            return t == 0 ? 1.0 : (t == 3 ? -0.25 : 0.0);
        case 1:
            return 0.125 * static_cast<double>(t) - 0.375;
        case 2:
            return std::sin(0.71 * static_cast<double>(t)) +
                   0.2 * std::cos(1.37 * static_cast<double>(t));
        default:
            return 0.0625 * static_cast<double>(
                static_cast<int>((static_cast<unsigned>(t) * 17u) % 13u) - 6);
    }
}

template <typename Real>
SelfTestResult checkDeterministicBatch(int n_fft, int batch,
                                       int sequence_offset) {
    using Traits = FftTraits<Real>;
    using Complex = typename Traits::Complex;
    const int bins = n_fft / 2 + 1;
    const std::size_t input_count = checkedProduct(
        static_cast<std::size_t>(batch), static_cast<std::size_t>(n_fft),
        "self-test input");
    const std::size_t output_count = checkedProduct(
        static_cast<std::size_t>(batch), static_cast<std::size_t>(bins),
        "self-test output");

    std::vector<Real> host_input(input_count);
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < n_fft; ++t) {
            host_input[static_cast<std::size_t>(b) * n_fft + t] =
                static_cast<Real>(deterministicSample<Real>(
                    sequence_offset + b, t));
        }
    }

    DeviceBuffer<Real> device_input(input_count);
    DeviceBuffer<Complex> device_output(output_count);
    cudaCheck(cudaMemcpy(device_input.get(), host_input.data(),
                         checkedProduct(input_count, sizeof(Real), "H2D copy"),
                         cudaMemcpyHostToDevice),
              "self-test cudaMemcpy H2D");

    CufftPlan plan = makePlan(n_fft, batch, Traits::plan_type);
    cufftCheck(Traits::execute(plan.get(), device_input.get(), device_output.get()),
               "self-test cufftExec");

    std::vector<Complex> host_output(output_count);
    cudaCheck(cudaMemcpy(host_output.data(), device_output.get(),
                         checkedProduct(output_count, sizeof(Complex), "D2H copy"),
                         cudaMemcpyDeviceToHost),
              "self-test cudaMemcpy D2H");

    SelfTestResult result;
    const double pi = std::acos(-1.0);
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < bins; ++k) {
            double reference_real = 0.0;
            double reference_imag = 0.0;
            for (int t = 0; t < n_fft; ++t) {
                const double angle = -2.0 * pi * static_cast<double>(k) *
                                     static_cast<double>(t) /
                                     static_cast<double>(n_fft);
                const double sample = static_cast<double>(
                    host_input[static_cast<std::size_t>(b) * n_fft + t]);
                reference_real += sample * std::cos(angle);
                reference_imag += sample * std::sin(angle);
            }
            const Complex actual = host_output[
                static_cast<std::size_t>(b) * bins + k];
            const double error_real = std::abs(static_cast<double>(actual.x) -
                                               reference_real);
            const double error_imag = std::abs(static_cast<double>(actual.y) -
                                               reference_imag);
            const double absolute = std::max(error_real, error_imag);
            const double scale = std::max(
                1.0, std::hypot(reference_real, reference_imag));
            const double scaled = absolute / scale;
            result.max_absolute = std::max(result.max_absolute, absolute);
            result.max_scaled = std::max(result.max_scaled, scaled);
            if (!std::isfinite(absolute) || scaled > Traits::tolerance) {
                throw std::runtime_error(
                    std::string(Traits::name) +
                    " deterministic DFT mismatch at batch " +
                    std::to_string(b) + ", bin " + std::to_string(k) +
                    ", scaled error " + std::to_string(scaled));
            }
        }
    }
    return result;
}

template <typename Real>
void runSelfTests() {
    constexpr int n_fft = 8;
    constexpr int full_batch = 3;
    constexpr int tail_batch = 2;
    const SelfTestResult full = checkDeterministicBatch<Real>(
        n_fft, full_batch, 0);
    const SelfTestResult tail = checkDeterministicBatch<Real>(
        n_fft, tail_batch, full_batch);
    std::cout << "self_test precision=" << FftTraits<Real>::name
              << " n_fft=" << n_fft
              << " signals=5 full_batch=3 tail_batch=2 status=PASS"
              << std::scientific
              << " full_max_abs=" << full.max_absolute
              << " full_max_scaled=" << full.max_scaled
              << " tail_max_abs=" << tail.max_absolute
              << " tail_max_scaled=" << tail.max_scaled
              << std::defaultfloat << '\n';
}

template <typename Real>
void fillSyntheticInput(std::vector<Real>& input, int n_fft, int batch) {
    // Integer hash data avoids any PCM/file dependency and keeps setup cheap.
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < n_fft; ++t) {
            std::uint32_t word =
                static_cast<std::uint32_t>(t) * 1664525u +
                static_cast<std::uint32_t>(b) * 1013904223u + 12345u;
            word ^= word >> 16;
            word *= 2246822519u;
            word ^= word >> 13;
            const double unit = static_cast<double>(word) / 4294967295.0;
            input[static_cast<std::size_t>(b) * n_fft + t] =
                static_cast<Real>(unit - 0.5);
        }
    }
}

template <typename Real>
void runCase(int n_fft, int batch, const Options& options) {
    using Traits = FftTraits<Real>;
    using Complex = typename Traits::Complex;
    const int bins = n_fft / 2 + 1;
    const std::size_t input_count = checkedProduct(
        static_cast<std::size_t>(batch), static_cast<std::size_t>(n_fft),
        "sweep input elements");
    const std::size_t output_count = checkedProduct(
        static_cast<std::size_t>(batch), static_cast<std::size_t>(bins),
        "sweep output elements");
    const std::size_t input_bytes = checkedProduct(
        input_count, sizeof(Real), "sweep input bytes");
    const std::size_t output_bytes = checkedProduct(
        output_count, sizeof(Complex), "sweep output bytes");

    // Plan creation is deliberately outside all execution-event intervals.
    cudaCheck(cudaDeviceSynchronize(), "pre-plan cudaDeviceSynchronize");
    const auto plan_start = std::chrono::steady_clock::now();
    CufftPlan plan = makePlan(n_fft, batch, Traits::plan_type);
    cudaCheck(cudaDeviceSynchronize(), "post-plan cudaDeviceSynchronize");
    const auto plan_stop = std::chrono::steady_clock::now();
    const double plan_ms = std::chrono::duration<double, std::milli>(
        plan_stop - plan_start).count();

    std::size_t workspace_bytes = 0;
    cufftCheck(cufftGetSize(plan.get(), &workspace_bytes), "cufftGetSize");

    DeviceBuffer<Real> device_input(input_count);
    DeviceBuffer<Complex> device_output(output_count);
    std::vector<Real> host_input(input_count);
    fillSyntheticInput(host_input, n_fft, batch);
    cudaCheck(cudaMemcpy(device_input.get(), host_input.data(), input_bytes,
                         cudaMemcpyHostToDevice),
              "sweep cudaMemcpy H2D");

    for (int i = 0; i < options.warmup; ++i) {
        cufftCheck(Traits::execute(plan.get(), device_input.get(),
                                   device_output.get()),
                   "warmup cufftExec");
    }
    cudaCheck(cudaDeviceSynchronize(), "post-warmup cudaDeviceSynchronize");

    CudaEvent start;
    CudaEvent stop;
    std::vector<float> elapsed_ms(static_cast<std::size_t>(options.iterations));
    for (int i = 0; i < options.iterations; ++i) {
        cudaCheck(cudaEventRecord(start.get()), "cudaEventRecord start");
        cufftCheck(Traits::execute(plan.get(), device_input.get(),
                                   device_output.get()),
                   "measured cufftExec");
        cudaCheck(cudaEventRecord(stop.get()), "cudaEventRecord stop");
        cudaCheck(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");
        cudaCheck(cudaEventElapsedTime(&elapsed_ms[static_cast<std::size_t>(i)],
                                       start.get(), stop.get()),
                  "cudaEventElapsedTime");
    }

    const auto minmax = std::minmax_element(elapsed_ms.begin(), elapsed_ms.end());
    double sum_ms = 0.0;
    for (float elapsed : elapsed_ms) {
        sum_ms += elapsed;
    }
    const double mean_ms = sum_ms / static_cast<double>(options.iterations);

    std::cout << Traits::name << ',' << n_fft << ',' << batch << ','
              << std::fixed << std::setprecision(6) << plan_ms << ','
              << workspace_bytes << ',' << options.warmup << ','
              << options.iterations << ',' << *minmax.first << ','
              << mean_ms << ',' << *minmax.second << ','
              << input_bytes << ',' << output_bytes << ','
              << (input_bytes + output_bytes) << '\n';
}

void printDevice() {
    int device = 0;
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties");
    std::cout << "# device_index=" << device
              << " name=\"" << properties.name << "\""
              << " compute_capability=" << properties.major << '.'
              << properties.minor
              << " total_global_memory_bytes=" << properties.totalGlobalMem
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        printDevice();
        std::cout << "# layout=input[batch][n_fft],output[batch][n_fft/2+1]"
                     ",istride=1,ostride=1,idist=n_fft,odist=n_fft/2+1\n";
        std::cout << "# timing=host_steady_clock_for_plan;CUDA_events_for_each_"
                     "post_warmup_execution;default_stream\n";

        runSelfTests<float>();
        runSelfTests<double>();
        if (options.self_test_only) {
            return 0;
        }

        const std::vector<int> n_ffts = {8192, 16384};
        const std::vector<int> batches = {1, 4, 16, 64, 128, 256, 512, 1024};
        std::cout << "precision,n_fft,batch,plan_create_ms,workspace_bytes,"
                     "warmup_iterations,measured_iterations,exec_ms_min,"
                     "exec_ms_mean,exec_ms_max,input_bytes,output_bytes,"
                     "device_buffer_bytes\n";
        for (int n_fft : n_ffts) {
            for (int batch : batches) {
                runCase<double>(n_fft, batch, options);
            }
            for (int batch : batches) {
                runCase<float>(n_fft, batch, options);
            }
        }
        cudaCheck(cudaDeviceSynchronize(), "final cudaDeviceSynchronize");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cufft_layout_probe: ERROR: " << error.what() << '\n';
        return 1;
    }
}
