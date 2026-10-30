// REVIEW DRAFT: isolated FFTW plan/layout inspector and optional single-thread
// copy+execute microbenchmark. Not executed by the ISA subagent.
//
// Double example (main Agent reviews/schedules before running):
//   g++ -O3 -march=x86-64-v3 -mno-avx512f -std=c++17 \
//     -I PREFIX/include isa_plan_probe.cpp PREFIX/lib/libfftw3.a -lm -o PROBE
// Float: add -DFFTW_PROBE_FLOAT=1 and link PREFIX/lib/libfftw3f.a.
//
// This is not an end-to-end/16-worker benchmark and does not validate Mel output.
#include <fftw3.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template <class R> struct FftwApi;

template <> struct FftwApi<double> {
    using Real = double;
    using Complex = fftw_complex;
    using Plan = fftw_plan;
    static const char* precision() { return "double"; }
    static const char* version() { return fftw_version; }
    static const char* compiler() { return fftw_cc; }
    static Real* alloc_real(size_t n) { return fftw_alloc_real(n); }
    static Complex* alloc_complex(size_t n) { return fftw_alloc_complex(n); }
    static void free_buffer(void* p) { fftw_free(p); }
    static int alignment(Real* p) { return fftw_alignment_of(p); }
    static void forget_wisdom() { fftw_forget_wisdom(); }
    static int import_wisdom(const char* path) {
        return fftw_import_wisdom_from_filename(path);
    }
    static int export_wisdom(const char* path) {
        return fftw_export_wisdom_to_filename(path);
    }
    static Plan plan_r2c(int n, Real* in, Complex* out, unsigned flags) {
        return fftw_plan_dft_r2c_1d(n, in, out, flags);
    }
    static Plan plan_c2c(int n, Complex* inout, unsigned flags) {
        return fftw_plan_dft_1d(n, inout, inout, FFTW_FORWARD, flags);
    }
    static Plan plan_many_r2c(int n, int batch, Real* in, int idist,
                              Complex* out, int odist, unsigned flags) {
        return fftw_plan_many_dft_r2c(1, &n, batch, in, nullptr, 1, idist,
                                      out, nullptr, 1, odist, flags);
    }
    static Plan plan_many_c2c(int n, int batch, Complex* inout,
                              int dist, unsigned flags) {
        return fftw_plan_many_dft(1, &n, batch, inout, nullptr, 1, dist,
                                  inout, nullptr, 1, dist, FFTW_FORWARD, flags);
    }
    static char* sprint(Plan p) { return fftw_sprint_plan(p); }
    static void execute(Plan p) { fftw_execute(p); }
    static void destroy(Plan p) { fftw_destroy_plan(p); }
};

template <> struct FftwApi<float> {
    using Real = float;
    using Complex = fftwf_complex;
    using Plan = fftwf_plan;
    static const char* precision() { return "float"; }
    static const char* version() { return fftwf_version; }
    static const char* compiler() { return fftwf_cc; }
    static Real* alloc_real(size_t n) { return fftwf_alloc_real(n); }
    static Complex* alloc_complex(size_t n) { return fftwf_alloc_complex(n); }
    static void free_buffer(void* p) { fftwf_free(p); }
    static int alignment(Real* p) { return fftwf_alignment_of(p); }
    static void forget_wisdom() { fftwf_forget_wisdom(); }
    static int import_wisdom(const char* path) {
        return fftwf_import_wisdom_from_filename(path);
    }
    static int export_wisdom(const char* path) {
        return fftwf_export_wisdom_to_filename(path);
    }
    static Plan plan_r2c(int n, Real* in, Complex* out, unsigned flags) {
        return fftwf_plan_dft_r2c_1d(n, in, out, flags);
    }
    static Plan plan_c2c(int n, Complex* inout, unsigned flags) {
        return fftwf_plan_dft_1d(n, inout, inout, FFTW_FORWARD, flags);
    }
    static Plan plan_many_r2c(int n, int batch, Real* in, int idist,
                              Complex* out, int odist, unsigned flags) {
        return fftwf_plan_many_dft_r2c(1, &n, batch, in, nullptr, 1, idist,
                                       out, nullptr, 1, odist, flags);
    }
    static Plan plan_many_c2c(int n, int batch, Complex* inout,
                              int dist, unsigned flags) {
        return fftwf_plan_many_dft(1, &n, batch, inout, nullptr, 1, dist,
                                   inout, nullptr, 1, dist, FFTW_FORWARD, flags);
    }
    static char* sprint(Plan p) { return fftwf_sprint_plan(p); }
    static void execute(Plan p) { fftwf_execute(p); }
    static void destroy(Plan p) { fftwf_destroy_plan(p); }
};

struct Options {
    int n = 8192;
    int batch = 1;
    int out_stride = 0;  // Complex-element units; zero means dense.
    int repeats = 0;
    int warmup = 0;
    bool many = false;
    std::string kind = "r2c";
    std::string planner = "estimate";
    std::string wisdom_in;
    std::string wisdom_out;
};

static void usage(const char* argv0) {
    std::printf(
        "usage: %s [--n 8192] [--kind r2c|half-c2c] [--api single|many]\n"
        "          [--planner estimate|measure] [--batch B]\n"
        "          [--out-stride COMPLEX_ELEMENTS] [--repeats N] [--warmup N]\n"
        "          [--wisdom-in FILE] [--wisdom-out FILE]\n"
        "Default repeats=0: plan/layout inspection only. half-c2c transforms\n"
        "n/2 complex values in place and excludes any A1 recovery stage.\n", argv0);
}

static int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end || value <= 0 || value > 1000000000L)
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<int>(value);
}

static Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
        std::string value = argv[++i];
        if (arg == "--n") o.n = parse_int(value.c_str(), "n");
        else if (arg == "--batch") o.batch = parse_int(value.c_str(), "batch");
        else if (arg == "--out-stride")
            o.out_stride = parse_int(value.c_str(), "out-stride");
        else if (arg == "--repeats") {
            // Zero is meaningful for repeats, unlike the other integer options.
            char* end = nullptr;
            long v = std::strtol(value.c_str(), &end, 10);
            if (!end || *end || v < 0 || v > 1000000000L)
                throw std::runtime_error("invalid repeats");
            o.repeats = static_cast<int>(v);
        } else if (arg == "--warmup") {
            char* end = nullptr;
            long v = std::strtol(value.c_str(), &end, 10);
            if (!end || *end || v < 0 || v > 1000000000L)
                throw std::runtime_error("invalid warmup");
            o.warmup = static_cast<int>(v);
        } else if (arg == "--kind") o.kind = value;
        else if (arg == "--planner") o.planner = value;
        else if (arg == "--api") {
            if (value == "many") o.many = true;
            else if (value == "single") o.many = false;
            else throw std::runtime_error("api must be single or many");
        } else if (arg == "--wisdom-in") o.wisdom_in = value;
        else if (arg == "--wisdom-out") o.wisdom_out = value;
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (o.n < 2 || o.n % 2) throw std::runtime_error("n must be positive and even");
    if (o.kind != "r2c" && o.kind != "half-c2c")
        throw std::runtime_error("kind must be r2c or half-c2c");
    if (o.planner != "estimate" && o.planner != "measure")
        throw std::runtime_error("planner must be estimate or measure");
    if (!o.many && o.batch != 1)
        throw std::runtime_error("batch>1 requires --api many");
    if (o.out_stride && (!o.many || o.kind != "r2c"))
        throw std::runtime_error("custom out-stride requires many R2C");
    return o;
}

template <class R> static int run(const Options& o) {
    using A = FftwApi<R>;
    using Real = typename A::Real;
    using Complex = typename A::Complex;
    using Plan = typename A::Plan;

    struct Buffer {
        void* p;
        explicit Buffer(void* q) : p(q) {}
        ~Buffer() { if (p) A::free_buffer(p); }
    };
    struct PlanHolder {
        Plan p;
        explicit PlanHolder(Plan q) : p(q) {}
        ~PlanHolder() { if (p) A::destroy(p); }
    };

    const int c2c_n = o.n / 2;
    const int out_width = o.kind == "r2c" ? o.n / 2 + 1 : c2c_n;
    const int out_stride = o.out_stride ? o.out_stride : out_width;
    if (out_stride < out_width) throw std::runtime_error("out-stride is too small");
    const size_t input_count = static_cast<size_t>(o.batch) * o.n;
    const size_t output_count = o.kind == "r2c"
        ? static_cast<size_t>(o.batch - 1) * out_stride + out_width : 0;

    Buffer input_holder(A::alloc_real(input_count));
    Real* input = static_cast<Real*>(input_holder.p);
    Buffer output_holder(o.kind == "r2c" ? A::alloc_complex(output_count) : nullptr);
    Complex* complex_input = reinterpret_cast<Complex*>(input);
    Complex* output = o.kind == "r2c" ? static_cast<Complex*>(output_holder.p) : complex_input;
    if (!input || (o.kind == "r2c" && !output_holder.p))
        throw std::runtime_error("FFTW allocation failed");

    // Allocate a second new-array layout before planning to expose accidental
    // dependence on one particular allocation. It is inspected, not executed.
    Buffer new_input_holder(A::alloc_real(input_count));
    Real* new_input = static_cast<Real*>(new_input_holder.p);
    Buffer new_output_holder(o.kind == "r2c" ? A::alloc_complex(output_count) : nullptr);
    Complex* new_complex_input = reinterpret_cast<Complex*>(new_input);
    Complex* new_output = o.kind == "r2c"
        ? static_cast<Complex*>(new_output_holder.p) : new_complex_input;
    if (!new_input || (o.kind == "r2c" && !new_output_holder.p))
        throw std::runtime_error("second FFTW allocation failed");

    A::forget_wisdom();
    std::printf("precision=%s version=%s compiler=%s\n",
                A::precision(), A::version(), A::compiler());
    std::printf("kind=%s n_fft=%d c2c_n=%d batch=%d api=%s planner=%s "
                "out_width_complex=%d out_stride_complex=%d\n",
                o.kind.c_str(), o.n, c2c_n, o.batch, o.many ? "many" : "single",
                o.planner.c_str(), out_width, out_stride);

    if (!o.wisdom_in.empty()) {
        auto start = Clock::now();
        int ok = A::import_wisdom(o.wisdom_in.c_str());
        std::printf("wisdom_import=%s file=%s import_s=%.9f\n",
                    ok ? "ok" : "failed", o.wisdom_in.c_str(), seconds_since(start));
        if (!ok) return 3;  // Never silently relabel an ESTIMATE fallback.
    } else {
        std::printf("wisdom_import=not-requested\n");
    }

    const unsigned flags = o.planner == "estimate" ? FFTW_ESTIMATE : FFTW_MEASURE;
    auto plan_start = Clock::now();
    Plan raw_plan = nullptr;
    if (o.many) {
        raw_plan = o.kind == "r2c"
            ? A::plan_many_r2c(o.n, o.batch, input, o.n, output, out_stride, flags)
            : A::plan_many_c2c(c2c_n, o.batch, complex_input, c2c_n, flags);
    } else {
        raw_plan = o.kind == "r2c"
            ? A::plan_r2c(o.n, input, output, flags)
            : A::plan_c2c(c2c_n, complex_input, flags);
    }
    std::printf("plan_s=%.9f\n", seconds_since(plan_start));
    if (!raw_plan) throw std::runtime_error("FFTW plan creation failed");
    PlanHolder plan(raw_plan);

    char* description = A::sprint(plan.p);
    if (!description) throw std::runtime_error("fftw_sprint_plan failed");
    std::printf("actual_plan_begin\n%s\nactual_plan_end\n", description);
    std::free(description);

    auto show_pointer = [](const char* tag, const char* which, int row,
                           Real* p) {
        uintptr_t address = reinterpret_cast<uintptr_t>(p);
        std::printf("layout=%s which=%s row=%d alignment_class=%d "
                    "address_mod16=%llu address_mod32=%llu address_mod64=%llu\n",
                    tag, which, row, A::alignment(p),
                    static_cast<unsigned long long>(address % 16),
                    static_cast<unsigned long long>(address % 32),
                    static_cast<unsigned long long>(address % 64));
    };
    auto show_set = [&](const char* tag, Real* in, Complex* out) {
        int rows = std::min(o.batch, 16);
        for (int b = 0; b < rows; ++b) {
            Real* in_row = in + static_cast<size_t>(b) * o.n;
            Complex* out_row = out + static_cast<size_t>(b) * out_stride;
            show_pointer(tag, "input", b, in_row);
            show_pointer(tag, "output", b, reinterpret_cast<Real*>(out_row));
        }
        if (rows != o.batch)
            std::printf("layout=%s rows_truncated=%d total_rows=%d\n", tag, rows, o.batch);
    };
    show_set("planning", input, output);
    show_set("new-array-inspection-only", new_input, new_output);

    if (!o.wisdom_out.empty()) {
        auto start = Clock::now();
        int ok = A::export_wisdom(o.wisdom_out.c_str());
        std::printf("wisdom_export=%s file=%s export_s=%.9f\n",
                    ok ? "ok" : "failed", o.wisdom_out.c_str(), seconds_since(start));
        if (!ok) return 4;
    }

    if (o.repeats > 0 || o.warmup > 0) {
        std::vector<Real> samples(input_count);
        for (size_t i = 0; i < input_count; ++i) {
            int residue = static_cast<int>((i * 17) % 257);
            samples[i] = static_cast<Real>((residue - 128) / 256.0);
        }
        auto copy_execute = [&] {
            std::copy(samples.begin(), samples.end(), input);
            A::execute(plan.p);
        };
        for (int r = 0; r < o.warmup; ++r) copy_execute();
        if (o.repeats > 0) {
            auto start = Clock::now();
            for (int r = 0; r < o.repeats; ++r) copy_execute();
            double elapsed = seconds_since(start);
            std::printf("single_thread_copy_execute_us=%.9f repeats=%d warmup=%d "
                        "batch_frames=%d per_frame_derived_us=%.9f\n",
                        elapsed * 1e6 / o.repeats, o.repeats, o.warmup, o.batch,
                        elapsed * 1e6 / o.repeats / o.batch);
        }
    }
    std::printf("scope=single-thread-copy-plus-execute-not-end-to-end\n");
    return 0;
}

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
#ifdef FFTW_PROBE_FLOAT
        return run<float>(options);
#else
        return run<double>(options);
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "isa_plan_probe: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }
}
