#pragma once

#include <fftw3.h>

#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>

#ifndef MELSPEC_FFTW_FLOAT
#define MELSPEC_FFTW_FLOAT 0
#endif

#ifdef MELSPEC_PROFILE
#include <array>
#include <chrono>
#include <cstdio>
namespace melspec_fftw {
using Clock = std::chrono::steady_clock;
struct Counters { std::array<double, 4> seconds{}; };
inline double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
}
#define FFTW_START(name) const auto name = melspec_fftw::Clock::now()
#define FFTW_COUNTERS(name) melspec_fftw::Counters name
#define FFTW_ADD(name, index, start) name.seconds[index] += melspec_fftw::elapsed(start)
#define FFTW_TIME(label, start) std::fprintf(stderr, label "=%.9f\n", melspec_fftw::elapsed(start))
#define FFTW_PRINT(name, id) std::fprintf(stderr, "worker=%d pack_s=%.9f fft_s=%.9f power_s=%.9f mel_s=%.9f\n", id, name.seconds[0], name.seconds[1], name.seconds[2], name.seconds[3])
#else
#define FFTW_START(name)
#define FFTW_COUNTERS(name)
#define FFTW_ADD(name, index, start)
#define FFTW_TIME(label, start)
#define FFTW_PRINT(name, id)
#endif

// Link ONLY libfftw3 (or libfftw3f), not its execution threading backends. The outer
// std::threads are the sole level of parallelism (16 workers, as in V1).
namespace melspec_fftw {

#if MELSPEC_FFTW_FLOAT
using Real = float;
using Complex = fftwf_complex;
#else
using Real = double;
using Complex = fftw_complex;
#endif

inline std::mutex& planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct Free {
    void operator()(void* p) const {
#if MELSPEC_FFTW_FLOAT
        fftwf_free(p);
#else
        fftw_free(p);
#endif
    }
};

template <class T> class Buffer {
    std::unique_ptr<T, Free> data_;

public:
    explicit Buffer(size_t n)
        : data_(static_cast<T*>(
#if MELSPEC_FFTW_FLOAT
              fftwf_malloc(n * sizeof(T))
#else
              fftw_malloc(n * sizeof(T))
#endif
          )) {
        if (!data_) throw std::bad_alloc();
    }
    T* get() const { return data_.get(); }
    T& operator[](size_t i) const { return get()[i]; }
};

inline unsigned planner_flags() {
#if defined(MELSPEC_FFTW_MEASURE) && MELSPEC_FFTW_MEASURE
    return FFTW_MEASURE;
#else
    return FFTW_ESTIMATE;
#endif
}

// Planner/destructor calls are serialized, never called from worker threads.
// new-array execute is thread-safe on a shared plan when each worker has its
// own fftw_malloc buffers, identical sizes, alignment and in/out placement.
#if MELSPEC_FFTW_FLOAT
class Plan {
    fftwf_plan plan_;

public:
    Plan(int n, float* input, fftwf_complex* output) {
        std::lock_guard<std::mutex> lock(planner_mutex());
        FFTW_START(plan_start);
        plan_ = fftwf_plan_dft_r2c_1d(n, input, output, planner_flags());
        if (!plan_) throw std::runtime_error("FFTW float R2C plan creation failed");
        FFTW_TIME("r2c_plan_s", plan_start);
    }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    ~Plan() {
        std::lock_guard<std::mutex> lock(planner_mutex());
        fftwf_destroy_plan(plan_);
    }
    void apply(float* input, fftwf_complex* output) const {
        fftwf_execute_dft_r2c(plan_, input, output);
    }
};
#else
class Plan {
    fftw_plan plan_;

public:
    Plan(int n, fftw_complex* buffer) {
        std::lock_guard<std::mutex> lock(planner_mutex());
        FFTW_START(plan_start);
        plan_ = fftw_plan_dft_1d(n, buffer, buffer, FFTW_FORWARD, planner_flags());
        if (!plan_) throw std::runtime_error("FFTW C2C plan creation failed");
        FFTW_TIME("c2c_plan_s", plan_start);
    }
    Plan(int n, double* input, fftw_complex* output) {
        std::lock_guard<std::mutex> lock(planner_mutex());
        FFTW_START(plan_start);
        plan_ = fftw_plan_dft_r2c_1d(n, input, output, planner_flags());
        if (!plan_) throw std::runtime_error("FFTW R2C plan creation failed");
        FFTW_TIME("r2c_plan_s", plan_start);
    }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    ~Plan() {
        std::lock_guard<std::mutex> lock(planner_mutex());
        fftw_destroy_plan(plan_);
    }
    void apply(fftw_complex* buffer) const {
        fftw_execute_dft(plan_, buffer, buffer);
    }
    void apply(double* input, fftw_complex* output) const {
        fftw_execute_dft_r2c(plan_, input, output);
    }
};
#endif

}  // namespace melspec_fftw
