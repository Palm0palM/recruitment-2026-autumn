#pragma once

#include <fftw3.h>

#include "fftw_wisdom.h"

#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>

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

// Link ONLY libfftw3, not its OpenMP/pthreads execution backends. The outer
// std::threads are the sole level of parallelism (at most 16 workers).
namespace melspec_fftw {

inline std::mutex& planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct Free {
    void operator()(void* p) const { fftw_free(p); }
};

template <class T> class Buffer {
    std::unique_ptr<T, Free> data_;

public:
    explicit Buffer(size_t n)
        : data_(static_cast<T*>(fftw_malloc(n * sizeof(T)))) {
        if (!data_) throw std::bad_alloc();
    }
    T* get() const { return data_.get(); }
    T& operator[](size_t i) const { return get()[i]; }
};

// Planner/import/destructor calls are serialized, never called from workers.
// Runtime planning is exclusively WISDOM_ONLY|ESTIMATE after a validated
// import, or ESTIMATE fallback. MEASURE/PATIENT exist only in the offline tool.
class Plan {
    fftw_plan plan_;

public:
    Plan(int n, fftw_complex* buffer) {
        std::lock_guard<std::mutex> lock(planner_mutex());
        FFTW_START(plan_start);
        plan_ = plan_c2c_with_wisdom(n, buffer);
        if (!plan_) throw std::runtime_error("FFTW C2C plan creation failed");
        FFTW_TIME("c2c_plan_s", plan_start);
    }
    Plan(int n, double* input, fftw_complex* output) {
        std::lock_guard<std::mutex> lock(planner_mutex());
        FFTW_START(plan_start);
        plan_ = plan_r2c_with_wisdom(n, input, output);
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

}  // namespace melspec_fftw
