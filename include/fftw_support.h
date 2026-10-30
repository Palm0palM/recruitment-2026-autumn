#pragma once

#include <fftw3.h>

#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>

// Link ONLY libfftw3, not its OpenMP/pthreads execution backends. The outer
// std::threads are the sole level of parallelism (16 workers, as in V1).
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
class Plan {
    fftw_plan plan_;

public:
    Plan(int n, fftw_complex* buffer) {
        std::lock_guard<std::mutex> lock(planner_mutex());
        plan_ = fftw_plan_dft_1d(n, buffer, buffer, FFTW_FORWARD, planner_flags());
        if (!plan_) throw std::runtime_error("FFTW C2C plan creation failed");
    }
    Plan(int n, double* input, fftw_complex* output) {
        std::lock_guard<std::mutex> lock(planner_mutex());
        plan_ = fftw_plan_dft_r2c_1d(n, input, output, planner_flags());
        if (!plan_) throw std::runtime_error("FFTW R2C plan creation failed");
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
