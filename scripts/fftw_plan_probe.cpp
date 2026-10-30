// Diagnostic only: cold planning, actual selected codelets, single-thread
// copy+execution cost. This is NOT the 16-worker end-to-end benchmark.
#include <fftw3.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

int main() {
    std::printf("version=%s compiler=%s\n", fftw_version, fftw_cc);
    constexpr int repetitions = 5000;
    for (int n : {8192, 16384}) {
        std::vector<double> samples(n);
        for (int i = 0; i < n; ++i) samples[i] = ((i * 17) % 257 - 128) / 256.0;
        for (bool r2c : {false, true}) {
            for (unsigned flags : {static_cast<unsigned>(FFTW_ESTIMATE), static_cast<unsigned>(FFTW_MEASURE)}) {
                fftw_forget_wisdom();
                auto* input = fftw_alloc_real(n);
                auto* output = fftw_alloc_complex(n / 2 + 1);
                if (!input || !output) return 1;
                auto start = Clock::now();
                fftw_plan plan = r2c ? fftw_plan_dft_r2c_1d(n, input, output, flags)
                    : fftw_plan_dft_1d(n / 2, reinterpret_cast<fftw_complex*>(input),
                                       reinterpret_cast<fftw_complex*>(input), FFTW_FORWARD, flags);
                double planning = seconds(start);
                if (!plan) return 1;
                char* description = fftw_sprint_plan(plan);
                std::printf("kind=%s n_fft=%d planner=%s plan_s=%.9f alignment_in=%d alignment_out=%d\n%s\n",
                    r2c ? "r2c" : "half-c2c", n, flags == FFTW_ESTIMATE ? "estimate" : "measure",
                    planning, fftw_alignment_of(input), fftw_alignment_of(reinterpret_cast<double*>(output)), description);
                std::free(description);
                start = Clock::now();
                for (int r = 0; r < repetitions; ++r) {
                    // Refresh inputs: repeated in-place FFTs would grow/overflow.
                    for (int i = 0; i < n; ++i) input[i] = samples[i];
                    fftw_execute(plan);
                }
                std::printf("single_thread_copy_execute_us=%.9f repeats=%d\n", seconds(start) * 1e6 / repetitions, repetitions);
                fftw_destroy_plan(plan);
                fftw_free(output);
                fftw_free(input);
            }
        }
    }
}
