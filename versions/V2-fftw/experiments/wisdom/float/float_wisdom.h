#pragma once

#include <fftw3.h>

#include <cstddef>
#include <string>

namespace melspec_fftwf {

struct FloatWisdomProblem {
    int size;
    int input_alignment;
    int output_alignment;
};

struct FloatWisdomImport {
    bool ready = false;
    std::string generated_planner;
    // Empty only when neither float wisdom source is configured. Otherwise a
    // non-empty diagnostic describes the explicit ESTIMATE fallback reason.
    std::string diagnostic;
};

struct FloatWisdomGenerationReport {
    double planning_seconds = 0.0;
    std::size_t wisdom_bytes = 0;
};

// Runtime entry point. The caller must hold the float planner mutex. This
// function never passes MEASURE/PATIENT to an FFTW planner.
fftwf_plan plan_r2c_with_float_wisdom(int n, float* input,
                                      fftwf_complex* output);

#ifdef MELSPEC_FFTWF_WISDOM_GENERATOR
// Offline-only R2C entry point, excluded from runtime builds. planner must be
// exactly "measure" or "patient". library_id identifies the actual libfftw3f.
bool generate_float_wisdom_file(const std::string& output_path, int size,
                                const std::string& planner,
                                const std::string& library_id,
                                FloatWisdomGenerationReport& report,
                                std::string& error);
#endif

}  // namespace melspec_fftwf
