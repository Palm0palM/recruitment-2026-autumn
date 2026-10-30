#pragma once

#include <fftw3.h>

#include <cstddef>
#include <string>

namespace melspec_fftw {

enum class WisdomTransform { r2c, c2c };

// The size is the actual FFTW transform size: n_fft for R2C and n_fft/2 for
// the half-length C2C route. Alignment values come from fftw_alignment_of.
struct WisdomProblem {
    WisdomTransform transform;
    int size;
    int input_alignment;
    int output_alignment;
};

struct WisdomImport {
    bool ready = false;
    std::string generated_planner;
    // Empty only when MELSPEC_FFTW_WISDOM is unset/empty. Otherwise any
    // non-empty diagnostic describes an explicit ESTIMATE fallback reason.
    std::string diagnostic;
};

struct WisdomGenerationReport {
    double planning_seconds = 0.0;
    std::size_t wisdom_bytes = 0;
};

const char* wisdom_transform_name(WisdomTransform transform);

// Runtime entry points. The caller must hold planner_mutex(). These functions
// never pass MEASURE/PATIENT to an FFTW planner.
fftw_plan plan_r2c_with_wisdom(int n, double* input, fftw_complex* output);
fftw_plan plan_c2c_with_wisdom(int n, fftw_complex* inout);

#ifdef MELSPEC_FFTW_WISDOM_GENERATOR
// Offline entry point, excluded from runtime builds. planner must be exactly
// "measure" or "patient". library_id is an operator-supplied identity for the
// actual linked FFTW library (preferably its SHA-256), not an identity inferred
// from PCM data.
bool generate_wisdom_file(const std::string& output_path,
                          WisdomTransform transform, int size,
                          const std::string& planner,
                          const std::string& library_id,
                          WisdomGenerationReport& report,
                          std::string& error);
#endif

}  // namespace melspec_fftw
