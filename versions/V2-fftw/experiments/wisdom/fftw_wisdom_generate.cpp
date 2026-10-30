// Offline-only FFTW wisdom generator. This tool has no PCM input and never
// participates in the melspec executable's runtime planning path.
#include "fftw_wisdom.h"

#include <charconv>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

void usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --transform r2c|half-c2c --size N --planner measure|patient"
           " --library-id ID --output FILE\n\n"
        << "  --size        Actual FFTW plan size: n_fft for R2C, n_fft/2 for half-C2C.\n"
        << "  --library-id  Identity of the linked libfftw3 (prefer its SHA-256).\n"
        << "The output is bound to this FFTW build, CPU, double precision and layout.\n";
}

bool positive_int(const std::string& text, int& value) {
    unsigned long long parsed = 0;
    const char* first = text.data();
    const char* last = first + text.size();
    auto result = std::from_chars(first, last, parsed);
    if (result.ec != std::errc() || result.ptr != last || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        return false;
    value = static_cast<int>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" ||
                      std::string(argv[1]) == "-h")) {
        usage(argv[0]);
        return 0;
    }
    std::string transform_name;
    std::string planner;
    std::string library_id;
    std::string output;
    int size = 0;
    bool have_size = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << key << "\n";
            usage(argv[0]);
            return 2;
        }
        const std::string value = argv[++i];
        if (key == "--transform")
            transform_name = value;
        else if (key == "--size") {
            if (!positive_int(value, size)) {
                std::cerr << "--size must be a positive int\n";
                return 2;
            }
            have_size = true;
        } else if (key == "--planner")
            planner = value;
        else if (key == "--library-id")
            library_id = value;
        else if (key == "--output")
            output = value;
        else {
            std::cerr << "Unknown argument: " << key << "\n";
            usage(argv[0]);
            return 2;
        }
    }
    if (!have_size || transform_name.empty() || planner.empty() ||
        library_id.empty() || output.empty()) {
        usage(argv[0]);
        return 2;
    }

    melspec_fftw::WisdomTransform transform;
    if (transform_name == "r2c")
        transform = melspec_fftw::WisdomTransform::r2c;
    else if (transform_name == "half-c2c" || transform_name == "c2c")
        transform = melspec_fftw::WisdomTransform::c2c;
    else {
        std::cerr << "--transform must be r2c or half-c2c\n";
        return 2;
    }

    melspec_fftw::WisdomGenerationReport report;
    std::string error;
    if (!melspec_fftw::generate_wisdom_file(output, transform, size, planner,
                                            library_id, report, error)) {
        std::cerr << "Wisdom generation failed: " << error << "\n";
        return 1;
    }
    std::cout << "output=" << output << " transform=" << transform_name
              << " size=" << size << " planner=" << planner
              << " precision=double planning_s=" << report.planning_seconds
              << " wisdom_bytes=" << report.wisdom_bytes << "\n";
    return 0;
}
