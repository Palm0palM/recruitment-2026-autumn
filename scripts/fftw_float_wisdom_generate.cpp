// Offline-only fftwf R2C wisdom generator. No PCM input is accepted or read.
#include "fftw_float_wisdom.h"

#include <charconv>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>

namespace {

void usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --size N --planner measure|patient --library-id ID --output FILE\n\n"
        << "Generate float R2C wisdom for actual n_fft=N. For directory mode,\n"
        << "name the output r2c-N.wisdom. ID should identify the linked libfftw3f.\n";
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
        if (key == "--size") {
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
    if (!have_size || planner.empty() || library_id.empty() || output.empty()) {
        usage(argv[0]);
        return 2;
    }

    melspec_fftwf::FloatWisdomGenerationReport report;
    std::string error;
    if (!melspec_fftwf::generate_float_wisdom_file(
            output, size, planner, library_id, report, error)) {
        std::cerr << "Float wisdom generation failed: " << error << "\n";
        return 1;
    }
    std::cout << "output=" << output << " transform=r2c size=" << size
              << " planner=" << planner
              << " precision=float planning_s=" << report.planning_seconds
              << " wisdom_bytes=" << report.wisdom_bytes << "\n";
    return 0;
}
