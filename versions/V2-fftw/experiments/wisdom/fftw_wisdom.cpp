#include "fftw_wisdom.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace melspec_fftw {
namespace {

constexpr std::size_t kMaxWisdomFileBytes = 16 * 1024 * 1024;
constexpr const char* kMagic = "MELSPEC_FFTW_WISDOM_V1";

struct FftwFree {
    void operator()(void* p) const { fftw_free(p); }
};
struct PlanDeleter {
    void operator()(fftw_plan p) const {
        if (p) fftw_destroy_plan(p);
    }
};
using RealPtr = std::unique_ptr<double, FftwFree>;
using ComplexPtr = std::unique_ptr<fftw_complex, FftwFree>;
using StringPtr = std::unique_ptr<char, FftwFree>;
using PlanPtr = std::unique_ptr<std::remove_pointer<fftw_plan>::type, PlanDeleter>;

std::string cpu_identity() {
    std::ostringstream out;
#if defined(__x86_64__) || defined(__i386__)
    out << "x86";
    auto append = [&out](const char* name, unsigned a, unsigned b,
                         unsigned c, unsigned d) {
        out << ';' << name << '=' << std::hex << a << ',' << b << ',' << c
            << ',' << d;
    };
    unsigned a = 0, b = 0, c = 0, d = 0;
    const unsigned max_leaf = __get_cpuid_max(0, nullptr);
    if (__get_cpuid(0, &a, &b, &c, &d)) append("leaf0", a, b, c, d);
    a = b = c = d = 0;
    bool osxsave = false;
    if (max_leaf >= 1 && __get_cpuid(1, &a, &b, &c, &d)) {
        // Initial APIC ID identifies the current logical CPU, not the machine.
        // It may differ between offline generation and a scheduled worker.
        b &= 0x00ffffffu;
        append("leaf1", a, b, c, d);
        osxsave = (c & (1u << 27)) != 0;
    }
    a = b = c = d = 0;
    if (max_leaf >= 7) {
        __cpuid_count(7, 0, a, b, c, d);
        append("leaf7.0", a, b, c, d);
    }
    const unsigned max_extended = __get_cpuid_max(0x80000000u, nullptr);
    a = b = c = d = 0;
    if (max_extended >= 0x80000001u &&
        __get_cpuid(0x80000001u, &a, &b, &c, &d))
        append("ext1", a, b, c, d);
    std::uint64_t xcr0 = 0;
    if (osxsave) {
        unsigned low = 0, high = 0;
        // Byte encoding avoids requiring -mxsave at application compile time.
        __asm__ volatile(".byte 0x0f,0x01,0xd0"
                         : "=a"(low), "=d"(high)
                         : "c"(0));
        xcr0 = (static_cast<std::uint64_t>(high) << 32) | low;
    }
    out << ";xcr0=" << std::hex << xcr0;
#elif defined(__aarch64__)
    out << "aarch64";
#elif defined(__arm__)
    out << "arm";
#else
    out << "unknown-architecture";
#endif
    return out.str();
}

std::uint64_t fnv1a64(const std::string& text) {
    std::uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

bool parse_u64(const std::string& text, int base, std::uint64_t& value) {
    if (text.empty()) return false;
    const char* first = text.data();
    const char* last = first + text.size();
    auto parsed = std::from_chars(first, last, value, base);
    return parsed.ec == std::errc() && parsed.ptr == last;
}

bool parse_nonnegative_int(const std::string& text, int& value) {
    std::uint64_t parsed = 0;
    if (!parse_u64(text, 10, parsed) ||
        parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parse_positive_int(const std::string& text, int& value) {
    return parse_nonnegative_int(text, value) && value > 0;
}

bool safe_metadata_value(const std::string& value) {
    return !value.empty() && value.size() <= 4096 &&
           value.find_first_of("\r\n\0", 0, 3) == std::string::npos;
}

const char* placement_name(WisdomTransform transform) {
    return transform == WisdomTransform::r2c ? "out-of-place" : "in-place";
}

struct Envelope {
    std::map<std::string, std::string> fields;
    std::string payload;
};

bool read_envelope(const std::string& path, Envelope& envelope,
                   std::string& error) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "file-missing-or-unreadable";
        return false;
    }
    const std::streamoff end = input.tellg();
    if (end <= 0 || static_cast<std::uint64_t>(end) > kMaxWisdomFileBytes) {
        error = "file-empty-or-too-large";
        return false;
    }
    std::string bytes(static_cast<std::size_t>(end), '\0');
    input.seekg(0, std::ios::beg);
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        error = "file-read-failed";
        return false;
    }
    const std::size_t separator = bytes.find("\n\n");
    if (separator == std::string::npos) {
        error = "envelope-header-missing";
        return false;
    }
    envelope.payload = bytes.substr(separator + 2);
    std::istringstream header(bytes.substr(0, separator));
    std::string line;
    if (!std::getline(header, line) || line != kMagic) {
        error = "envelope-version-mismatch";
        return false;
    }
    while (std::getline(header, line)) {
        const std::size_t equal = line.find('=');
        if (equal == std::string::npos || equal == 0) {
            error = "envelope-malformed-field";
            return false;
        }
        const std::string key = line.substr(0, equal);
        const std::string value = line.substr(equal + 1);
        if (!envelope.fields.emplace(key, value).second) {
            error = "envelope-duplicate-field";
            return false;
        }
    }
    static constexpr std::array<const char*, 13> required = {
        "fftw_version", "fftw_cc", "precision", "library_id", "cpu",
        "transform", "size", "placement", "input_alignment",
        "output_alignment", "planner", "payload_bytes", "payload_fnv1a64"};
    if (envelope.fields.size() != required.size()) {
        error = "envelope-field-count-mismatch";
        return false;
    }
    for (const char* key : required) {
        if (envelope.fields.find(key) == envelope.fields.end()) {
            error = std::string("envelope-missing-") + key;
            return false;
        }
    }
    return true;
}

WisdomImport import_for_problem(const WisdomProblem& problem) {
    WisdomImport result;
    const char* file = std::getenv("MELSPEC_FFTW_WISDOM");
    const char* directory = std::getenv("MELSPEC_FFTW_WISDOM_DIRECTORY");
    std::string path;
    if (directory && *directory)
        path = std::string(directory) + "/" + wisdom_transform_name(problem.transform) +
               "-" + std::to_string(problem.size) + ".wisdom";
    else if (file && *file)
        path = file;
    if (path.empty()) return result;

    // Do not let partially imported or ambient wisdom affect a fallback plan.
    fftw_forget_wisdom();
    auto fail = [&result](const std::string& reason) {
        result.ready = false;
        result.diagnostic = reason;
        return result;
    };

    const char* library_id = std::getenv("MELSPEC_FFTW_LIBRARY_ID");
    if (!library_id || !*library_id)
        return fail("library-id-required");

    Envelope envelope;
    std::string error;
    if (!read_envelope(path, envelope, error)) return fail(error);
    const auto& f = envelope.fields;
    auto matches = [&f](const char* key, const std::string& expected) {
        return f.at(key) == expected;
    };
    if (!matches("fftw_version", fftw_version))
        return fail("fftw-version-mismatch");
    if (!matches("fftw_cc", fftw_cc)) return fail("fftw-build-flags-mismatch");
    if (!matches("precision", "double")) return fail("precision-mismatch");
    if (!matches("library_id", library_id)) return fail("library-id-mismatch");
    if (!matches("cpu", cpu_identity())) return fail("cpu-mismatch");
    if (!matches("transform", wisdom_transform_name(problem.transform)))
        return fail("transform-layout-mismatch");
    if (!matches("placement", placement_name(problem.transform)))
        return fail("placement-layout-mismatch");
    if (!matches("planner", "measure") && !matches("planner", "patient"))
        return fail("generator-planner-invalid");

    int saved_size = 0;
    if (!parse_positive_int(f.at("size"), saved_size))
        return fail("saved-size-invalid");
    if (saved_size != problem.size) return fail("transform-size-mismatch");
    int input_alignment = -1, output_alignment = -1;
    if (!parse_nonnegative_int(f.at("input_alignment"), input_alignment))
        return fail("saved-input-alignment-invalid");
    if (!parse_nonnegative_int(f.at("output_alignment"), output_alignment))
        return fail("saved-output-alignment-invalid");
    if (input_alignment != problem.input_alignment ||
        output_alignment != problem.output_alignment)
        return fail("alignment-layout-mismatch");

    std::uint64_t payload_bytes = 0;
    if (!parse_u64(f.at("payload_bytes"), 10, payload_bytes) ||
        payload_bytes != envelope.payload.size())
        return fail("payload-length-mismatch");
    std::uint64_t saved_hash = 0;
    if (!parse_u64(f.at("payload_fnv1a64"), 16, saved_hash) ||
        saved_hash != fnv1a64(envelope.payload))
        return fail("payload-checksum-mismatch");
    if (envelope.payload.empty() ||
        envelope.payload.find('\0') != std::string::npos)
        return fail("payload-invalid");
    if (!fftw_import_wisdom_from_string(envelope.payload.c_str())) {
        fftw_forget_wisdom();
        return fail("fftw-import-rejected-payload");
    }

    result.ready = true;
    result.generated_planner = f.at("planner");
    return result;
}

template <class Planner>
fftw_plan runtime_plan(const WisdomProblem& problem, Planner planner) {
    WisdomImport imported = import_for_problem(problem);
    if (!imported.diagnostic.empty()) {
        std::fprintf(stderr,
                     "FFTW wisdom status=fallback planner=ESTIMATE reason=%s\n",
                     imported.diagnostic.c_str());
    }
    if (imported.ready) {
        // More rigorous offline wisdom is valid for an ESTIMATE request. The
        // WISDOM_ONLY bit prohibits runtime measurement or heuristic planning.
        fftw_plan plan = planner(FFTW_WISDOM_ONLY | FFTW_ESTIMATE);
        if (plan) {
            std::fprintf(stderr,
                         "FFTW wisdom status=loaded transform=%s size=%d "
                         "generated_with=%s runtime=WISDOM_ONLY|ESTIMATE\n",
                         wisdom_transform_name(problem.transform), problem.size,
                         imported.generated_planner.c_str());
            return plan;
        }
        fftw_forget_wisdom();
        std::fprintf(stderr,
                     "FFTW wisdom status=fallback planner=ESTIMATE "
                     "reason=no-applicable-imported-plan\n");
    }
    return planner(FFTW_ESTIMATE);
}

}  // namespace

const char* wisdom_transform_name(WisdomTransform transform) {
    return transform == WisdomTransform::r2c ? "r2c" : "half-c2c";
}

fftw_plan plan_r2c_with_wisdom(int n, double* input, fftw_complex* output) {
    const WisdomProblem problem = {
        WisdomTransform::r2c, n, fftw_alignment_of(input),
        fftw_alignment_of(reinterpret_cast<double*>(output))};
    return runtime_plan(problem, [n, input, output](unsigned flags) {
        return fftw_plan_dft_r2c_1d(n, input, output, flags);
    });
}

fftw_plan plan_c2c_with_wisdom(int n, fftw_complex* inout) {
    const int alignment = fftw_alignment_of(reinterpret_cast<double*>(inout));
    const WisdomProblem problem = {WisdomTransform::c2c, n, alignment,
                                   alignment};
    return runtime_plan(problem, [n, inout](unsigned flags) {
        return fftw_plan_dft_1d(n, inout, inout, FFTW_FORWARD, flags);
    });
}

#ifdef MELSPEC_FFTW_WISDOM_GENERATOR
bool generate_wisdom_file(const std::string& output_path,
                          WisdomTransform transform, int size,
                          const std::string& planner,
                          const std::string& library_id,
                          WisdomGenerationReport& report,
                          std::string& error) {
    error.clear();
    report = WisdomGenerationReport{};
    if (output_path.empty()) {
        error = "output path is empty";
        return false;
    }
    if (size < 1 || (size & (size - 1)) != 0) {
        error = "size must be a positive power of two";
        return false;
    }
    unsigned flags = 0;
    if (planner == "measure")
        flags = FFTW_MEASURE;
    else if (planner == "patient")
        flags = FFTW_PATIENT;
    else {
        error = "planner must be measure or patient";
        return false;
    }
    if (!safe_metadata_value(library_id)) {
        error = "library_id must be non-empty, single-line and at most 4096 bytes";
        return false;
    }
    if (!safe_metadata_value(fftw_version) || !safe_metadata_value(fftw_cc)) {
        error = "FFTW version/compiler identity cannot be represented safely";
        return false;
    }

    fftw_forget_wisdom();
    RealPtr input;
    ComplexPtr output;
    ComplexPtr inout;
    PlanPtr plan;
    WisdomProblem problem{transform, size, 0, 0};

    if (transform == WisdomTransform::r2c) {
        input.reset(fftw_alloc_real(size));
        output.reset(fftw_alloc_complex(size / 2 + 1));
        if (!input || !output) {
            error = "FFTW buffer allocation failed";
            return false;
        }
        problem.input_alignment = fftw_alignment_of(input.get());
        problem.output_alignment =
            fftw_alignment_of(reinterpret_cast<double*>(output.get()));
        const auto start = std::chrono::steady_clock::now();
        plan.reset(fftw_plan_dft_r2c_1d(size, input.get(), output.get(), flags));
        report.planning_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                .count();
        if (!plan) {
            error = "offline R2C planning failed";
            return false;
        }
        // Planning (especially MEASURE/PATIENT) may overwrite buffers. Data is
        // deliberately initialized only afterwards and is unrelated to PCM.
        std::fill_n(input.get(), size, 0.0);
        fftw_execute_dft_r2c(plan.get(), input.get(), output.get());
    } else {
        inout.reset(fftw_alloc_complex(size));
        if (!inout) {
            error = "FFTW buffer allocation failed";
            return false;
        }
        problem.input_alignment = problem.output_alignment =
            fftw_alignment_of(reinterpret_cast<double*>(inout.get()));
        const auto start = std::chrono::steady_clock::now();
        plan.reset(fftw_plan_dft_1d(size, inout.get(), inout.get(), FFTW_FORWARD,
                                    flags));
        report.planning_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                .count();
        if (!plan) {
            error = "offline C2C planning failed";
            return false;
        }
        std::fill_n(reinterpret_cast<double*>(inout.get()), 2 * size, 0.0);
        fftw_execute_dft(plan.get(), inout.get(), inout.get());
    }

    plan.reset();
    StringPtr exported(fftw_export_wisdom_to_string());
    if (!exported) {
        error = "FFTW wisdom export failed";
        return false;
    }
    const std::string payload(exported.get());
    if (payload.empty()) {
        error = "FFTW exported empty wisdom";
        return false;
    }
    report.wisdom_bytes = payload.size();

    std::ostringstream header;
    header << kMagic << '\n'
           << "fftw_version=" << fftw_version << '\n'
           << "fftw_cc=" << fftw_cc << '\n'
           << "precision=double\n"
           << "library_id=" << library_id << '\n'
           << "cpu=" << cpu_identity() << '\n'
           << "transform=" << wisdom_transform_name(transform) << '\n'
           << "size=" << size << '\n'
           << "placement=" << placement_name(transform) << '\n'
           << "input_alignment=" << problem.input_alignment << '\n'
           << "output_alignment=" << problem.output_alignment << '\n'
           << "planner=" << planner << '\n'
           << "payload_bytes=" << payload.size() << '\n'
           << "payload_fnv1a64=" << hex64(fnv1a64(payload)) << "\n\n";

    // The generator is serial by contract. A fixed sibling temporary prevents
    // exposing a partially written destination; concurrent writers to the same
    // output path are intentionally unsupported.
    const std::string temporary = output_path + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "cannot open temporary output: " + temporary;
            return false;
        }
        const std::string header_text = header.str();
        out.write(header_text.data(),
                  static_cast<std::streamsize>(header_text.size()));
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        out.close();
        if (!out) {
            std::remove(temporary.c_str());
            error = "failed while writing temporary output";
            return false;
        }
    }
    if (std::rename(temporary.c_str(), output_path.c_str()) != 0) {
        std::remove(temporary.c_str());
        error = "cannot rename temporary output to destination";
        return false;
    }
    return true;
}
#endif  // MELSPEC_FFTW_WISDOM_GENERATOR

}  // namespace melspec_fftw
