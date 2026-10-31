// Diagnostic only: compile ORIGINAL main.cpp with -finstrument-functions.
// Compile this file without that flag. No entrypoint or I/O replacement.
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

void read_pcm(const std::string&, std::vector<double>&);
void write_output(const std::string&, const std::vector<double>&, int, int);

static double read_start, write_start;
static double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

extern "C" void __cyg_profile_func_enter(void* fn, void*) {
    if (fn == reinterpret_cast<void*>(&read_pcm)) read_start = now();
    if (fn == reinterpret_cast<void*>(&write_output)) write_start = now();
}

extern "C" void __cyg_profile_func_exit(void* fn, void*) {
    if (fn == reinterpret_cast<void*>(&read_pcm))
        std::fprintf(stderr, "io read_s=%.9f\n", now() - read_start);
    if (fn == reinterpret_cast<void*>(&write_output))
        std::fprintf(stderr, "io write_s=%.9f\n", now() - write_start);
}
