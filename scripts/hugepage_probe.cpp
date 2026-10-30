// Does MADV_HUGEPAGE make the 394 MB output zero-initialisation cheap?
#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static void print_setting(const char* path) {
    std::ifstream f(path);
    std::string line;
    if (std::getline(f, line)) std::printf("%s=%s\n", path, line.c_str());
}

static void huge_pages(const char* tag) {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("AnonHugePages", 0) == 0)
            std::printf("%s %s\n", tag, line.c_str());
}

static const size_t kOut = 256ull * 192397;

int main() {
    print_setting("/sys/kernel/mm/transparent_hugepage/enabled");
    print_setting("/sys/kernel/mm/transparent_hugepage/defrag");

    {
        auto t = Clock::now();
        std::vector<double> out(kOut);
        std::printf("plain_resize_ms=%.3f size=%zu\n", ms_since(t), out.size());
        huge_pages("plain");
    }

    {
        std::vector<double> out;
        auto t0 = Clock::now();
        out.reserve(kOut);
        const double reserve_ms = ms_since(t0);
        t0 = Clock::now();
        const int rc = madvise(out.data(), kOut * sizeof(double), MADV_HUGEPAGE);
        const double advise_ms = ms_since(t0);
        t0 = Clock::now();
        out.resize(kOut);
        std::printf("huge_reserve_ms=%.3f advise_ms=%.3f (rc=%d) resize_ms=%.3f\n",
                    reserve_ms, advise_ms, rc, ms_since(t0));
        huge_pages("huge");
    }

    {
        // Real pipeline shape: resize in a worker thread while the main thread
        // does ~50 ms of unrelated host work, then join.
        std::vector<double> out;
        out.reserve(kOut);
        madvise(out.data(), kOut * sizeof(double), MADV_HUGEPAGE);
        auto t = Clock::now();
        out.resize(kOut);
        std::printf("huge_resize_again_ms=%.3f\n", ms_since(t));
    }
    return 0;
}
