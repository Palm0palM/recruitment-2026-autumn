// 内部 FFT microbenchmark；不作为正式 Melspec 程序链接。
#include "../src/melspec.cpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    int n = 4096;
    int frames = 20000;
    int repeats = 5;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 >= argc) return 2;
        int value = std::atoi(argv[++i]);
        if (key == "--n") n = value;
        else if (key == "--frames") frames = value;
        else if (key == "--repeats") repeats = value;
        else return 2;
    }
    if (n < 1 || (n & (n - 1)) != 0 || frames < 1 || repeats < 1) return 2;
    FFTPlan plan(n);
    ComplexBuffer a(n);
    std::vector<double> input(n);
    for (int i = 0; i < n; ++i)
        input[i] = 0.25 + 0.5 * std::sin(0.017 * i) + 0.1 * std::cos(0.071 * i);
    using Clock = std::chrono::steady_clock;
    volatile double checksum = 0.0;
    std::cout << "n,repeat,pack_ns_per_frame,fft_ns_per_frame,total_ns_per_frame,checksum\n";
    for (int rep = 0; rep < repeats; ++rep) {
        auto begin = Clock::now();
        double pack_seconds = 0.0;
        for (int frame = 0; frame < frames; ++frame) {
            auto pack_begin = Clock::now();
            for (int i = 0; i < n; ++i) {
                const double value = input[(i + frame) % n] * (1.0 + 1e-9 * frame);
                a.re[plan.reversed[i]] = value;
                a.im[plan.reversed[i]] = value * 0.125;
            }
            auto pack_end = Clock::now();
            pack_seconds += std::chrono::duration<double>(pack_end - pack_begin).count();
            plan.apply_permuted(a);
            checksum += a.re[frame % n] + a.im[(frame * 7) % n];
        }
        auto end = Clock::now();
        const double total = std::chrono::duration<double>(end - begin).count();
        const double fft = total - pack_seconds;
        std::cout << n << ',' << rep << ',' << std::setprecision(12)
                  << pack_seconds * 1e9 / frames << ',' << fft * 1e9 / frames << ','
                  << total * 1e9 / frames << ',' << checksum << '\n';
    }
}
