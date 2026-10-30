// Diagnostic only: compare actual C++ prefix-moment projection with direct dot.
#define MELSPEC_MOMENTS 1
#include "../src/melspec.cpp"

#include <iomanip>
#include <iostream>

int main() {
    constexpr int n_fft = 8192;
    constexpr int sr = 96000;
    constexpr int n_mels = 256;
    constexpr int n_bins = n_fft / 2 + 1;
    auto direct = build_compressed_mel_filterbank(sr, n_fft, n_mels, 0, 22050);
    auto moment = build_moment_mel_filterbank(sr, n_fft, n_mels, 0, 22050);
    std::vector<std::pair<const char*, std::vector<double>>> cases;
    cases.emplace_back("zero", std::vector<double>(n_bins, 0.0));
    std::vector<double> smooth(n_bins), dynamic(n_bins, 0.0), impulse(n_bins, 0.0);
    for (int k = 0; k < n_bins; ++k) {
        smooth[k] = 0.5 + 0.25 * std::sin(k * 0.173);
        dynamic[k] = 1e-6 * (1 + k % 7);
    }
    dynamic[0] = 1e12;
    impulse[n_bins - 1] = 1.0;
    cases.emplace_back("smooth", std::move(smooth));
    cases.emplace_back("dynamic_prefix", std::move(dynamic));
    cases.emplace_back("last_bin_impulse", std::move(impulse));

    for (const auto& item : cases) {
        const auto& power = item.second;
        std::vector<double> p(n_bins + 1), q(n_bins + 1);
        for (int k = 0; k < n_bins; ++k) {
            p[k + 1] = p[k] + power[k];
            q[k + 1] = q[k] + k * power[k];
        }
        double max_abs = 0, max_rel = 0;
        int negative = 0, nonfinite = 0;
        for (int m = 0; m < n_mels; ++m) {
            double ref = 0;
            const auto& d = direct[m];
            for (int k = d.first; k < d.last; ++k)
                ref += d.weights[k - d.first] * power[k];
            const auto& f = moment[m];
            double got = 0;
            if (f.first != f.last) {
                got = f.a_left * (q[f.center] - q[f.first]) +
                      f.b_left * (p[f.center] - p[f.first]) +
                      f.a_right * (q[f.last] - q[f.center]) +
                      f.b_right * (p[f.last] - p[f.center]);
            }
            const double error = std::abs(got - ref);
            max_abs = std::max(max_abs, error);
            if (std::abs(ref) > 1e-12)
                max_rel = std::max(max_rel, error / std::abs(ref));
            negative += got < 0;
            nonfinite += !std::isfinite(got);
        }
        std::cout << std::setprecision(17)
                  << "{\"case\":\"" << item.first << "\",\"max_abs\":" << max_abs
                  << ",\"max_rel\":" << max_rel << ",\"negative\":" << negative
                  << ",\"nonfinite\":" << nonfinite << "}\n";
    }
}
