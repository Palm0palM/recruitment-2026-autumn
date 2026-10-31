#pragma once
// Four measured configurations only; not a general 64-thread cost model.
// Generated from threads64-ab-461.json
struct L40Threads64Rule {
    int sr, n_fft, hop, mels;
    double f_min, f_max;
    bool cuda;
};
inline constexpr L40Threads64Rule l40_threads64_rules[] = {
    // official-default: CPU64 0.676118s, CUDA 0.828187s
    {96000, 8192, 128, 256, 0.0, 22050.0, false},
    // official-small_mels: CPU64 0.431342s, CUDA 0.565689s
    {96000, 8192, 128, 64, 0.0, 22050.0, false},
    // official-large_fft: CPU64 0.881795s, CUDA 0.833695s
    {96000, 16384, 128, 256, 0.0, 22050.0, true},
    // official-narrow_band: CPU64 0.836275s, CUDA 0.840109s
    {96000, 16384, 128, 256, 300.0, 8000.0, false},
};
