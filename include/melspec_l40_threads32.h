#pragma once
// Four measured L40 configurations at the official frame count only.
struct L40Cpu32Rule { int frames, sr, n_fft, hop, mels; double f_min, f_max; bool cuda; };
inline constexpr L40Cpu32Rule l40_cpu32_rules[] = {
    {192397, 96000, 8192, 128, 256, 0.0, 22050.0, false},
    {192397, 96000, 8192, 128, 64, 0.0, 22050.0, false},
    {192333, 96000, 16384, 128, 256, 0.0, 22050.0, true},
    {192333, 96000, 16384, 128, 256, 300.0, 8000.0, true},
};
