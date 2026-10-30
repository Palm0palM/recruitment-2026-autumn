# bench 2026-10-30 14:47 tmp/binaries/melspec-overlap-pruned-power

case        |  n_fft |  hop | mels | f_min |  f_max |   max|err| | pass |  time(s) |  librosa | speedup
-------------------------------------------------------------------------------------------------------
default | n_fft=8192 hop=128 mels=256 | 1.100s | err=0.03125 | ok=True | librosa=None
small_mels | n_fft=8192 hop=128 mels=64 | 0.793s | err=0.01171875 | ok=True | librosa=None
large_fft | n_fft=16384 hop=128 mels=256 | 1.619s | err=0.125 | ok=True | librosa=None
narrow_band | n_fft=16384 hop=128 mels=256 | 1.510s | err=0.0625 | ok=True | librosa=None
-------------------------------------------------------------------------------------------------------
