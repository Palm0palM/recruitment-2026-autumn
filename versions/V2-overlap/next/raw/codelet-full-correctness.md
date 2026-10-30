# bench 2026-10-30 15:46 tmp/binaries/melspec-overlap-codelet

case        |  n_fft |  hop | mels | f_min |  f_max |   max|err| | pass |  time(s) |  librosa | speedup
-------------------------------------------------------------------------------------------------------
default | n_fft=8192 hop=128 mels=256 | 1.330s | err=0.03125 | ok=True | librosa=None
small_mels | n_fft=8192 hop=128 mels=64 | 0.974s | err=0.01171875 | ok=True | librosa=None
large_fft | n_fft=16384 hop=128 mels=256 | 2.699s | err=0.125 | ok=True | librosa=None
narrow_band | n_fft=16384 hop=128 mels=256 | 2.345s | err=0.0625 | ok=True | librosa=None
-------------------------------------------------------------------------------------------------------
