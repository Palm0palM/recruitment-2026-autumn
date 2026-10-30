# bench 2026-10-30 13:48 tmp/binaries/melspec-overlap-half

case        |  n_fft |  hop | mels | f_min |  f_max |   max|err| | pass |  time(s) |  librosa | speedup
-------------------------------------------------------------------------------------------------------
default | n_fft=8192 hop=128 mels=256 | 1.549s | err=0.03125 | ok=True | librosa=None
small_mels | n_fft=8192 hop=128 mels=64 | 0.871s | err=0.01171875 | ok=True | librosa=None
large_fft | n_fft=16384 hop=128 mels=256 | 1.964s | err=0.125 | ok=True | librosa=None
narrow_band | n_fft=16384 hop=128 mels=256 | 1.884s | err=0.0625 | ok=True | librosa=None
-------------------------------------------------------------------------------------------------------
