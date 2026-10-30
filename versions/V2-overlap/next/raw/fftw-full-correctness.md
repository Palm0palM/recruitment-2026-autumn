# bench 2026-10-30 14:58 tmp/binaries/melspec-overlap-fftw

case        |  n_fft |  hop | mels | f_min |  f_max |   max|err| | pass |  time(s) |  librosa | speedup
-------------------------------------------------------------------------------------------------------
default | n_fft=8192 hop=128 mels=256 | 0.955s | err=0.03125 | ok=True | librosa=None
small_mels | n_fft=8192 hop=128 mels=64 | 0.627s | err=0.01171875 | ok=True | librosa=None
large_fft | n_fft=16384 hop=128 mels=256 | 1.359s | err=0.125 | ok=True | librosa=None
narrow_band | n_fft=16384 hop=128 mels=256 | 1.319s | err=0.0625 | ok=True | librosa=None
-------------------------------------------------------------------------------------------------------
