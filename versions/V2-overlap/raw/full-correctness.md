# bench 2026-10-30 13:22 ./melspec

case        |  n_fft |  hop | mels | f_min |  f_max |   max|err| | pass |  time(s) |  librosa | speedup
-------------------------------------------------------------------------------------------------------
default | n_fft=8192 hop=128 mels=256 | 1.482s | err=0.03125 | ok=True | librosa=None
small_mels | n_fft=8192 hop=128 mels=64 | 1.311s | err=0.01171875 | ok=True | librosa=None
large_fft | n_fft=16384 hop=128 mels=256 | 3.803s | err=0.125 | ok=True | librosa=None
narrow_band | n_fft=16384 hop=128 mels=256 | 5.074s | err=0.0625 | ok=True | librosa=None
-------------------------------------------------------------------------------------------------------
