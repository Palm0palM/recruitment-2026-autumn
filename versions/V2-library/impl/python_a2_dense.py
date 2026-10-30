#!/usr/bin/env python3
"""Official-CLI wrapper: blocked NumPy/SciPy + dense BLAS Mel, 16 FFT workers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_melspec import main

if __name__ == "__main__":
    sys.argv += ["--mode", "a2", "--block-size", "128", "--fft-workers", "16",
                 "--blas-threads", "16"]
    main()
