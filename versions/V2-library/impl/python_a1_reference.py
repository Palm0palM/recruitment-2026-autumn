#!/usr/bin/env python3
"""Official-CLI wrapper: provided NumPy/SciPy reference (A1), with stage stats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_melspec import main

if __name__ == "__main__":
    sys.argv += ["--mode", "a1"]
    main()
