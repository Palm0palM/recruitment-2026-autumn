#!/usr/bin/env python3
"""Small deterministic MPI correctness matrix, run only inside Slurm.

Uses a shared-home temporary directory because every rank must see test input.
Not a benchmark and not the official validator.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MPIEXEC = Path(os.environ.get("OPENMPI_HOME", "/cache/toolchain/openmpi-4.1.6")) / "bin/mpirun"
DRIVER = ROOT / "tmp/bin/mpi-melspec"
REFERENCE = ROOT / "tmp/bin/a2-estimate"


def launch(ranks):
    nodes = int(os.environ.get("SLURM_NNODES", "1"))
    ranks_per_node = (ranks + nodes - 1) // nodes
    cpus_per_rank = 16 // ranks_per_node
    return [str(MPIEXEC), "-np", str(ranks),
            "--map-by", f"ppr:{ranks_per_node}:node:pe={cpus_per_rank}",
            "--bind-to", "core", str(DRIVER)]


def main():
    rng = np.random.default_rng(20261031)
    random_pcm = rng.integers(-32768, 32768, 192000, dtype=np.int16)
    cases = [
        ("random-default", random_pcm, dict(sr=96000, n_fft=8192, hop_length=128, n_mels=256, f_min=0, f_max=22050)),
        ("random-large", random_pcm, dict(sr=96000, n_fft=16384, hop_length=128, n_mels=64, f_min=300, f_max=8000)),
        ("one-frame", random_pcm[:8192], dict(sr=96000, n_fft=8192, hop_length=128, n_mels=16, f_min=0, f_max=22050)),
        ("two-frames", random_pcm[:8192 + 127], dict(sr=96000, n_fft=8192, hop_length=127, n_mels=16, f_min=0, f_max=48000)),
        ("hop-greater-than-fft", random_pcm[:8192], dict(sr=96000, n_fft=256, hop_length=512, n_mels=16, f_min=0, f_max=48000)),
        ("fft-one", random_pcm[:2048], dict(sr=96000, n_fft=1, hop_length=127, n_mels=4, f_min=0, f_max=48000)),
        ("fft-eight", random_pcm[:4096], dict(sr=96000, n_fft=8, hop_length=127, n_mels=4, f_min=0, f_max=48000)),
    ]
    work = ROOT / "tmp/mpi-check"
    work.mkdir(exist_ok=True)
    passed = 0
    with tempfile.TemporaryDirectory(dir=work) as td:
        td = Path(td)
        src, ref, out = td / "input.pcm", td / "reference.bin", td / "mpi.bin"
        for name, pcm, params in cases:
            pcm.astype("<i2").tofile(src)
            flags = [s for k, v in params.items() for s in (f"--{k}", str(v))]
            subprocess.run([str(REFERENCE), str(src), str(ref), *flags], check=True, capture_output=True)
            reference = ref.read_bytes()
            for ranks in (1, 2, 4):
                for mode in ("bcast", "scatter", "shared"):
                    env = dict(os.environ, MELSPEC_MPI_IO=mode)
                    result = subprocess.run([*launch(ranks), str(src), str(out), *flags],
                                            check=True, capture_output=True, text=True, env=env)
                    got = out.read_bytes()
                    exact = got == reference
                    a = np.frombuffer(reference, dtype="<f4")
                    b = np.frombuffer(got, dtype="<f4")
                    valid = a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-3, atol=1e-4))
                    if not exact or not valid:
                        error = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else None
                        raise SystemExit(f"FAIL {name} ranks={ranks} io={mode} exact={exact} valid={valid} err={error}\n{result.stdout}\n{result.stderr}")
                    passed += 1
                    print(f"MPI_CHECK {name} ranks={ranks} io={mode} exact=True PASS", flush=True)
            print(result.stdout, end="", flush=True)
        for length in (0, 1, 2047):
            np.zeros(length, dtype="<i2").tofile(src)
            result = subprocess.run([*launch(2), str(src), str(out)],
                                    capture_output=True, text=True,
                                    env=dict(os.environ, MELSPEC_MPI_IO="scatter"))
            if result.returncode == 0:
                raise SystemExit(f"FAIL short input N={length}: MPI returned success")
            print(f"MPI_CHECK short-N{length} rejected PASS", flush=True)
    print(f"MPI_CHECK total={passed} exact outputs + 3 rejected short inputs: PASS", flush=True)


if __name__ == "__main__":
    main()
