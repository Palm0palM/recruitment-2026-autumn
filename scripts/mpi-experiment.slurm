#!/usr/bin/env bash
#SBATCH --job-name=melspec-mpi-spike
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode[0-3]
#SBATCH --time=00:30:00
#SBATCH --output=tmp/logs/mpi-experiment-%j.out
#SBATCH --error=tmp/logs/mpi-experiment-%j.err

# MODE=check | bench | official. All work is inside a Slurm allocation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENMPI_HOME=/cache/toolchain/openmpi-4.1.6
UCX_HOME=/cache/toolchain/ucx-1.18.1
export PATH="$OPENMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib:$UCX_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$OPENMPI_HOME" OPAL_LIBDIR="$OPENMPI_HOME/lib"
export OMPI_MCA_rmaps_base_oversubscribe=1
export FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
export MELSPEC_MPI_CPUS=16
PY="${PY:-$HOME/venv/bin/python}"
MODE="${MODE:-check}"
echo "job=$SLURM_JOB_ID mode=$MODE nodes=$SLURM_JOB_NODELIST"
scontrol show hostname "$SLURM_JOB_NODELIST"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD
  git status --short
else
  echo 'source snapshot: no Git metadata (recorded by the local orchestrator)'
fi
"$OPENMPI_HOME/bin/ompi_info" --version
srun --mpi=list 2>&1 || true
mkdir -p tmp/logs tmp/bin tmp/mpi-check

JOBS=8 bash scripts/build_fftw.sh > "tmp/logs/fftw-build-${SLURM_JOB_ID}.txt" 2>&1
make -B -j4 FFT_MODE=r2c FFTW_PLAN=estimate EXEC=tmp/bin/a2-estimate > "tmp/logs/a2-build-${SLURM_JOB_ID}.txt" 2>&1
g++ -std=c++17 -O3 -march=native -pthread -Iinclude -I"$FFTW_PREFIX/include" \
  -I"$OPENMPI_HOME/include" scripts/mpi_melspec.cpp src/melspec.cpp \
  "$FFTW_PREFIX/lib/libfftw3.a" -L"$OPENMPI_HOME/lib" \
  -Wl,-rpath,"$OPENMPI_HOME/lib" -lmpi -lm -o tmp/bin/mpi-melspec
sha256sum src/main.cpp src/melspec.cpp include/fftw_support.h scripts/mpi_melspec.cpp \
  "$FFTW_PREFIX/lib/libfftw3.a" tmp/bin/a2-estimate tmp/bin/mpi-melspec
ldd tmp/bin/mpi-melspec | grep -E 'not found|mpi|open-rte|open-pal|fftw' || true
scratch=$(mktemp -d "/tmp/melspec-mpi-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"

case "$MODE" in
  check)
    "$PY" -u scripts/mpi_check.py
    ;;
  bench)
    "$PY" -u scripts/mpi_bench.py --runs "${RUNS:-3}" --out "tmp/logs/mpi-bench-${SLURM_JOB_ID}.json"
    ;;
  official)
    export MELSPEC_MPI_RANKS="${MELSPEC_MPI_RANKS:-2}" MELSPEC_MPI_IO="${MELSPEC_MPI_IO:-scatter}"
    "$PY" -u validate.py ./scripts/mpi-official-wrapper.sh --runs "${RUNS:-3}" | tee "tmp/logs/mpi-official-${SLURM_JOB_ID}.txt"
    grep -q 'Result: 4/4 PASSED' "tmp/logs/mpi-official-${SLURM_JOB_ID}.txt"
    ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
