#!/usr/bin/env bash
#SBATCH --job-name=melspec-library
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=tmp/logs/library-%j.out
#SBATCH --error=tmp/logs/library-%j.err

# Python library-first stage. Runs on a compute node only; no --mem (cluster bug).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
PY="${PY:-$HOME/venv/bin/python}"
mkdir -p tmp/logs tmp/results/stats tmp/bin tmp/build tmp/scratch
echo "job=$SLURM_JOB_ID node=$(hostname) mode=${MODE:-python}"
echo "base=${BASE_SHA:-b4552f3962751b5a02d2aa56798907c5a1c29afc}"
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E 'Model name|^CPU\(s\)'
free -g | head -2
"$PY" -c 'import sys,numpy,scipy; print(sys.version); print(numpy.__version__, scipy.__version__)'

scratch=$(mktemp -d "/tmp/library-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"

if [ "${MODE:-python}" = baseline ]; then
  echo '== same-job FFTW A2 baseline vs Python A3 =='
  export FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11"
  JOBS=8 bash scripts/build_fftw.sh > "tmp/logs/fftw-build-${SLURM_JOB_ID}.txt" 2>&1
  make -B -j8 FFT_MODE=r2c FFTW_PLAN=estimate FFTW_PREFIX="$FFTW_PREFIX" EXEC=tmp/bin/fftw-a2
  sha256sum src/melspec.cpp src/main.cpp Makefile "$FFTW_PREFIX/lib/libfftw3.a" tmp/bin/fftw-a2
  "$PY" -u versions/V2-library/scripts/library_bench.py --runs 3 \
    --out "tmp/logs/baseline-${SLURM_JOB_ID}.json" \
    "fftw-a2=tmp/bin/fftw-a2 {pcm} {out}" \
    "a3=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a3 --block-size 128 --fft-workers 16" \
    | tee "tmp/logs/baseline-${SLURM_JOB_ID}.txt"
  exit 0
fi

if [ "${MODE:-python}" != python ]; then
  echo "unsupported MODE=${MODE}" >&2
  exit 1
fi

echo '== all-four stage benchmark: A1 reference, A2 dense, A3 sparse =='
"$PY" -u versions/V2-library/scripts/library_bench.py --runs 1 \
  --out "tmp/logs/python-bench-${SLURM_JOB_ID}.json" \
  "a1=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a1" \
  "a2-dense=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a2 --block-size 128 --fft-workers 16 --blas-threads 16" \
  "a3-sparse=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a3 --block-size 128 --fft-workers 16" \
  | tee "tmp/logs/python-bench-${SLURM_JOB_ID}.txt"

echo '== A3 batch/worker sweep on default =='
"$PY" -u versions/V2-library/scripts/library_bench.py --cases default --runs 1 \
  --out "tmp/logs/python-sweep-${SLURM_JOB_ID}.json" \
  "a3-b64-w16=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a3 --block-size 64 --fft-workers 16" \
  "a3-b128-w8=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a3 --block-size 128 --fft-workers 8" \
  "a3-b128-w16=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a3 --block-size 128 --fft-workers 16" \
  "a3-b256-w16=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a3 --block-size 256 --fft-workers 16" \
  "a2-b128-w16=$PY -B versions/V2-library/impl/library_melspec.py {pcm} {out} --mode a2 --block-size 128 --fft-workers 16 --blas-threads 16" \
  | tee "tmp/logs/python-sweep-${SLURM_JOB_ID}.txt"

echo '== complete official validate.py, A3 sparse, 3 runs =='
"$PY" -u validate.py ./versions/V2-library/impl/python_a3_sparse.py --runs 3 \
  | tee "tmp/logs/python-official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/python-official-${SLURM_JOB_ID}.txt"
