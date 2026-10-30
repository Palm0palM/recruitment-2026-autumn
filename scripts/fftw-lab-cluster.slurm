#!/usr/bin/env bash
#SBATCH --job-name=fftw-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=tmp/logs/lab-%j.out
#SBATCH --error=tmp/logs/lab-%j.err
# Parent/orchestrator only. One allocated job, release builds, no phase timers.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
export FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11"
PY="$HOME/venv/bin/python"
echo "job=$SLURM_JOB_ID node=$(hostname)"
git rev-parse HEAD
git status --short
grep Cpus_allowed_list /proc/self/status
g++ --version
JOBS=8 bash scripts/build_fftw.sh
FLOAT_PREFIX="$PWD/tmp/opt/fftw-3.3.11-float"
if [ ! -f "$FLOAT_PREFIX/lib/libfftw3f.a" ]; then
  mkdir -p tmp/deps/build-fftw-float
  (
    cd tmp/deps/build-fftw-float
    CC=gcc CFLAGS='-O3 -march=native' ../src/fftw-3.3.11/configure \
      --prefix="$FLOAT_PREFIX" --enable-float --enable-static --disable-shared \
      --disable-fortran --disable-threads --disable-openmp --disable-mpi \
      --enable-sse --enable-avx --enable-avx2 --enable-avx512
    make -j8
    make install
  ) > "tmp/logs/float-build-$SLURM_JOB_ID.txt" 2>&1
fi
mkdir -p tmp/fftw-lab/orchestrator
O=tmp/fftw-lab/orchestrator
E=versions/V2-fftw/experiments
make -B -j4 FFT_MODE=r2c FFTW_PLAN=estimate EXEC="$O/a2"
for p in 0 1; do
  g++ -std=c++17 -O3 -march=native -pthread -Iinclude -I"$FLOAT_PREFIX/include" \
    -DMELSPEC_FFTWF_DOUBLE_POWER=$p src/main.cpp "$E/precision/replacement.cpp" \
    "$FLOAT_PREFIX/lib/libfftw3f.a" -lm -o "$O/float-$p"
done
g++ -std=c++17 -O3 -march=native -pthread -Iinclude -I"$E/wisdom" -I"$FFTW_PREFIX/include" \
  src/main.cpp "$E/wisdom/melspec.cpp" "$E/wisdom/fftw_wisdom.cpp" "$FFTW_PREFIX/lib/libfftw3.a" -lm -o "$O/wisdom"
g++ -std=c++17 -O3 -march=native -pthread -I"$E/wisdom" -I"$FFTW_PREFIX/include" \
  -DMELSPEC_FFTW_WISDOM_GENERATOR=1 "$E/wisdom/fftw_wisdom.cpp" "$E/wisdom/fftw_wisdom_generate.cpp" \
  "$FFTW_PREFIX/lib/libfftw3.a" -lm -o "$O/generate"
export MELSPEC_FFTW_LIBRARY_ID="$(sha256sum "$FFTW_PREFIX/lib/libfftw3.a" | cut -d' ' -f1)"
export MELSPEC_FFTW_WISDOM_DIRECTORY="$PWD/$O/wisdom-files"
mkdir -p "$MELSPEC_FFTW_WISDOM_DIRECTORY"
for n in 1 2 8 256 2048 8192 16384; do
  "$O/generate" --transform r2c --size "$n" --planner measure --library-id "$MELSPEC_FFTW_LIBRARY_ID" \
    --output "$MELSPEC_FFTW_WISDOM_DIRECTORY/r2c-$n.wisdom"
done
sha256sum "$FFTW_PREFIX/lib/libfftw3.a" "$FLOAT_PREFIX/lib/libfftw3f.a" "$O/a2" "$O/float-0" "$O/float-1" "$O/wisdom"
scratch=$(mktemp -d "/tmp/fftw-lab-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" scripts/fftw_check.py --reference "$O/a2" "$O/float-0" "$O/float-1" "$O/wisdom"
"$PY" -u scripts/fftw_bench.py --runs 3 --programs "a2=$O/a2" "float-fp=$O/float-0" \
  "float-dp=$O/float-1" "wisdom=$O/wisdom" --out "tmp/logs/lab-compare-${SLURM_JOB_ID}.json"
# Rank by geometric mean of SAME-JOB case medians; do not multiply independent speedups.
winner=$("$PY" -c 'import json,math,sys; r=json.load(open(sys.argv[1])); names={x["program"] for x in r}; print(min(names,key=lambda n:sum(math.log(x["median_s"]) for x in r if x["program"]==n)))' "tmp/logs/lab-compare-${SLURM_JOB_ID}.json")
case "$winner" in
  a2) program="$O/a2";; float-fp) program="$O/float-0";; float-dp) program="$O/float-1";; wisdom) program="$O/wisdom";; *) exit 1;;
esac
echo "Official validation winner=$winner program=$program"
"$PY" -u validate.py "./$program" --runs 3 | tee "tmp/logs/lab-official-${SLURM_JOB_ID}.txt"
grep -q 'Result: 4/4 PASSED' "tmp/logs/lab-official-${SLURM_JOB_ID}.txt"
"$PY" scripts/fftw_lab_memory.py --programs "a2=$O/a2" "float-fp=$O/float-0" "float-dp=$O/float-1" \
  "wisdom=$O/wisdom" --out "tmp/logs/lab-memory-${SLURM_JOB_ID}.json"
