#!/bin/bash
#SBATCH -J melspec-overlap-next
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=tmp/logs/next-bench-%j.out
#SBATCH --error=tmp/logs/next-bench-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs tmp/binaries tmp/src
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY="${PY:-$HOME/venv/bin/python}"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/melspec-next-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"

echo "== job $SLURM_JOB_ID on $(hostname) =="
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E 'Model name|^CPU\(s\)'
g++ --version | head -1

if [ ! -f tmp/fftw/lib/libfftw3.a ]; then
  echo '== build private FFTW =='
  rm -rf tmp/src/fftw-3.3.10
  tar -xzf tmp/fftw-3.3.10.tar.gz -C tmp/src
  cd tmp/src/fftw-3.3.10
  ./configure --prefix="$OLDPWD/tmp/fftw" --disable-shared --enable-static \
    --enable-avx2 --enable-avx512 CFLAGS='-O3 -march=native'
  make -j 16
  make install
  cd "$OLDPWD"
fi
export FFTW_INC="$PWD/tmp/fftw/include"
export FFTW_LIB="$PWD/tmp/fftw/lib"

make -B -j 16
cp melspec tmp/binaries/melspec-v1
g++ -std=c++17 -O3 -march=native -pthread -Iinclude src/main.cpp \
  versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp -o tmp/binaries/melspec-overlap-half-delta
g++ -std=c++17 -O3 -march=native -pthread -Iinclude src/main.cpp \
  versions/V2-overlap/next/melspec_overlap_pruned_power_prototype.cpp -o tmp/binaries/melspec-overlap-pruned-power
for variant in fftw fftw_pruned; do
  g++ -std=c++17 -O3 -march=native -pthread -Iinclude -I"$FFTW_INC" src/main.cpp \
    "versions/V2-overlap/next/melspec_overlap_${variant}_prototype.cpp" \
    -L"$FFTW_LIB" -lfftw3 -lm -o "tmp/binaries/melspec-overlap-${variant/_/-}"
done
g++ -std=c++17 -O3 -march=native -DHAVE_FFTW -I"$FFTW_INC" \
  versions/V2-overlap/next/microbench_fft128.cpp -L"$FFTW_LIB" -lfftw3 -lm \
  -o tmp/binaries/microbench-fft128

sha256sum src/melspec.cpp versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp \
  versions/V2-overlap/next/*prototype.cpp tmp/binaries/melspec-v1 \
  tmp/binaries/melspec-overlap-half-delta tmp/binaries/melspec-overlap-pruned-power \
  tmp/binaries/melspec-overlap-fftw tmp/binaries/melspec-overlap-fftw-pruned

echo '== 128-point microbenchmark =='
tmp/binaries/microbench-fft128 2000 | tee "tmp/logs/microbench-${SLURM_JOB_ID}.txt"

for variant in v1 half-delta pruned-power fftw fftw-pruned; do
  if [ "$variant" = v1 ]; then exe=tmp/binaries/melspec-v1; else exe="tmp/binaries/melspec-overlap-${variant}"; fi
  echo "== official validate.py: $variant =="
  "$PY" -u validate.py "$exe" --runs 3 \
    | tee "tmp/logs/validate-${variant}-${SLURM_JOB_ID}.txt"
  grep -q 'Result: 4/4 PASSED' "tmp/logs/validate-${variant}-${SLURM_JOB_ID}.txt"
done

for case in default large_fft narrow_band; do
  echo "== interleaved candidates: $case =="
  "$PY" -u versions/V2-overlap/next/ab_next.py --case "$case" --rounds 3 \
    --programs v1,half_delta,pruned,fftw,fftw_pruned \
    --out "tmp/logs/ab-${case}-${SLURM_JOB_ID}.json"
done
