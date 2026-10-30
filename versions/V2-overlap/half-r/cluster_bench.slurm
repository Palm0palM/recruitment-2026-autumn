#!/bin/bash
#SBATCH -J melspec-v2-overlap-bench
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=tmp/logs/cluster-half-r-bench-%j.out
#SBATCH --error=tmp/logs/cluster-half-r-bench-%j.err

# 同节点依次测 V1、half、half+delta；不要写 --mem。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs tmp/binaries
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
PY="${PY:-$HOME/venv/bin/python}"

scratch=$(mktemp -d "${TMPDIR:-/tmp}/melspec-overlap-bench-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"

echo "== job $SLURM_JOB_ID on $(hostname) =="
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E 'Model name|^CPU\(s\)'
g++ --version | head -1
"$PY" -c 'import sys,numpy,librosa; print(sys.version); print("numpy",numpy.__version__,"librosa",librosa.__version__)'

echo '== build V1 and two independent experimental sources =='
make -B -j 16
cp melspec tmp/binaries/melspec-v1
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  src/main.cpp versions/V2-overlap/melspec_overlap_half_prototype.cpp \
  -o tmp/binaries/melspec-half
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  src/main.cpp versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp \
  -o tmp/binaries/melspec-half-delta
sha256sum src/melspec.cpp src/main.cpp validate.py \
  versions/V2-overlap/melspec_overlap_half_prototype.cpp \
  versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp \
  tmp/binaries/melspec-v1 tmp/binaries/melspec-half tmp/binaries/melspec-half-delta

for variant in v1 half half-delta; do
  exe="tmp/binaries/melspec-${variant}"
  log="tmp/logs/validate-${variant}-${SLURM_JOB_ID}.txt"
  echo "== official validate.py: $variant =="
  "$PY" -u validate.py "$exe" --runs 3 | tee "$log"
  grep -q 'Result: 4/4 PASSED' "$log"
done
