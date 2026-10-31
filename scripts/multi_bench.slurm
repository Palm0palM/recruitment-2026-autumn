#!/bin/bash
#SBATCH -J melspec-multi
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/multi-%j.out
#SBATCH --error=tmp/logs/multi-%j.err

# PROGRAMS='v1=path/v1 simd=path/simd [fftw=path/fftw]'；不传 --mem。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
PY="${PY:-$HOME/venv/bin/python}"
RUNS="${RUNS:-6}"
CASES="${CASES:-default small_mels large_fft narrow_band}"
if [ -z "${PROGRAMS:-}" ]; then
  echo 'PROGRAMS 必须包含至少两个 LABEL=PATH' >&2
  exit 2
fi

echo "== job $SLURM_JOB_ID on $(hostname) =="
nproc
grep Cpus_allowed_list /proc/self/status
lscpu | grep -E 'Model name|^CPU\(s\)'
g++ --version
for spec in $PROGRAMS; do
  printf '%s: ' "${spec%%=*}"
  sha256sum "${spec#*=}"
done
scratch=$(mktemp -d "${TMPDIR:-/tmp}/melspec-multi-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
# shellcheck disable=SC2086
"$PY" -u -B scripts/multi_bench.py $PROGRAMS --cases $CASES --runs "$RUNS" \
  --out "tmp/logs/multi-${SLURM_JOB_ID}.json" | tee "tmp/logs/multi-${SLURM_JOB_ID}.txt"
