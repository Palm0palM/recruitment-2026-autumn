#!/bin/bash
#SBATCH -J cpu-simd-lab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/pipeline-%j.out
#SBATCH --error=tmp/logs/pipeline-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source tmp/session-env.sh
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "job=$SLURM_JOB_ID node=$(hostname) mode=${MODE:-profile}"
grep Cpus_allowed_list /proc/self/status
lscpu
g++ --version
sha256sum src/main.cpp src/melspec.cpp validate.py tmp/control/melspec.cpp
scratch=$(mktemp -d "/tmp/cpu-simd-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
mkdir -p tmp/bin
flags=(-std=c++17 -O3 -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16)
if [ "${MODE:-profile}" = profile ]; then
    "$PY" -u scripts/cpu_pipeline_profile.py --source tmp/control/melspec.cpp --runs 3
    cp tmp/pipeline-profile/control/results.json "tmp/logs/profile-$SLURM_JOB_ID.json"
    g++ "${flags[@]}" src/main.cpp tmp/control/melspec.cpp -o tmp/bin/frozen-b16
    "$PY" -u scripts/bench.py tmp/bin/frozen-b16 --runs 3 --no-librosa
else
    echo 'Set MODE=profile until a separately reviewed experiment is available.' >&2
    exit 2
fi
