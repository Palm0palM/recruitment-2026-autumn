#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-warm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/amd-warm-%j.out
#SBATCH --error=tmp/logs/amd-warm-%j.err

# A/B: early HIP prewarm thread off vs on, profile + interleaved benchmark.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin tmp/results tmp/rocfft-cache
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY="$HOME/venv/bin/python"
echo "host=$(hostname) job=$SLURM_JOB_ID"
grep Cpus_allowed_list /proc/self/status
sha256sum src/melspec_hip.cu Makefile

programs=()
for spec in off:0 on:1; do
  IFS=: read -r name warm <<< "$spec"
  make -B -j4 HIP=1 HIP_PREWARM="$warm" EXEC="tmp/bin/bench-warm-$name" \
    > "tmp/logs/build-warm-$name.txt" 2>&1
  make -B -j4 HIP=1 HIP_PROFILE=1 HIP_PREWARM="$warm" EXEC="tmp/bin/prof-warm-$name" \
    >> "tmp/logs/build-warm-$name.txt" 2>&1
  echo "== profile prewarm=$name ==" | tee -a "tmp/logs/warm-profile-${SLURM_JOB_ID}.txt"
  "$PY" -u scripts/cuda_profile.py "tmp/bin/prof-warm-$name" \
    | tee -a "tmp/logs/warm-profile-${SLURM_JOB_ID}.txt"
  programs+=("$name=tmp/bin/bench-warm-$name")
done

if [ "${BENCH:-1}" = 1 ]; then
  scratch=$(mktemp -d "/tmp/melspec-warm-${SLURM_JOB_ID}-XXXXXX")
  trap 'rm -rf -- "$scratch"' EXIT
  export TMPDIR="$scratch"
  "$PY" -u scripts/cuda_bench.py --programs "${programs[@]}" --runs "${RUNS:-5}" \
    --out "tmp/results/amd-warm-${SLURM_JOB_ID}.json" \
    | tee "tmp/logs/warm-bench-${SLURM_JOB_ID}.txt"
fi
