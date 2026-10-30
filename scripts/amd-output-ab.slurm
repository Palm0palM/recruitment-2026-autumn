#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:25:00
#SBATCH --output=tmp/logs/amd-out-%j.out
#SBATCH --error=tmp/logs/amd-out-%j.err

# A/B: fused float output vs direct double D2H; staging row count.
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

# name:direct_output:staging_rows
VARIANTS=(${VARIANTS:-base:0:16 direct:1:16 s8:0:8})
programs=()
for spec in "${VARIANTS[@]}"; do
  IFS=: read -r name direct rows <<< "$spec"
  make -B -j4 HIP=1 HIP_DIRECT_OUTPUT="$direct" HIP_STAGING_ROWS="$rows" \
    EXEC="tmp/bin/out-$name" > "tmp/logs/build-out-$name.txt" 2>&1
  make -B -j4 HIP=1 HIP_PROFILE=1 HIP_DIRECT_OUTPUT="$direct" HIP_STAGING_ROWS="$rows" \
    EXEC="tmp/bin/outprof-$name" >> "tmp/logs/build-out-$name.txt" 2>&1
  programs+=("$name=tmp/bin/out-$name")
done

for spec in "${VARIANTS[@]}"; do
  IFS=: read -r name _ _ <<< "$spec"
  echo "== profile $name ==" | tee -a "tmp/logs/out-profile-${SLURM_JOB_ID}.txt"
  "$PY" -u scripts/cuda_profile.py "tmp/bin/outprof-$name" \
    | tee -a "tmp/logs/out-profile-${SLURM_JOB_ID}.txt"
done

scratch=$(mktemp -d "/tmp/melspec-out-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/cuda_bench.py --programs "${programs[@]}" --runs "${RUNS:-5}" \
  --out "tmp/results/amd-out-${SLURM_JOB_ID}.json" \
  | tee "tmp/logs/out-bench-${SLURM_JOB_ID}.txt"
