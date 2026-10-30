#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-hp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/amd-hp-%j.out
#SBATCH --error=tmp/logs/amd-hp-%j.err

# A/B: aligned MADV_HUGEPAGE hint off vs on, with direct double output.
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
cat /sys/kernel/mm/transparent_hugepage/enabled
sha256sum src/melspec_hip.cu Makefile

programs=()
for spec in nohp:0 hp:1; do
  IFS=: read -r name hp <<< "$spec"
  make -B -j4 HIP=1 HIP_HUGEPAGE="$hp" EXEC="tmp/bin/hp-$name" \
    > "tmp/logs/build-hp-$name.txt" 2>&1
  make -B -j4 HIP=1 HIP_PROFILE=1 HIP_HUGEPAGE="$hp" EXEC="tmp/bin/hpprof-$name" \
    >> "tmp/logs/build-hp-$name.txt" 2>&1
  programs+=("$name=tmp/bin/hp-$name")
done

for spec in nohp:0 hp:1; do
  IFS=: read -r name _ <<< "$spec"
  echo "== profile $name ==" | tee -a "tmp/logs/hp-profile-${SLURM_JOB_ID}.txt"
  "$PY" -u scripts/cuda_profile.py "tmp/bin/hpprof-$name" \
    | tee -a "tmp/logs/hp-profile-${SLURM_JOB_ID}.txt"
done

scratch=$(mktemp -d "/tmp/melspec-hp-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/cuda_bench.py --programs "${programs[@]}" --runs "${RUNS:-5}" \
  --out "tmp/results/amd-hp-${SLURM_JOB_ID}.json" \
  | tee "tmp/logs/hp-bench-${SLURM_JOB_ID}.txt"
