#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/amd-sweep-%j.out
#SBATCH --error=tmp/logs/amd-sweep-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin tmp/results tmp/rocfft-cache
source tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
export NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
PY="$HOME/venv/bin/python"
git rev-parse HEAD 2>/dev/null || echo "source=rsync-snapshot"
grep Cpus_allowed_list /proc/self/status
programs=()
for spec in g2:0:512:512:1 pin:1:512:512:1 pin_b256:1:512:256:1 pin_noasync:1:512:256:0; do
  IFS=: read -r name pin batch large async <<< "$spec"
  make -B -j4 HIP=1 HIP_PINNED_INPUT="$pin" HIP_BATCH="$batch" \
    HIP_BATCH_LARGE="$large" HIP_ASYNC_OUTPUT="$async" EXEC="tmp/bin/hip-$name"
  programs+=("$name=tmp/bin/hip-$name")
done
sha256sum src/melspec_hip.cu Makefile tmp/bin/hip-*
scratch=$(mktemp -d "/tmp/melspec-amd-sweep-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/cuda_bench.py --programs "${programs[@]}" --runs 5 \
  --out "tmp/results/amd-sweep-${SLURM_JOB_ID}.json"
