#!/bin/bash
#SBATCH -J melspec-overlap-next-profile
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=tmp/logs/next-profile-%j.out
#SBATCH --error=tmp/logs/next-profile-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs tmp/scratch
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
PY="${PY:-$HOME/venv/bin/python}"
echo "== next profile job $SLURM_JOB_ID on $(hostname) =="
grep Cpus_allowed_list /proc/self/status
for case in default narrow_band; do
  for variant in delta pruned; do
    if [ "$variant" = delta ]; then
      source=versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp
    else
      source=versions/V2-overlap/next/melspec_overlap_pruned_power_prototype.cpp
    fi
    "$PY" -u versions/V2-overlap/stage_profile.py --case "$case" --source "$source" \
      --out "tmp/logs/stage-${variant}-${case}-${SLURM_JOB_ID}.json"
  done
done
