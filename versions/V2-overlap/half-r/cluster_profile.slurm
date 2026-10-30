#!/bin/bash
#SBATCH -J melspec-v2-overlap-profile
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=tmp/logs/cluster-half-r-profile-%j.out
#SBATCH --error=tmp/logs/cluster-half-r-profile-%j.err

# perf_event_paranoid=4：使用独立 chrono 插桩副本，不修改正式源码。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
PY="${PY:-$HOME/venv/bin/python}"
echo "== profile job $SLURM_JOB_ID on $(hostname) =="
grep Cpus_allowed_list /proc/self/status
sha256sum versions/V2-overlap/melspec_overlap_half_prototype.cpp \
  versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp \
  versions/V2-overlap/stage_profile.py

for variant in half half-delta; do
  if [ "$variant" = half ]; then
    source=versions/V2-overlap/melspec_overlap_half_prototype.cpp
  else
    source=versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp
  fi
  for case in default large_fft; do
    echo "== $variant / $case =="
    "$PY" -u versions/V2-overlap/stage_profile.py --case "$case" --source "$source" \
      --out "tmp/logs/stage-${variant}-${case}-${SLURM_JOB_ID}.json"
  done
done
