#!/bin/bash
#SBATCH -J melspec-profile
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/profile-%j.out
#SBATCH --error=tmp/logs/profile-%j.err

# perf_event_paranoid=4，改用诊断副本阶段计时 + /usr/bin/time -v。
# 提交前 mkdir -p tmp/logs；CASE=all sbatch scripts/profile.slurm。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"
PY="${PY:-$HOME/venv/bin/python}"
echo "== profile job $SLURM_JOB_ID on $(hostname) =="
grep Cpus_allowed_list /proc/self/status
sha256sum src/melspec.cpp src/main.cpp scripts/profile.py
scratch=$(mktemp -d "${TMPDIR:-/tmp}/melspec-profile-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/profile.py --case "${CASE:-all}"
