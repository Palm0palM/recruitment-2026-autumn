#!/usr/bin/env bash
#SBATCH --job-name=melspec-amd-thp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode0
#SBATCH --time=00:05:00
#SBATCH --output=tmp/logs/amd-thp-%j.out
#SBATCH --error=tmp/logs/amd-thp-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp/logs tmp/bin
source tmp/amd-gpu/env.sh
echo "host=$(hostname) job=$SLURM_JOB_ID"
"$CXX" -O3 -march=native -std=c++17 scripts/hugepage_probe.cpp \
  -o tmp/bin/hugepage_probe
tmp/bin/hugepage_probe | tee "tmp/logs/hugepage-${SLURM_JOB_ID}.txt"
