#!/bin/bash
#SBATCH -J melspec-perf
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --hint=nomultithread
#SBATCH --output=logs/perf-%j.out
#SBATCH --error=logs/perf-%j.err
#
# profile：perf 采样 + 火焰图 + 阶段计时。
#   mkdir -p logs && sbatch scripts/profile.slurm
#   sbatch --export=ALL,VARIANT=v3,SLICE=full scripts/profile.slurm
#
# 只跑单例（-n 1），避免多线程干扰采样；要看并行利用率用 htop/glances。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

VARIANT="${VARIANT:-./melspec}"
case "${SLICE:-full}" in
  full) PCM=data/audio.pcm ;;
  *)    PCM="${SLICE}" ;;
esac
ARGS="${ARGS:---sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 --f_min 0 --f_max 22050}"

echo "== perf on $(hostname) =="
perf stat -e cycles,instructions,cache-references,cache-misses \
          -e LLC-loads,LLC-load-misses \
          "$VARIANT" "$PCM" /tmp/out_perf.bin $ARGS 2>&1 | tee "logs/perf-stat-${SLURM_JOB_ID}.txt" || true

perf record -F 999 -g -o "logs/perf-${SLURM_JOB_ID}.data" -- \
     "$VARIANT" "$PCM" /tmp/out_perf.bin $ARGS
perf report -i "logs/perf-${SLURM_JOB_ID}.data" --stdio --percent-limit 1 \
     | head -60 | tee "logs/perf-report-${SLURM_JOB_ID}.txt"

if command -v stackcollapse-perf.pl >/dev/null && command -v flamegraph.pl >/dev/null; then
  perf script -i "logs/perf-${SLURM_JOB_ID}.data" \
    | stackcollapse-perf.pl | flamegraph.pl > "logs/flame-${SLURM_JOB_ID}.svg"
  echo "火焰图: logs/flame-${SLURM_JOB_ID}.svg"
fi
