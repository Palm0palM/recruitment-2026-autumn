#!/bin/bash
#SBATCH -J melspec-bench
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --output=logs/bench-%j.out
#SBATCH --error=logs/bench-%j.err
#
# 正式 benchmark：在计算节点上跑官方 validate.py（成绩以这里的数字为准）。
#   mkdir -p logs && sbatch scripts/bench.slurm
#   sbatch --export=ALL,VARIANT=v3 scripts/bench.slurm   # 指定被测二进制
#
# 集群实测（2026-10-30 复核，见 versions/_cluster/recheck-2026-10-30.md）：
#   * 绝对不要写 --mem：节点 RealMemory=1，任何 --mem 都排不上
#   * 必须 --hint=nomultithread：否则 16 CPU 只给 8 个物理核
#   * 集群 python3 没有 numpy/librosa → 默认用 ~/venv/bin/python

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

VARIANT="${VARIANT:-./melspec}"
if [ -z "${PY:-}" ]; then
  if [ -x "$HOME/venv/bin/python" ]; then PY="$HOME/venv/bin/python"; else PY=python3; fi
fi
echo "python: $PY   CPU: nproc=$(nproc)   cpuset: $(grep Cpus_allowed_list /proc/self/status)"
echo "== job ${SLURM_JOB_ID} on $(hostname) =="
echo "节点: $(scontrol show hostname "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "CPU: $(nproc)  被测程序: ${VARIANT}"
lscpu | grep -E "Model name|^CPU\(s\)" || true
free -g | head -2 || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

PY="${PY:-python3}"
CASE="${CASE:-all}"RUNS="${RUNS:-3}"

if [ "$CASE" = "all" ]; then
  echo "== 正确性 + 计时（官方 validate.py，四组全跑） =="
  "$PY" validate.py "$VARIANT" --runs "$RUNS" | tee "logs/validate-${SLURM_JOB_ID}.txt"
else
  echo "== 单组用例 ${CASE}：正确性 + 计时（参数组取自 validate.py 的 TEST_CASES） =="
  "$PY" scripts/bench.py "$VARIANT" --case "$CASE" --runs "$RUNS" \
        --out "logs/bench-${SLURM_JOB_ID}.md" | tee "logs/validate-${SLURM_JOB_ID}.txt"
fi
