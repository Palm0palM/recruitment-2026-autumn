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
# 需要先确认集群上有 python（含 numpy/librosa）；没有的话用 spack：
#   spack load python py-numpy py-librosa   （或自己建 uv venv）

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

VARIANT="${VARIANT:-./melspec}"
echo "== job ${SLURM_JOB_ID} on $(hostname) =="
echo "节点: $(scontrol show hostname "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "CPU: $(nproc)  被测程序: ${VARIANT}"
lscpu | grep -E "Model name|^CPU\(s\)" || true
free -g | head -2 || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

PY="${PY:-python3}"
CASE="${CASE:-all}"
RUNS="${RUNS:-3}"

if [ "$CASE" = "all" ]; then
  echo "== 正确性 + 计时（官方 validate.py，四组全跑） =="
  "$PY" validate.py "$VARIANT" --runs "$RUNS" | tee "logs/validate-${SLURM_JOB_ID}.txt"
else
  echo "== 单组用例 ${CASE}：正确性 + 计时（参数组取自 validate.py 的 TEST_CASES） =="
  "$PY" scripts/bench.py "$VARIANT" --case "$CASE" --runs "$RUNS" \
        --out "logs/bench-${SLURM_JOB_ID}.md" | tee "logs/validate-${SLURM_JOB_ID}.txt"
fi
