#!/bin/bash
#SBATCH -J melspec-bench
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/bench-%j.out
#SBATCH --error=tmp/logs/bench-%j.err
#
# 正式 benchmark：在计算节点上跑官方 validate.py（成绩以这里的数字为准）。
#   mkdir -p tmp/logs && sbatch scripts/bench.slurm
#   sbatch --export=ALL,VARIANT=v3 scripts/bench.slurm   # 指定被测二进制
#
# 集群实测（2026-10-30 复核，见 versions/_cluster/recheck-2026-10-30.md）：
#   * 绝对不要写 --mem：节点 RealMemory=1，任何 --mem 都排不上
#   * 必须 --hint=nomultithread：否则 16 CPU 只给 8 个物理核
#   * 集群 python3 没有 numpy/librosa → 默认用 ~/venv/bin/python

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache"

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
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CASE="${CASE:-all}"
RUNS="${RUNS:-3}"

echo '== 编译（计算节点原生指令集） =='
make -B -j "$SLURM_CPUS_PER_TASK"
sha256sum src/melspec.cpp src/main.cpp include/melspec.h validate.py Makefile "$VARIANT"
g++ --version
"$PY" -c 'import sys, numpy, librosa; print(sys.version); print("numpy", numpy.__version__, "librosa", librosa.__version__)'

scratch=$(mktemp -d "${TMPDIR:-/tmp}/melspec-bench-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"

if [ "$CASE" = "all" ]; then
  echo "== 正确性 + 计时（官方 validate.py，四组全跑） =="
  "$PY" -u validate.py "$VARIANT" --runs "$RUNS" | tee "tmp/logs/validate-${SLURM_JOB_ID}.txt"
  # 官方脚本即使 FAIL 也返回 0；不修改它，在包装层检查结果。
  grep -q 'Result: 4/4 PASSED' "tmp/logs/validate-${SLURM_JOB_ID}.txt"
else
  echo "== 单组用例 ${CASE}：正确性 + 计时（参数组取自 validate.py 的 TEST_CASES） =="
  "$PY" -u scripts/bench.py "$VARIANT" --case "$CASE" --runs "$RUNS" \
        --out "tmp/logs/bench-${SLURM_JOB_ID}.md" | tee "tmp/logs/validate-${SLURM_JOB_ID}.txt"
fi
