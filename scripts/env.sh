# 用法：source scripts/env.sh
# 提供 $PY（带 numpy/scipy/librosa 的解释器）与 $ROOT，并把常用目录加进 PATH。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export ROOT
export PY="$ROOT/tmp/.venv/bin/python"
export PATH="$HOME/.local/bin:$ROOT/tmp/.venv/bin:$PATH"

if [ ! -x "$PY" ]; then
  echo "警告: 未找到 $PY，请先运行 scripts/setup.sh" >&2
fi

# 集群上常用的环境变量（本地无效，忽略即可）
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
