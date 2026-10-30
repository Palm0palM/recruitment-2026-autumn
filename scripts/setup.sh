#!/usr/bin/env bash
# 本地/登录节点环境配置（幂等，可重复执行）
#
#   ./scripts/setup.sh              # 建 venv + 链接数据 + 编译
#   MELSPEC_DATA=/path/to/data ./scripts/setup.sh
#
# 说明：
#   * 仓库自带的 data 是指向集群 /cache/recuritment/melspec/data 的符号链接，
#     本地（WSL）通常是断的。本脚本不会改动这个链接，而是在 tmp/data 下
#     建立可用的副本链接，供 tmp/ 内的脚本使用。
#   * 所有临时产物（venv、数据链接、bench 输出）都在 tmp/ 下，已被 .gitignore 忽略。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPYCACHEPREFIX="$ROOT/tmp/pycache"

VENV="$ROOT/tmp/.venv"
PY="$VENV/bin/python"

# 数据来源优先级：显式变量 > 仓库 data > 本地既有副本
CANDIDATES=(
  "${MELSPEC_DATA:-}"
  "$ROOT/data"
  "$HOME/projects/heptagon/autumn26/cluster_melspec/data"
)

echo "== 1. Python 环境 =="
if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv，请先安装：https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" numpy scipy librosa
else
  echo "复用已有 venv: $VENV"
fi
"$PY" - <<'EOF'
import librosa, numpy, scipy, sys
print(f"python {sys.version.split()[0]}  numpy {numpy.__version__}  "
      f"scipy {scipy.__version__}  librosa {librosa.__version__}")
EOF

echo "== 2. 数据 =="
SRC=""
for c in "${CANDIDATES[@]}"; do
  [ -n "$c" ] || continue
  if [ -f "$c/audio.pcm" ] && [ -d "$c/expected" ]; then SRC="$(cd "$c" && pwd)"; break; fi
done
mkdir -p "$ROOT/tmp/data"
if [ -n "$SRC" ]; then
  ln -sfn "$SRC/audio.pcm" "$ROOT/tmp/data/audio.pcm"
  ln -sfn "$SRC/expected" "$ROOT/tmp/data/expected"
  echo "数据来源: $SRC"
  echo "样例数: $(ls "$SRC/expected" | wc -l) 个 expected，$(stat -c%s "$SRC/audio.pcm") 字节 PCM"
else
  echo "警告: 未找到可用的 data/audio.pcm 与 data/expected/" >&2
  echo "      本地可用 scripts/smoke.py 合成音频做小样例检查；" >&2
  echo "      集群上请确认 /cache/recuritment/melspec/data 可读。" >&2
fi

echo "== 3. 编译 =="
make -j"$(nproc)"

echo
echo "完成。常用命令："
echo "  source scripts/env.sh                     # 设置 PY / PATH"
echo "  \$PY scripts/smoke.py                         # 随机小样例正确性"
echo "  \$PY scripts/bench.py --slice 0.5 --runs 1     # 真实 PCM 小样例"
echo "  集群流程见 scripts/README.md；全量 benchmark/profile 通过 Slurm 运行。"
