# MPI 原型复现步骤

## 隔离与保护

仅在 `melspec-v3-mpi` worktree、`v3/mpi-spike` 分支工作。`src/main.cpp`、`validate.py`、`data`
不得修改；独立 driver 不是正式 main。共享 Python/PCM/expected 只读，私有 FFTW 源码和构建均在本实验 tmp。
不运行原始 V0，不覆盖 V1/V2/CPU/CUDA 的本地或远端目录。

## 本地（只读设计与基础检查）

```bash
source tmp/session-env.sh
make -j4
$PY scripts/smoke.py ./melspec
```

本地没有可用 MPI launcher 时，不在 WSL 安装/修改共享环境；MPI 功能测试放在 Slurm 计算节点。
帧划分和打包可用独立模型验证，但不能替代真实 MPI 输出检查。

## 部署

远端使用专属 `/home/zsx/melspec-v3-mpi-spike`。可用 Git bundle 或 rsync 源码快照，
排除 `.git`、`tmp`、二进制、对象文件、音频和 expected。不要把 worktree 的 `.git` 指针文件复制到远端。

## 计算节点作业

```bash
# 正确性优先使用 2 节点：1/2 Rank 每节点 1 Rank；4 Rank 每节点 2 Rank，
# 各 8 核且 --bind-to core。此小矩阵不用于性能结论。
ssh hepnode 'cd ~/melspec-v3-mpi-spike && mkdir -p tmp/logs && \
  sbatch --wait -N2 -n4 --ntasks-per-node=2 -c8 \
  --export=ALL,MODE=check scripts/mpi-experiment.slurm'
# check 完成且正确后，性能测试再申请 4 节点、每节点 1 Rank ×16 CPU：
ssh hepnode 'cd ~/melspec-v3-mpi-spike && \
  sbatch --wait --export=ALL,MODE=bench,RUNS=3 scripts/mpi-experiment.slurm'
# 研究 wrapper 的官方四组验证（不是正式提交资格证明）：
ssh hepnode 'cd ~/melspec-v3-mpi-spike && \
  sbatch --wait --export=ALL,MODE=official,MELSPEC_MPI_RANKS=2,MELSPEC_MPI_IO=scatter,RUNS=3 \
  scripts/mpi-experiment.slurm'
```

作业内完成 FFTW 静态库、A2 和 MPI driver 构建，不在登录节点编译/运行计算负载。
申请 4 节点、每节点 1 task × 16 CPU；资源不足时等待，不用抢占/并行提交本实验多个作业。
不写 `--mem`。所有大输出在 root 所在节点的本地 scratch，脚本 trap 只清理自己创建的目录。

## OpenMPI 环境

```bash
export OPENMPI_HOME=/cache/toolchain/openmpi-4.1.6
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib:/cache/toolchain/ucx-1.18.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$OPENMPI_HOME" OPAL_LIBDIR="$OPENMPI_HOME/lib"
```

手工 g++ 链接 `-lmpi` 是因为现有 wrapper 的 `/opt` 前缀失效；先用 hello 验证。
正常 `srun` 必须显式 `--mpi=pmix`；本实验默认用已验证的 `mpirun`。
所有脚本保留版本、commit/源码哈希、二进制哈希、节点列表、Rank host/cpuset 和原始计时。

## 归档规则

需要展示的文本/JSON 放 `versions/V3-mpi/raw`；大输出、库、二进制和 MPI/性能临时数据留 tmp，不提交。
研究 wrapper 的结果只能标记为“独立 MPI 原型通过未修改 validate.py 的研究验证”，不能省略 launcher 兼容性限制。
