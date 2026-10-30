# V3 CUDA 复现说明

## 源码与接口

- Worktree/branch：`melspec-v3-cuda` / `v3/cuda`。
- 算法源自 V1 `15f90558b4ab83ad0e6500e84aa807a942564229`，没有引入 FFTW 或其他 CPU 实验源码。
- `src/main.cpp`、`validate.py`、`include/melspec.h` 和 `data` 未修改。
- 公共接口仍为 `compute_melspectrogram(..., std::vector<double>&)`；输出由原 main 转 float32。
- 最终候选的正式代码身份见 [BENCHMARK.md](BENCHMARK.md)；文档提交不改变算法。

## 构建

默认 CPU 对照（无 CUDA 依赖）：

```bash
source tmp/session-env.sh
make CUDA=0 -j4
$PY scripts/smoke.py ./melspec
```

L40 候选（在本实验私有 CUDA 13.4 工具链下）：

```bash
export CUDA_HOME=/home/zsx/melspec-v3-cuda-agent/toolkit/cuda-13.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
make -B -j4 CUDA=1 CUDA_HOME="$CUDA_HOME" CUDA_ARCH=89 \
  CUDA_PRECISION=32 CUDA_BATCH=512 CUDA_FUSED=1 \
  CUDA_OVERLAP_INIT=1 CUDA_PINNED=0 CUDA_PROFILE=0 EXEC=tmp/bin/cuda-fp32
```

本机功能测试需用本机架构（RTX 5060 为 120），不能用 L40 的 sm_89 二进制：

```bash
make -B -j4 CUDA=1 CUDA_ARCH=120 CUDA_PRECISION=32 CUDA_BATCH=512 \
  CUDA_OVERLAP_INIT=1 EXEC=tmp/bin/cuda-fp32
$PY scripts/smoke.py tmp/bin/cuda-fp32
```

其他独立实验：`CUDA_PRECISION=64`、`CUDA_FUSED=0`、`CUDA_BATCH=128/256`、`CUDA_PINNED=1`、
`CUDA_OVERLAP_INIT=0`。`CUDA=0` 始终是 V1 CPU 对照。最终默认值为
`CUDA=1, CUDA_PRECISION=32, CUDA_BATCH=512, CUDA_FUSED=1, CUDA_OVERLAP_INIT=1, CUDA_PINNED=0, CUDA_PROFILE=0`。
Makefile 对每次 make 强制重新链接，避免切换配置后误用旧可执行文件。

## 私有 CUDA 工具链

集群原始状态只有 NVIDIA driver，没有 nvcc/cuFFT 开发文件。复现环境使用本机 CUDA 13.4.92
复制到实验私有目录，不安装系统包：

```text
toolkit/cuda-13.4/
  bin/                               # nvcc、cicc、ptxas、fatbinary 等
  nvvm/                              # NVVM/libdevice
  targets/x86_64-linux/include/      # CUDA/cuFFT 头文件
  targets/x86_64-linux/lib/          # nvcc 默认设备/host 链接搜索目录
  lib64/                             # 运行时 rpath 目录
```

库包括 CUDA Runtime/cuFFT 动态库，以及 nvcc 需要的 `libcudadevrt.a`、`libcudart_static.a`、
`libculibos.a`。以绝对 `-L` 和 `-Wl,-rpath` 链接，不使用系统 FFTW/CUDA 安装。
CUDA/cuFFT 是 NVIDIA 的工具和库，遵守其 EULA；工具链、二进制、音频与 perf.data 不进入 Git。

## 正确性与实验

```bash
# 8 项基础回归
$PY scripts/smoke.py tmp/bin/cuda-fp32

# 74 项合成 + 全量官方输入（harness 在 prototypes/precision 留档）
tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py tmp/bin/cuda-fp32
tmp/v3-cuda-lab/precision/python.sh \
  tmp/v3-cuda-lab/precision/precision_harness.py tmp/bin/cuda-fp32 --suite official --timeout 3600

# 本地交错进程对照，不运行全量 librosa
$PY scripts/cuda_bench.py --programs v1=./melspec cuda=tmp/bin/cuda-fp32 \
  --runs 3 --out tmp/results/local.json
```

容差固定 `rtol=1e-3, atol=1e-4`；n_fft=1 的 V1 周期窗与 SciPy 特例不同，
小/空 Mel 响应不能单独证明 FFT 正确，必须同时看随机、大变换和官方 expected。

## Slurm

通过 bundle/clone 保留源码身份，不使用只 rsync 的无 Git 目录作为正式候选：

```bash
git bundle create tmp/v3-cuda.bundle v3/cuda
rsync -av tmp/v3-cuda.bundle hepnode:melspec-v3-cuda-agent-new.bundle
ssh hepnode 'git clone -b v3/cuda ~/melspec-v3-cuda-agent-new.bundle ~/melspec-v3-cuda-agent/candidate'
ssh hepnode 'cd ~/melspec-v3-cuda-agent/candidate && mkdir -p tmp/logs && \
  sbatch --wait -w hepnode2 --export=ALL,MODE=bench,P=32,CUDA_BATCH=512,CUDA_OVERLAP_INIT=1,\
CUDA_HOME=/home/zsx/melspec-v3-cuda-agent/toolkit/cuda-13.4 scripts/cuda-cluster.slurm'
```

所有 GPU 构建/运行都在 Slurm 内，1 task、16 CPU、1 GPU、nomultithread，不写 `--mem`。
大输出在节点本地 `/tmp/melspec-cuda-$JOBID-*` scratch，日志/小结果再复制回本实验 home。
串行作业，不覆盖其他路线目录或日志。`MODE=smoke` 可运行小测试，`MODE=profile` 配合
`PROFILE=1` 生成阶段诊断；正式 benchmark 必须 `PROFILE=0`。

A2 对照固定为 `b4552f3962751b5a02d2aa56798907c5a1c29afc`，在独立目录重建，
比较采用同作业交错 `scripts/cuda_bench.py`，不能只用 V1 证明 GPU 加速。
`cuda-sweep.slurm` 比较 batch/fused/precision/pinned/overlap；`cuda_profile.py` 保留阶段 stderr。
`cufft-probe` 的独立 probe 在 `prototypes/cufft/`，只用于 cuFFT layout/plan/事件分析。

## 解读限制

- `cuda_profile total_ms` 只覆盖 `compute_melspectrogram_cuda`，不含 main 的 PCM 读取与最终写出。
- gather/FFT/Mel 的诊断是 host wall + launch/sync，不是纯 kernel event 时间；正式 score 不启用它。
- D2H 阶段在 pinned 路径还含 host staging 分配；`cudaMemGetInfo` 是设备级使用量，不是进程专属显存。
- 本地 RTX 5060 与集群 L40 的绝对时间不可混用；官方结论以 hepnode2/L40 的全量结果为准。
