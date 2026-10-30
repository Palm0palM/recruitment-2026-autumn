# V3 CUDA 环境与工具链勘察

## Kimi 身份核验

首次只读 `kimi-researcher` 子任务 session：`ses_e746bca4dffeg9AY0CCpOxgtrR`。
OpenCode API 实际返回 `providerID=ppio`、`id=moonshotai/kimi-k3.1-pro-highspeed`、`variant=high`，
agent 为 `kimi-researcher`，outcome succeeded；usage 中 reasoning tokens 为 1689。
这同时验证了真实子会话模型身份与 Thinking 使用，而非仅依赖子 Agent 自我陈述。原始 JSON 归档于 raw。

## 实际 Slurm GPU 探测（作业 99）

- 唯一分区 `hepnodes`；`hepnode[1-3]` 配置 `gpu:L40:2`，hepnode0 无 GPU。
- 作业：`1 node / 1 task / 4 CPU / --gres=gpu:1 / --hint=nomultithread`，未指定 `--mem`。
- hepnode3，作业 `COMPLETED / 0:0`；`CUDA_VISIBLE_DEVICES=0`、`SLURM_GPUS_ON_NODE=1`。
- 实际设备：NVIDIA L40，driver `595.91.07`，`compute_cap=8.9`，每卡 `46068 MiB`。
- 节点两张 L40；本次只申请一张。`nvidia-smi topo -m` 显示 GPU0 对应 CPU affinity 16–31、NUMA1；
  GPU1 对应 32–47、NUMA2；GPU 间为 SYS，不是 NVLink。
- 节点存在 `/usr/bin/nvidia-smi` 和 `libcuda.so.1`，但 **没有 nvcc、CUDA toolkit、cuFFT/cuBLAS 头文件或开发库**。
- `/home` 可用约 3.7 TB；节点 `/tmp` 为本地大容量 scratch。`RealMemory=1` 异常仍需遵守不写 mem。

## 无 sudo 私有工具链方案

复用本机同版本 CUDA 13.4（nvcc 13.4.92），复制到本实验远端目录
`~/melspec-v3-cuda-agent/toolkit/cuda-13.4/`，不写系统路径、不修改其他实验：
`bin/`、`nvvm/`、`targets/x86_64-linux/include/`、私有 `lib64` 及 nvcc 默认搜索的
`targets/x86_64-linux/lib/`。包含 CUDA Runtime/cuFFT 动态库，以及 nvcc 链接所需的
`libcudadevrt.a`、`libcudart_static.a`、`libculibos.a`。
不复制 cuBLAS、调试器、文档或无关大库，降低部署体积。用 `-arch=sm_89` 生成 L40 原生代码，
在 Slurm 内通过绝对 include/lib/rpath 构建与运行；不得依赖登录节点的预装 toolkit。

先用 `tmp/cuda-smoke.cu` 做 8 点 R2C 与直接 DFT 对照。只有该最小编译/运行通过后，才开展并行原型研究。
工具链和 .so 均不进入 Git；复现记录保留来源版本、目录结构、编译命令和实际运行日志。

## 最小编译/运行结果

作业 123（hepnode3）`COMPLETED / 0:0`：nvcc 13.4.92 在 Slurm 内成功构建，
`ldd` 确认使用实验私有 `libcufft.so.12` 和 `libcudart.so.13`，设备报告 `NVIDIA L40 sm=8.9`，
8 点 R2C 全 5 个非负频率 bin 与直接 DFT 对照通过。

早期失败已保留：作业 102/107 因 nvcc 的设备链接和 host 链接还需 `libcudadevrt`、`libcudart_static`，
以及默认库搜索目录为 `targets/x86_64-linux/lib` 而失败。补齐这些小型静态运行库和默认目录后成功，
没有安装系统包或放宽运行要求。原始日志见 `raw/smoke-102.err`、`raw/smoke-107.err`、`raw/smoke-123.out`。
