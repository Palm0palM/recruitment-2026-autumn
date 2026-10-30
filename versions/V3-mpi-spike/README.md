# V3 MPI-Spike 公共实验上下文

状态：仅完成环境准备，未开始优化。所有正式改动必须留在 `v3/mpi-spike` 与本 worktree。

## 基线与目标

- 算法 BASE：`b4552f3962751b5a02d2aa56798907c5a1c29afc`，FFTW A2 原生 R2C + ESTIMATE，double。
- 参考 V1：`15f90558b4ab83ad0e6500e84aa807a942564229`，不使用 v2/fftw 最新 HEAD 代替固定 BASE。
- 配置提交允许 HEAD 前移；任何算法提交必须证明源自上述 BASE，并记录源码/构建哈希。
- 目标：以 spike 方式评估多节点 MPI 是否有端到端价值，显式计入通信、初始化、同步和写出。
- 不根据 PCM/expected 内容特化，不把单节点 FFT 加速直接外推为多节点收益。

正式集群最小 wall time（每组三次）：

| 用例 | V1（hepnode3，秒） | FFTW A2 ESTIMATE（hepnode2，秒） |
|---|---:|---:|
| default | 1.537 | 0.985 |
| small_mels | 1.196 | 0.650 |
| large_fft | 2.634 | 1.356 |
| narrow_band | 2.578 | 1.300 |

两组均四组 PASS；MPI 候选须与同节点单进程基线及可比的多节点作业口径同时记录，不能隐去通信成本。

## 正确性与禁止事项

- `src/main.cpp`、`validate.py`、`data` 不得修改；保持普通 CLI、参数和异常行为。
- 输出必须 float32 小端、无文件头、行主序 `[n_mels, n_frames]`；汇集结果顺序必须确定。
- 官方四组必须全部满足 `rtol=1e-3, atol=1e-4`；任何 rank 失败都视为验证失败。
- 先跑随机 PCM、边界输入和 `scripts/smoke.py`，再全量 expected；小样本也要覆盖首尾 rank 的分块。
- 保留 commit、源码/二进制哈希、MPI/UCX/FFTW 版本、节点数、rank 映射、cpuset 和原始日志。

## 测试流程

1. 本地：`source tmp/session-env.sh`；独立构建并运行 smoke；数据从只读 `tmp/data` 读取。
2. 集群使用独立远端目录，在 Slurm 计算节点运行；禁止登录节点计算或调试 MPI 负载。
3. 单节点对照为 1 task、16 CPU、`--hint=nomultithread`；多节点配置必须显式记录，不写 `--mem`。
4. 大输出放节点本地 scratch，必要结果再复制到本实验 home；串行作业，不覆盖其他实验。
5. 完整官方 `validate.py` 检查输出；阶段计时与正式计时分离，集群不能依赖 perf。

## 资源、限制与边界

- 共享 `tmp/.venv`、PCM/expected 仅按只读约定复用，不修改共享 Python 环境。
- 每个 worktree 独立构建、缓存、输出、日志及原型；worker 仅写 `tmp/prototypes/`。
- 集群无预装 FFTW，沿用无 sudo 私有静态构建；不要借用 V2 工作区的构建产物。
- 历史记录中 OpenMPI 4.1.6 / UCX 1.18.1 的 wrapper/runtime 路径存在故障，须先小作业验证，不默认可用。
- `RealMemory=1` 异常导致一节点同时只能一个作业；`perf_event_paranoid=4`。
- 子 Agent 只用项目级 Kimi K3.1 Pro Highspeed 配置；主 Agent 负责集成、远程操作和正式测量。
