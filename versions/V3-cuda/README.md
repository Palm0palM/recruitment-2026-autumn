# V3 CUDA 公共实验上下文

当前候选：FP32 cuFFT R2C、batch512、全 GPU 加窗/模平方/稀疏 Mel、初始化重叠。
G1–G3 与 G4 实验均已完成；FP32/FP64 均通过本地 74 项合成测试与四组官方音频检查。
最终默认构建在 hepnode2/L40 的作业 238 中四组全 PASS，时间为 1.011 / 0.567 / 1.038 / 1.031 秒；
详见 [BENCHMARK.md](BENCHMARK.md)。
环境、设计、profile 和复现分别见 [ENVIRONMENT.md](ENVIRONMENT.md)、[IMPLEMENTATION.md](IMPLEMENTATION.md)、
[PROFILE.md](PROFILE.md)、[REPRODUCE.md](REPRODUCE.md)。本地 GPU 时间不作为集群成绩。

所有正式改动必须留在 `v3/cuda` 与本 worktree。

## 基线与目标

- 算法 BASE：`15f90558b4ab83ad0e6500e84aa807a942564229`，V1 手写半长 FFT + 稀疏 Mel，double。
- 性能参照 FFTW A2：`b4552f3962751b5a02d2aa56798907c5a1c29afc`；不以 v2/fftw 最新 HEAD 代替。
- 配置提交允许 HEAD 前移；任何算法提交必须证明源自上述 BASE，并记录源码/构建哈希。
- 目标：研究 GPU 上的 FFT、模平方与 Mel 流水，保持数学语义、普通 CLI 和 CPU 回退行为。
- 不根据 PCM/expected 内容特化；显式计入传输、初始化、转换与写回的全流程成本。

正式集群最小 wall time（每组三次）：

| 用例 | V1（hepnode3，秒） | FFTW A2 ESTIMATE（hepnode2，秒） |
|---|---:|---:|
| default | 1.537 | 0.985 |
| small_mels | 1.196 | 0.650 |
| large_fft | 2.634 | 1.356 |
| narrow_band | 2.578 | 1.300 |

两组均四组 PASS；GPU 候选必须与 CPU 基线采用可比的输入、输出和计时口径，不能只比较 kernel 时间。

## 正确性与禁止事项

- `src/main.cpp`、`validate.py`、`data` 不得修改；保持普通 CLI、参数和异常行为。
- 输出必须 float32 小端、无文件头、行主序 `[n_mels, n_frames]`。
- 官方四组必须全部满足 `rtol=1e-3, atol=1e-4`；GPU 数值差异单独记录，不得降低容差。
- 先跑随机 PCM、边界输入和 `scripts/smoke.py`，再全量 expected；不能只测静音/低能量前缀。
- 保留 commit、源码/二进制哈希、编译参数、GPU/驱动/工具链版本、节点和原始日志。

## 测试流程

1. 本地：`source tmp/session-env.sh`；独立构建 CPU 基线并运行 smoke；数据从只读 `tmp/data` 读取。
2. GPU 原型先小规模功能测试，不用重型 Benchmark 代替正确性；失败实验也保留。
3. 集群使用独立远端目录，在 Slurm GPU 计算节点执行；不得假设登录节点有 GPU 或 nvcc。
4. 不写 `--mem`；GPU/CPU 资源、线程、绑核和输出 scratch 显式记录，串行提交作业。
5. 完整官方 `validate.py` 在计算节点执行；本地 perf 不能替代 GPU 节点计时或声称 PMU 数据。

## 资源、限制与边界

- 共享 `tmp/.venv`、PCM/expected 仅按只读约定复用，不修改共享 Python 环境。
- 每个 worktree 独立构建、缓存、输出、日志及原型；worker 仅写 `tmp/prototypes/`。
- 本地记录有 CUDA 13.4 / RTX 5060 Laptop，但与集群 L40 不可混用性能结论。
- 历史集群复核：GPU 节点有 L40，但没有 nvcc/CUDA toolkit；后续须准备私有工具链或容器替代方案。
- `RealMemory=1` 异常仍存在，一节点同时只能一个作业；集群 perf 不可用。
- 子 Agent 只用项目级 Kimi K3.1 Pro Highspeed 配置；主 Agent 负责集成、远程操作和正式测量。
