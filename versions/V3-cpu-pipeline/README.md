# V3 CPU-Pipeline 公共实验上下文

状态：**已完成 V3 CPU-Pipeline 与数据流融合验证**。最终推荐 B=1 / sparse / 16 workers / `FUSE_POWER_MEL=1`，见 [RESULTS.md](RESULTS.md)；复现见 [REPRODUCE.md](REPRODUCE.md)，数据流实验见 [DATAFLOW.md](DATAFLOW.md)。所有正式改动均留在 `v3/cpu-pipeline` 与本 worktree。

## 基线与目标

- 算法 BASE：`b4552f3962751b5a02d2aa56798907c5a1c29afc`，FFTW A2 原生 R2C + ESTIMATE，double。
- 参考 V1：`15f90558b4ab83ad0e6500e84aa807a942564229`，不使用 v2/fftw 的最新 HEAD 代替固定 BASE。
- 配置提交允许 HEAD 前移；任何算法提交必须证明源自上述 BASE，并记录源码/构建哈希。
- 目标：在不改数学语义、CLI 和受保护文件的前提下，研究 CPU-Pipeline、准备阶段和小批量处理。
- 不根据 PCM/expected 内容特化，不物化全量功率谱，不把微基准收益直接乘成端到端加速比。

正式集群最小 wall time（每组三次）：

| 用例 | V1（hepnode3，秒） | FFTW A2 ESTIMATE（hepnode2，秒） |
|---|---:|---:|
| default | 1.537 | 0.985 |
| small_mels | 1.196 | 0.650 |
| large_fft | 2.634 | 1.356 |
| narrow_band | 2.578 | 1.300 |

两组均四组 PASS；比较性能时优先同作业、同节点、同编译参数的基线，不能跨平台/不同 harness 混算。
A2 后 worker FFT 仍约 68–75%，default 读入/准备/写出已占诊断 wall 约 61%。

## 正确性与禁止事项

- `src/main.cpp`、`validate.py`、`data` 不得修改；保持普通 CLI、参数和异常行为。
- 输出必须 float32 小端、无文件头、行主序 `[n_mels, n_frames]`。
- 官方四组必须全部满足 `rtol=1e-3, atol=1e-4`；最大绝对误差不是独立判据。
- 先跑随机 PCM、边界输入和 `scripts/smoke.py`，再全量 expected；不得只测能量很小的音频前缀。
- FFT/CPU-Pipeline 的正式 Benchmark 保留 commit、源码/二进制哈希、编译命令、节点、cpuset 和原始日志。

## 测试流程

1. 本地：`source tmp/session-env.sh`；独立 `make -j4`，运行 smoke；本地数据来自只读 `tmp/data`。
2. 本地只做开发对照，不运行重型全量 librosa；正式候选记录逐次结果和失败实验。
3. 集群使用独立远端目录，在 Slurm 计算节点重新构建并跑完整 `validate.py --runs 3`。
4. CPU 作业 1 task、16 CPU、`--hint=nomultithread`，不写 `--mem`；禁止登录节点计算。
5. 大输出放节点本地临时目录；同一实验串行提交作业，不覆盖 V1/V2 或其他 Agent 的日志。
6. 正式计时与插桩分离；本地 WSL perf 可作 CPU 用户态热点参考，不能推断集群 PMU 指标。

## 资源、限制与边界

- 共享 `tmp/.venv`、PCM/expected 仅按只读约定复用，不运行 setup.sh 或修改共享 Python 环境。
- 每个 worktree 独立 `tmp/build`、缓存、输出、日志及原型；worker 仅写 `tmp/prototypes/` 或分配的 `tmp/v3-cpu-lab/<task>/`。
- 集群无预装 FFTW/BLAS，沿用 `scripts/build_fftw.sh` 的无 sudo 私有静态构建方案。
- 已知 `RealMemory=1` 异常，一节点同时只能一个作业；`perf_event_paranoid=4`，集群不能 perf。
- 本地 i7-14650HX 仅 AVX2/FMA；集群 Xeon Gold 6338 支持 AVX-512，结论须回集群复验。
- 子 Agent 只用项目级 Kimi K3.1 Pro Highspeed 配置；主 Agent 负责集成、远程操作和正式测量。
