# V3 CPU-Pipeline 基线与独立实验设计

## 固定算法与测量边界

- FFTW A2 BASE：`b4552f3962751b5a02d2aa56798907c5a1c29afc`。
- 保留原生 R2C、ESTIMATE、double、共享单变换 Plan 和外层 worker 分区。
- 不研究 FFTW Wisdom、plan_many/Batch FFT 或 Codelet；这些属于其他 Agent。
- 不修改 `src/main.cpp`、`validate.py`、`data`，不根据 PCM/expected 内容特化。
- 所有候选依次通过 smoke、随机/边界测试、本地 A/B；有效候选再在同节点集群复测。
- 正式 wall time、诊断插桩和本地 perf 分开记录；不以微基准代替端到端结果。

## 已验证的集群 A2 基线

hepnode2，1 task × 16 CPU，`--hint=nomultithread`，官方 `validate.py --runs 3` 取最小，4/4 PASS。

| 用例 | A2 ESTIMATE（s） |
|---|---:|
| default | 0.985 |
| small_mels | 0.650 |
| large_fft | 1.356 |
| narrow_band | 1.300 |

A2 诊断作业 59 中，default 读取 0.180045 s、compute 准备 0.210458 s、compute 全部 0.565350 s、
转换/写出 0.222535 s。worker 阶段比例不是全进程比例，必须同时看两种口径。

## 并行研究分工

先以只读 `kimi-researcher` 实际调用验证项目配置成功（session `ses_e746c39f1ffe453AnzL7LrE8Mn`）。
随后仅使用项目级 Kimi K3.1 Pro Highspeed（`ppio/moonshotai/kimi-k3.1-pro-highspeed#high`）：

| 子任务 | 独立目录 | 研究范围 |
|---|---|---|
| A：Mel 算法 | `tmp/v3-cpu-lab/mel/` | 稀疏点积、零/一阶矩双前缀、直接压缩滤波器、数值抵消 |
| B：帧分块 | `tmp/v3-cpu-lab/frames/` | 小 B 帧 scratch、m→frame→k 输出局部性、块尾和缓存 |
| C：内存/线程 | `tmp/v3-cpu-lab/memory/` | resize/首次触页、first-touch、NUMA、线程数与绑定、合法初始化 |

子 Agent 不写正式源码、不运行集群；主 Agent 审查后独立提交、独立测量。

## 候选实验顺序

1. **BASE**：当前算法与 b4552f3 相同，本地五次进程计时及四组阶段插桩。
2. **压缩滤波器**：仅改 compute 内部准备，保持公开 `build_mel_filterbank` 接口与逐 bin 权重公式。
3. **小帧块**：仍逐帧调用同一 FFTW 单变换 Plan；改变 scratch 与 Mel/输出循环，不合并 FFT backend 修改。
4. **矩查询**：仅在数学/容差和性能均有证据时考虑；必须保留抵消与回退数据。
5. **内存/线程**：细分初始化，不以 UB 跳过 `vector<double>` 初始化；线程/NUMA 实验单独测量。

每组至少报告非 FFT 阶段、FFT execute、compute、端到端、正确性和内存。回退版本保留而不强行采用。

## 本地固定 A2 复测（研发开始）

当前 `src/melspec.cpp` 与 BASE 一致，编译 `-O3 -march=native -pthread`，系统 FFTW 3.3.11。
8 项 smoke 全 PASS；四组全量、五次进程计时均与 expected 相符：

| 用例 | min（s） | median（s） | 诊断 read/setup/compute/write（s） |
|---|---:|---:|---|
| default | 0.752183 | 0.801738 | 0.118795 / 0.166515 / 0.480318 / 0.199738 |
| small_mels | 0.485293 | 0.488870 | 0.101455 / 0.032730 / 0.359661 / 0.062055 |
| large_fft | 1.131332 | 1.167819 | 0.101672 / 0.133747 / 0.847628 / 0.184566 |
| narrow_band | 1.135767 | 1.161458 | 0.117369 / 0.149323 / 0.835561 / 0.187143 |

诊断与正式计时分离，setup 是 compute 的子集，不与 compute 相加。原始文件见 `raw/base-*`。
