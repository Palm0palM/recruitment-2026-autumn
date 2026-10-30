# V3 CPU-Pipeline 最终结果

## 推荐候选

最终算法提交：`2e85d8138faf4bef058beba15ee6c606c4e3c39e`，从固定 FFTW A2 `b4552f39...` 继承。
默认参数：`FFT_MODE=r2c FFTW_PLAN=estimate FRAME_BLOCK=1 MEL_PROJECTION=sparse CPU_THREADS=16 FUSE_POWER_MEL=1`。
候选二进制 SHA-256：`0d7e4129cd7ef6e4b79bc3ede348750aa69f833feabed0d9106acf7c8452c0e9`。
`src/main.cpp`、`validate.py`、`data` 未修改；普通 CLI 和 float32 小端行主序输出保持。

核心改动是 **S1 压缩 Mel + R2C 功率谱/Mel 融合**：只在各滤波器非零支撑 bin 上计算模平方并立即累加，
不再完整写入/读取 power 中间数组。FFTW Plan、FFT 精度、线程数和输出语义不变。
普通 Frame Block、双前缀矩和 output 初始化重叠均未采用。

## 集群正式结果

hepnode2，Xeon Gold 6338，1 task × 16 CPU，cpuset `0-15`，nomultithread，不写 mem。
作业 261：完整官方 `validate.py --runs 3`，每组取最小，**4/4 PASS**。

| 用例 | 最终候选（s） | max_abs_err | 同作业内 librosa / 候选 |
|---|---:|---:|---:|
| default | **0.961** | 0.03125 | 15.076× |
| small_mels | **0.641** | 0.01171875 | 21.527× |
| large_fft | **1.320** | 0.125 | 22.752× |
| narrow_band | **1.256** | 0.0625 | 24.057× |

两轮同节点 stable/fusion 交错对照（median，秒）：

| 用例 | 作业 244 stable → fusion | 作业 261 stable → fusion | 结论 |
|---|---|---|---|
| default | 0.978 → 0.982 | 0.980 → 0.973 | 接近噪声 |
| small_mels | 0.661 → 0.649 | 0.664 → 0.650 | 约 1.7–2.2% 改善 |
| large_fft | 1.357 → 1.343 | 1.349 → 1.350 | 接近噪声 |
| narrow_band | 1.308 → 1.274 | 1.298 → 1.267 | 约 2.4–2.7% 改善 |

收益是**小而有针对性的端到端改进**，不是所有用例同幅度加速；narrow_band 最稳定。
默认 FFTW A2 的官方基线为 `0.985 / 0.650 / 1.356 / 1.300 s`，但不同作业的最小值不用于计算加速比。

## 非 FFT 与端到端 Profile（作业 262，诊断构建）

诊断四组输出全部 PASS；插桩不作为正式成绩。

| 用例 | read（s） | compute setup（s） | compute 全部（s） | 转换/写出（s） | 诊断 wall（s） | 峰值 RSS（KiB） |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.181743 | 0.211038 | 0.548073 | 0.219678 | 0.980949 | 776808 |
| small_mels | 0.179860 | 0.052399 | 0.393793 | 0.067778 | 0.661506 | 343772 |
| large_fft | 0.181943 | 0.204150 | 0.892642 | 0.224484 | 1.327009 | 772756 |
| narrow_band | 0.184697 | 0.201130 | 0.825867 | 0.233977 | 1.273723 | 772524 |

setup 细分：压缩 Mel 构造仅约 73–146 μs；default output.resize 约 0.201 s，
其余大输出用例约 0.198–0.201 s，small_mels 约 0.050 s。接口约束下不能合法跳过初始化。
output 初始化重叠实验在集群没有稳定收益，未采用。

worker 变化：
- 独立 `power_s` 已变为 0，工作融合进 `mel_s`；FFT execute 线程秒基本不变。
- default 原 power+Mel 线程秒约 0.913，融合后约 0.831。
- narrow_band 原约 1.075，融合后约 0.572，解释其较稳定的端到端收益。
- default FFT execute 约 71.29%（线程秒口径），不能把该比例误当成全进程占比。

## 独立采用与回退

- **采用 S1 压缩滤波器**：公开稠密接口不变，39 组输出与 A2 完全相同。
- **采用 R2C 功率谱/Mel 融合**：减少支撑外功率计算和完整 power pass，最终默认 `FUSE_POWER_MEL=1`。
- **保持 B=1**：B=2/4/8/16 在集群整体回退。
- **拒绝双前缀矩**：局部变快但其他用例回退，并存在严重抵消风险。
- **拒绝 output 初始化重叠**：small_mels 明显回退，其他组无稳定收益；官方验证虽 PASS 也不采用。
- **保持 16 workers**：集群四组均为 16 < 12 < 8。
- **不额外手写 SIMD**：GCC 已对压缩 Mel、融合循环和 Hann pack 生成 32-byte AVX2 vectors。
- **不做 output 初始化 hack、额外线程/NUMA 调整或跨帧缓存**：没有满足约束的明确收益假设。

最终融合版另通过 ASan/UBSan、8 项 smoke、39 组逐元素一致检查及短输入拒绝。
Kimi 只读审查未发现分块越界、矩索引、默认计时宏副作用或线程失败回退问题。

## 复现与证据

- 构建/集群步骤：[REPRODUCE.md](REPRODUCE.md)
- 首轮全部实验：[EXPERIMENTS.md](EXPERIMENTS.md)
- 数据流与融合实验：[DATAFLOW.md](DATAFLOW.md)
- 基线：[BASELINE.md](BASELINE.md)
- 最终原始：`raw/cluster/cpu-261.out`、`official-261.txt`、`compare-261.json`、`cpu-262.out`、`final-accounting.txt`
- 其他失败与流程证据：`raw/cluster/cpu-212*`、`bench-216*`、`bench-248*`

后续优化应转向真正的计算量或受保护接口之外的合法数据流，而不是重复已否决的 Frame Block、
双前缀、线程数调整、FFTW 后端变化或 output 初始化规避。任何新改动仍需同节点 A/B。
