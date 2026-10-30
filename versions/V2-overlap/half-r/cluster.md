# half-r 集群 Benchmark 与 Profile

日期：2026-10-30。专属远端目录：`~/recruitment-2026-autumn-v2-overlap-half-r`。
未覆盖 V1 或其他实验目录，未修改 master。

## 1. 测量口径

- Benchmark 作业 **77**，节点 **hepnode2**，1 task × 16 CPU，`--hint=nomultithread`，无 `--mem`。
- Intel Xeon Gold 6338 @ 2.00GHz，GCC 15.2.0；计算节点重新编译三个独立二进制。
- Python 3.14.4、numpy 2.5.4、librosa 1.0.0；官方 `validate.py --runs 3`。
- 同节点按 V1 → half → half+delta 顺序运行；每组取三次进程 wall-time 最小值，含 I/O。
- 输出使用节点本地 scratch；所有版本四组均 **4/4 PASS**。
- 作业总耗时 15:00 包含编译及三份 librosa 对照，不是程序单次耗时。
- 原始日志：[作业输出](raw/cluster-half-r-bench-77.out)、
  [V1](raw/validate-v1-77.txt)、[half](raw/validate-half-77.txt)、
  [half+delta](raw/validate-half-delta-77.txt)。

## 2. 官方全量结果

| 用例 | V1 秒 | half 秒 | half+delta 秒 | V1/half | V1/half+delta |
|---|---:|---:|---:|---:|---:|
| default | 1.543 | 1.403 | **1.387** | 1.100× | **1.113×** |
| small_mels | 1.207 | 1.060 | **1.053** | 1.139× | **1.146×** |
| large_fft | 2.639 | 2.141 | **2.120** | 1.233× | **1.245×** |
| narrow_band | 2.597 | 2.090 | **2.088** | 1.243× | **1.244×** |

即 half+delta 在同节点 V1 对照下，四组耗时降低约 **10.1% / 12.7% / 19.6% / 19.5%**。
delta 相对只做 half 的额外加速分别为 **1.012× / 1.007× / 1.010× / 1.001×**，
收益较小，但这次集群四组均不为负。

同作业内官方 librosa 对照：

| 用例 | V1 speedup | half speedup | half+delta speedup |
|---|---:|---:|---:|
| default | 9.088× | 10.414× | **11.101×** |
| small_mels | 11.413× | 13.421× | **14.156×** |
| large_fft | 11.168× | 14.064× | **14.813×** |
| narrow_band | 11.368× | 14.296× | **15.123×** |

加速比结论必须使用同作业的 V1 时间，不能把本次程序时间与旧作业的 librosa 时间混合。

## 3. Profile

首次 profile 作业 **80** 在 hepnode3 失败：远端没有 `tmp/scratch`，
`stage_profile.py` 创建临时目录时抛出 `FileNotFoundError`，没有产生有效阶段数据。
失败原文保留在 `raw/cluster-half-r-profile-80.err`。随后在脚本中显式创建该目录并重跑。

成功的 profile 作业 **85** 在 **hepnode2** 完成，`COMPLETED / 0:0`，总作业时间 1:33。
使用独立 `chrono` 插桩副本，强制单线程；四种变体/用例组合的输出全部 PASS。
集群 `perf_event_paranoid=4`，不使用不可用的 perf 硬件计数。
以下是各阶段占插桩总时间的比例；插桩有额外开销，不能与正式 benchmark 成绩混用。

| 阶段 | half default | delta default | half large | delta large |
|---|---:|---:|---:|---:|
| 子谱递推/刷新/delta | 9.40% | **8.30%** | 9.29% | **8.22%** |
| 重组相位乘法/位反转填充 | 11.93% | 12.04% | 11.86% | 12.01% |
| 128 点 FFT 蝶形 | **43.67%** | **44.03%** | **43.43%** | **43.60%** |
| 高低频恢复写回 | 6.25% | 6.59% | 6.22% | 6.64% |
| Hann + 功率谱 | 23.86% | 24.03% | 24.10% | 24.36% |
| Mel 投影 | 4.89% | 5.00% | 5.10% | 5.17% |

delta 的直接效果：

| 用例 | half 递推秒 | delta 递推秒 | 递推降幅 | half 总秒 | delta 总秒 | 总降幅 |
|---|---:|---:|---:|---:|---:|---:|
| default | 1.154 | 1.007 | **12.77%** | 12.271 | 12.122 | **1.22%** |
| large_fft | 2.252 | 1.968 | **12.61%** | 24.249 | 23.936 | **1.29%** |

这解释了 delta 在正式 benchmark 中只有约 0–1% 额外收益的原因：它只优化占全流程约 8–9% 的阶段。
128 点 FFT 仍是最大单项，Hann/功率谱和相位乘法也是重要成本；本轮不继续扩大优化范围。

原始证据：[profile 作业日志](raw/cluster-half-r-profile-85.out)、
[half default](raw/stage-half-default-85.json)、
[delta default](raw/stage-half-delta-default-85.json)、
[half large](raw/stage-half-large_fft-85.json)、
[delta large](raw/stage-half-delta-large_fft-85.json)。

## 4. 最终结论

- 新 half/half-delta 的正确性和性能收益均在同节点集群上得到验证。
- half+delta 是本次正式结果中四组最快的版本，但相对 half 的额外收益只有约 0–1%。
- 若优先选择最简单实现，half 已获得主要收益；若接受很小的 delta 缓冲，half+delta 略优。
- 默认 `src/melspec.cpp` 仍为 V1，实验代码和结果只保留在 `v2/overlap` 分支。
