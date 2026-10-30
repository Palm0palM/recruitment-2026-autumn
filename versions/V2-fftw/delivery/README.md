# 交付索引

| 文档 | 内容 |
|---|---|
| [FROZEN.md](FROZEN.md) | 冻结哈希、干净 checkout 构建、Wisdom 部署、运行与校验 |
| [FREEZE.json](FREEZE.json) | 机器可读冻结清单（源码/库/可执行文件/Wisdom SHA） |
| [DEFENSE.md](DEFENSE.md) | 答辩用优化历程、各版本 Benchmark、关键技术、失败实验 |
| [AB.md](AB.md) | 与 SIMD Batch8、旧版 A2 的同节点交错 A/B 运行说明 |
| [../experiments/RESULTS.md](../experiments/RESULTS.md) | 两轮全部集群数据、验收状态、内存 |
| [../experiments/DESIGN.md](../experiments/DESIGN.md) | 四项实验的数学与工程原理 |
| [../experiments/FAILURES.md](../experiments/FAILURES.md) | 失败构建/测量与原因 |
| [../experiments/REPRODUCE.md](../experiments/REPRODUCE.md) | 完整复现与集群作业顺序 |

## 当前候选

- 代码：`d8234a5`（其后仅文档/脚本变化）。
- 构建：`FFT_PRECISION=float FFT_POWER=float FFTW_WISDOM=1`。
- 运行：加载 PATIENT Wisdom；失配时安全回退 ESTIMATE。
- 成绩：Job 151 五轮同作业中位数，几何平均相对 A2 **1.313×**；Job 209 官方 **4/4 PASS**。
- 默认 `make` 仍为 double A2，float 是显式选项。
