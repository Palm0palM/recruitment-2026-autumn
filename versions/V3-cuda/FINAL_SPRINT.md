# V3 CUDA 最终性能冲刺（独立实验分支）

**状态：已封存（2026-10-31）。A/B/AB 均未采用；最终 CUDA 解仍是冻结的 `3e111946`。**
冻结基线：`3e11194688b4b581b8ca527c5d292dbee451b0f1`。
官方成绩 0.782 / 0.536 / 0.800 / 0.800 秒，四组 PASS。本分支只做实验，不改变冻结候选身份。
A/B 代码以 `CUDA_BACKGROUND_INIT=0`、`CUDA_GPU_INPUT_CONVERT=0` 默认关闭保存，仅用于复查/答辩记录，
后续不再基于它们继续调参。`v3/cuda` 分支的稳定源码和默认构建保持不变。

## A. GPU 初始化后台化

实现 `CUDA_BACKGROUND_INIT=1`：后台线程构造全部 device buffers 与 full/tail cuFFT plans；
主线程并行完成 y 转换、窗口/Mel 构建和 output resize。使用异常捕获、RAII join 和统一
`output_elements`，避免资源生命周期/同步错误。四路 smoke 和 compute-sanitizer 0 错误。

L40 同作业 5 轮交错（作业 321，hepnode3，秒）：

| 方案 | default min/median | small min/median | large min/median | narrow min/median |
|---|---|---|---|---|
| stable | 0.8038 / 0.8058 | 0.5327 / 0.5401 | 0.8089 / 0.8255 | 0.8023 / 0.8096 |
| A background init | 0.7947 / 0.8077 | 0.5379 / 0.5415 | 0.8069 / 0.8116 | 0.7993 / 0.8018 |

A 只在 large/narrow 有约 0.4–1.7% 的方向性改善，default/small 的 min 或 median 有回退，
不属于稳定的单项获益。**不采用。**

失败机理：稳定版已经把 output.resize 和 context 轻量初始化并行化；A 把真正的
`cudaMalloc` 和 plan 构建也移到后台后，它们与 host_y 转换、Mel 构建和 output 的 memset
同时争用主机内存带宽与线程调度。主线程在 `device_init.finish()` 的残余等待并未稳定缩短。
本机诊断中 `device_alloc_wait` 反而达到约 187–277 ms；L40 A/B 也未出现稳定端到端收益。
这不是同步错误——A 路径 smoke 全过、sanitizer 0 错误——而是并行化后的资源竞争抵消了收益。

## B. 输入转换移到 GPU

实现 `CUDA_GPU_INPUT_CONVERT=1`：主机直接 H2D 传输完整 double y（197 MB），
GPU grid-stride kernel 一次性转成持久 float y 缓冲，Gather 仍读取同一 float 缓冲；
没有每帧读取 double 或重复转换。FP32 only，保留完整容差。

| 方案 | default min/median | small min/median | large min/median | narrow min/median |
|---|---|---|---|---|
| stable | 0.8038 / 0.8058 | 0.5327 / 0.5401 | 0.8089 / 0.8255 | 0.8023 / 0.8096 |
| B GPU input convert | 0.7936 / 0.7958 | 0.5341 / 0.5395 | 0.7960 / 0.8302 | 0.8021 / 0.8043 |

B 的 default 约有 1.2% 改善，small/narrow 接近持平，但 large 的 median 回退；
没有稳定跨四组的收益。**不采用。**

失败机理：CPU y 转换约 50 ms 被完全消除，但完整 double y 的 H2D 从约 98.5 MB 变为
197 MB；GPU 一次性转换本身只需约 2–3 ms。主机页可固定的 pageable H2D 增量、GPU 转换与额外的
double 设备缓冲基本抵消了节省。正确的详细 Profile 本地样本中 y CPU 转换=0、H2D 约16–17 ms、
GPU convert 约2.3–2.8 ms；这不是集群成绩，只用于确认成本来源。
B 没有逐帧读取 double，也没有改变 FFT/Mel 数学；不采用是端到端收益不足，不是正确性问题。
若输入本来已是 float 或 H2D 能使用 pinned/异步高带宽路径，B 的结论可能不同；本题受 main 的
double `y` 接口约束，不能改造其输入所有权。

## AB 组合

| 方案 | default min/median | small min/median | large min/median | narrow min/median |
|---|---|---|---|---|
| AB | 0.7897 / 0.8036 | 0.5427 / 0.5455 | 0.8035 / 0.8421 | 0.8040 / 0.8176 |

组合没有带来叠加收益，large/narrow median 甚至明显回退。按“单独获益才组合”规则，
**AB 不采用**。A 的分配竞争与 B 的 H2D 增量叠加后，`compute` 内的内存带宽冲突更明显。

## 与新的 CPU 目标 SIMD FP32 Batch16 同作业比较

CPU 源：`v2/simd` 提交 `e448c989c921762884582804efe23d6607678ff9`，
以 `MELSPEC_FP32_BATCH=16` 在 hepnode3 重新编译；四组完整 expected PASS。

| 用例 | SIMD FP32 B16 min/median | 冻结 GPU min/median | GPU vs CPU（min / median） |
|---|---|---|---|
| default | 0.8293 / 0.9374 | 0.8038 / 0.8058 | 1.032× / 1.163× |
| small_mels | 0.5824 / 0.5960 | 0.5327 / 0.5401 | 1.093× / 1.104× |
| large_fft | 1.2646 / 1.3076 | 0.8089 / 0.8255 | 1.564× / 1.584× |
| narrow_band | 1.1573 / 1.2280 | 0.8023 / 0.8096 | 1.443× / 1.517× |

同一 L40 作业和相同 cpuset 下，冻结 GPU 候选四组均快于 SIMD FP32 Batch16；
这次 small_mels 也不仅通过 Hybrid 而直接领先。CPU 的绝对时间受该作业/节点
分配影响，不能用它替代其自身正式报告；但同作业交错比较是有效的。
A/B/AB 均未改变这一结论。

## 证据索引

- 完整 A/B 原始轮次与 min/median：`raw/final-sprint/sprint-ab-321.out/.json`。
- A/B/AB 与 stable 的 smoke：`raw/final-sprint/cuda-*-smoke.txt`。
- compute-sanitizer 0 错误：`raw/final-sprint/cuda-*-sanitizer.txt`。
- 本机 A/B：`raw/final-sprint/local-sprint-A-B.txt/.json`。
- A/B 本机阶段 Profile（含分配等待/GPU 输入转换计时）：`raw/final-sprint/profile-A-local.txt`、
  `profile-B-local.txt`。更新后的正确 H2D/B Profile：`raw/final-sprint/profile-B-local-fixed.txt`。
- CPU 对照源码与编译身份：`v2/simd` 的 `e448c989c921762884582804efe23d6607678ff9`，
  `MELSPEC_FP32_BATCH=16`；B16 源码不复制进本仓库，远端 AB 脚本从只读导出文件编译。

## 结论与封存

- 最终采用版本仍是冻结的 `3e111946`，不改默认源码；`v3/cuda` 的 276 官方结果继续有效。
- A/B/AB 都经历“默认关闭 → 独立 smoke/sanitizer → 同作业 5 轮 A/B → 单变量判定”；
  没有一个能在保持稳定的前提下降低四组端到端时间，因此不进入默认候选。
- A 的正确性/异常处理实现可复用，但性能方向无足够收益；B 仅适用于可改变输入表示或
  H2D 成本更低的部署，不适用于当前受保护 main 的 double 输入接口。
- 当前 GPU 已超过新的 CPU B16 目标；再优化应针对 device alloc/y 转换/host 转换等固定成本，
  但 A/B 已说明简单后台化/GPU 输入转换没有稳定收益，不建议继续增加复杂度。
- **CUDA 路线到此结束**：不再提交 A/B 后续实验，不再合并 `v3/cuda-sprint` 到 `v3/cuda`。
  A/B 原代码与失败记录保留在本分支，供答辩和后续 V4 参考。
