# V3 CUDA 最终性能冲刺（独立实验分支）

冻结基线：`3e11194688b4b581b8ca527c5d292dbee451b0f1`。
官方成绩 0.782 / 0.536 / 0.800 / 0.800 秒，四组 PASS。本分支只做实验，不改变冻结候选身份。

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
不属于稳定的单项获益。后台分配与 host 准备/转换争用内存带宽，抵消了重叠收益。**不采用。**

## B. 输入转换移到 GPU

实现 `CUDA_GPU_INPUT_CONVERT=1`：主机直接 H2D 传输完整 double y（197 MB），
GPU grid-stride kernel 一次性转成持久 float y 缓冲，Gather 仍读取同一 float 缓冲；
没有每帧读取 double 或重复转换。FP32 only，保留完整容差。

| 方案 | default min/median | small min/median | large min/median | narrow min/median |
|---|---|---|---|---|
| stable | 0.8038 / 0.8058 | 0.5327 / 0.5401 | 0.8089 / 0.8255 | 0.8023 / 0.8096 |
| B GPU input convert | 0.7936 / 0.7958 | 0.5341 / 0.5395 | 0.7960 / 0.8302 | 0.8021 / 0.8043 |

B 的 default 约有 1.2% 改善，small/narrow 接近持平，但 large 的 median 回退；
没有稳定跨四组的收益。double H2D 翻倍字节与省下的 50 ms CPU 转换基本抵消。**不采用。**

## AB 组合

| 方案 | default min/median | small min/median | large min/median | narrow min/median |
|---|---|---|---|---|
| AB | 0.7897 / 0.8036 | 0.5427 / 0.5455 | 0.8035 / 0.8421 | 0.8040 / 0.8176 |

组合没有带来叠加收益，large/narrow median 甚至明显回退。按“单独获益才组合”规则，
**AB 不采用**。

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

## 结论

- 最终采用版本仍是冻结的 `3e111946`，不改默认源码。
- A/B 实验保留在 `v3/cuda-sprint`，有完整 smoke、sanitizer、原始 A/B 和 profile。
- 当前 GPU 已超过新的 CPU B16 目标；再优化应针对 device alloc/y 转换/host 转换等固定成本，
  但 A/B 已说明简单后台化/GPU 输入转换没有稳定收益，不建议继续增加复杂度。
- 若继续，只能做更细的分配器复用/多线程主机转换实验，并需严格同作业多轮验证。
