# Host preparation / D2H / conversion pipeline 实验

基线：FP32 batch512 fused 初始化重叠稳定版；受保护文件未修改。

## 1. 细粒度 Profile 与合法重叠

新增诊断字段：y 转换、窗口/滤波器和 output.resize 分开记录；staging 分配、CUDA event 等待、
CPU double 转换、异步输出总时间分别记录。正式二进制不启用计时。

原始稳定版（本机 RTX 5060，`CUDA_PROFILE=1`）显示 default：
y 转换约 101 ms、窗口/滤波器 16 ms、output.resize 220 ms、D2H 167 ms、float→double 53 ms；
gather+cuFFT+Mel 只有约 143 ms（profile 自身分阶段同步会放大该数）。因此继续优化 FFT 没有意义。

实施三种独立开关：

- `CUDA_DEFER_RESIZE=1`：GPU kernel 入队后、同步前执行 output.resize，与 GPU 计算重叠。
- `CUDA_ASYNC_OUTPUT=1`：按 `CUDA_STAGING_ROWS` 行切块，两套 pinned staging、两个非阻塞 stream、
  双 event 预取；D2H 与 CPU float→double 转换流水化。
- `CUDA_RESIZE_THREAD=1`：独立线程执行 output.resize，与 context 初始化、y 转换和滤波器构建重叠；
  与 defer resize 互斥。

所有路径使用统一 `output_elements = n_mels*n_frames`。首版延迟 resize 暴露了一个真实 bug：
device buffer 大小误用尚未创建的 `output.size()`，导致空指针 GPU 写。compute-sanitizer 定位后改用
独立元素数；修正版四路 smoke 全 PASS，async/defer 组合 memcheck **0 errors**。

## 2. 本机 A/B（5 次取最小，不是集群成绩）

| 方案 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| stable | 0.909 | 0.575 | 1.016 | 1.017 |
| defer resize | 1.027 | 0.634 | 1.093 | 0.934 |
| async D2H | **0.878** | **0.565** | 0.860 | 0.924 |
| async + defer resize | 0.905 | 0.608 | **0.854** | **0.897** |

defer resize 单用不稳定且常变慢；异步 pinned 输出最稳定。async+defer 只有部分用例进一步改善，
可能与本机噪声/内存带宽竞争有关。

## 3. L40 第一次正式 A/B（作业 270，hepnode2，5 轮最小）

代码 `268dda4`，FP32 batch512 fused 初始化重叠。所有比较均校验四组完整 expected。

| 方案 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| 同作业 A2 | 1.162 | 0.893 | 1.781 | 1.691 |
| stable GPU | 1.013 | 0.571 | 1.035 | 1.045 |
| defer resize | 1.081 | 0.596 | 1.070 | 1.085 |
| **async r16** | **0.880** | **0.539** | **0.917** | **0.912** |
| async + defer | 0.955 | 0.572 | 0.954 | 0.951 |
| async r8 | 0.952 | 0.573 | 0.945 | 0.951 |
| async r32 | 0.975 | 0.598 | 0.967 | 0.974 |

结论：pinned 双缓冲 + 异步 D2H/CPU 转换有效；16 行 staging 优于 8/32；defer-only 无收益，
async+defer 反而比 async-only 差。中间值可能预热或内存带宽噪声影响，不将差异夸大为算法结论。

同一作业中 async GPU 相对 A2 约 `1.32 / 1.66 / 1.94 / 1.85×`。
相对外部记录的当前 FFTW float+PATIENT（0.826/0.491/1.024/0.976）：

- default：GPU 仍慢约 6.5%
- small_mels：GPU 仍慢约 9.8%
- large_fft：GPU 快约 **10.4%**
- narrow_band：GPU 快约 **6.6%**

这达到“优先争取 large/narrow 稳定领先”的目标；不宣称所有用例全面胜出。
不同作业的 FFTW/GPU 绝对秒数仍有口径差异，最终需以本实验最终候选的官方 4/4 和后续 Hybrid 分派为准。

## 4. L40 resize-thread A/B（作业 272，5 轮最小）

代码 `c20d3e6`。同一作业交错，所有输出均 PASS。`resize_thread` 让独立线程在 host 准备阶段
并行完成 output.resize，再与 async staging 组合。

| 方案 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| 同作业 A2 | 1.010 | 0.885 | 1.796 | 1.713 |
| stable | 1.112 | 0.612 | 1.124 | 1.129 |
| resize thread only | 0.925 | 0.564 | 0.913 | 0.923 |
| async only | 0.966 | 0.577 | 0.993 | 0.986 |
| **async + resize thread** | **0.796** | **0.529** | **0.793** | **0.796** |

`async+resize` 四组均为最快，且相对 stable 分别提升约 28%/14%/29%/29%；
相对同作业 A2 约 1.27×/1.67×/2.26×/2.15×。resize thread 本身也有明显收益，但不是全部；
与异步 pinned 输出结合后收益叠加。异步输出路径的四组完整 expected 在主 sweep 中均 PASS；
另在本地 74 项合成/边界测试中 async 路径全部通过。
