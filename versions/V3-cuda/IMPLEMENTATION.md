# V3 CUDA 实现记录（G1–G3 首候选）

## 设计与接口

当前候选不继承 CPU FFT 内核，使用 cuFFT 原生 R2C/D2Z；`src/main.cpp`、`validate.py`、
`include/melspec.h` 和 `data` 均未修改。默认 `make` 仍构建 V1 CPU；
`make CUDA=1 CUDA_PRECISION=32/64 CUDA_BATCH=... CUDA_FUSED=...` 构建独立 CUDA 候选。

流程：CPU main 已完成 int16→double 归一化；主机只转换一次完整 y 并 H2D，GPU 按 hop 直接 gather
重叠帧并施加周期 Hann；cuFFT 自然紧凑 `[batch][n_fft]` 输入与 `[batch][n_fft/2+1]` 输出；
GPU 稀疏 Mel 直接写 `[n_mels,n_frames]`；只在最后回传完整输出并恢复 host double vector，
由原 main 完成 float32 小端写盘。没有 CPU 重叠帧矩阵或全量功率谱回传。

- FP32：CUFFT_R2C，FFT/Mel/输出均为 float；D2H 后转换 double。
- FP64：CUFFT_D2Z，全程 double，直接 D2H 到 output。
- `CUDA_FUSED=1`：每个 Mel tap 即时计算 |FFT|²，不物化功率矩阵；相邻滤波器会重复读 bin。
- `CUDA_FUSED=0`：单独功率 kernel + Mel，比较多一次功率写/读与减少重复计算的取舍。
- 一个 warp 负责一个 frame/Mel 内积，支持 first/last 连续区间，空滤波器返回零。
- full batch 与精确 tail batch 各有 plan；所有 flat index 用 size_t，单流顺序执行。
- n_fft=1 按周期窗定义直接输出零，短输入/非法参数在计算前拒绝。

## 子任务审查

三个 Kimi worker 子会话均经 OpenCode API 验证模型为
`ppio/moonshotai/kimi-k3.1-pro-highspeed#high` 且启用 Thinking：

- cuFFT probe：`ses_e743bd5d3ffd0yIm2ig6Eslyjw`
- GPU Pipeline：`ses_e743bd5d4ffeuMVLrNM0W3gqpg`
- Precision/Memory：`ses_e743bd5d2ffdquWRGLt18ZReMI`

A 的 plan_many/事件计时、B 的 gather/warp reduction/fused kernel 经主 Agent 审查后集成。
所有子任务只写私有 tmp 实验目录，没有修改正式源码或自行操作集群。

## 本地正确性

本地 GPU 为 RTX 5060（sm_120），只用于功能验证，不作为集群性能依据。FP32/FP64 × fused/separate
四种组合均通过 `scripts/smoke.py` 的 8 项测试（随机 PCM 官方四组、CLI 默认、n_fft=1/2/8）。
FP32 最大绝对误差示例：narrow_band 小样本 `0.0012207031`，仍满足官方容差；FP64 与 V1 一致。
原始日志见 `raw/local-cuda-*-smoke.txt`。

早期本机以 sm_89 构建时因 GPU 架构不匹配而失败，改用本机 sm_120 后通过；
集群始终使用 sm_89，不能把本机架构或性能混入正式结果。

## cuFFT L40 微基准（作业 133）

A 的独立 probe 在 hepnode3/L40 上通过两种精度的 8 点直接 DFT 和 tail-batch 检查。
自然布局 `idist=n_fft, odist=n_fft/2+1`，plan 与 warm execution 分开计时，CUDA events，3 次预热 + 10 次执行。

主要观察（不是端到端成绩）：

- FP32 远快于 FP64；例如 8192/batch256 执行均值约 0.0124 ms vs 0.1347 ms。
- 8192 随 batch 增大继续改善；16384/batch1024 明显退化（FP32 均值约 0.2426 ms，
  batch512 约 0.0383 ms），因此后续优先 128/256/512，不采用“越大越好”的假设。
- 微基准不含分帧、Mel、H2D/D2H、main 读写；不能据此推算最终加速比。
- 完整 CSV/内存/workspace 见 `raw/cufft-probe-133.out`。

## 同节点 CPU 基线（作业 125）

固定 FFTW A2 commit `b4552f3962751b5a02d2aa56798907c5a1c29afc` 在 hepnode3 独立重建，
官方四组全 PASS，三次最小 wall time：default 1.004 s、small_mels 0.944 s、
large_fft 1.759 s、narrow_band 1.753 s。该作业 cpuset 为 `0-3,16-19,32-35,48-51`，
与旧 hepnode2 基线不同，故后续 GPU 对比必须使用同环境实测，不照搬旧秒数。
原始构建哈希与官方输出见 `raw/baseline-a2-*`。

## 精度、内存与初始化重叠补充

FP32/FP64 在本地均通过 74/74 合成用例及 4/4 官方音频检查，原始日志见
`raw/precision-*.txt`。私有 harness 初次读取本地断开的题目 data 链接而失败，改用只读
`tmp/data` 后通过；该失败不是 CUDA 数值错误，也没有修改题目链接。

G4 初始化重叠变体将 `cudaFree(0)` 放到独立线程，与 host 窗口/滤波器/输出初始化重叠。
本地五次最小值从 serial 的 0.933/0.514/1.036/1.022 秒降至 0.856/0.455/0.807/0.850 秒，
输出全部 PASS。阶段诊断中 `device_alloc` 从约 232 ms 降到约 2 ms，证明主要收益来自提前创建 context。
这只是本机结果，集群采用前仍需正式验证。Pinned 变体在不同用例上有正有负，不设为默认。
后续 Kimi 复审指出的初始化成员顺序、profile 错误检查、batch 边界和陈旧二进制问题已修复，
正式采用以修复后 commit 为准。详见 [REVIEW.md](REVIEW.md)。
