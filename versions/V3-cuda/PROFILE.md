# V3 CUDA Profile 与瓶颈迁移

## 首次正式候选（作业 202，FP32/b256/fused/pageable）

hepnode2/L40，另编译 `CUDA_PROFILE=1`，四组输出全部 PASS。下表单位 ms；
`total` 只含 `compute_melspectrogram_cuda`，不含 main 的读取和最终写出。

| 阶段 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| host 准备 | 256.1 | 99.1 | 268.6 | 266.8 |
| CUDA context/常驻分配 | 173.3 | 150.6 | 145.5 | 145.0 |
| 静态 H2D | 10.4 | 9.7 | 10.2 | 10.3 |
| batch 分配 | 0.2 | 0.2 | 0.2 | 0.2 |
| cuFFT plan | 3.0 | 4.8 | 5.4 | 3.3 |
| gather + Hann | 9.7 | 9.7 | 13.6 | 13.6 |
| cuFFT 执行 | 11.0 | 11.0 | 17.7 | 17.7 |
| power + Mel | 13.2 | 8.0 | 13.8 | 12.9 |
| 末尾检查/同步 | 34.0 | 28.7 | 45.2 | 44.4 |
| D2H/staging | 126.9 | 31.2 | 125.3 | 124.5 |
| host 转 double | 47.7 | 11.9 | 48.3 | 47.5 |
| **compute 总 wall** | **660.4** | **338.0** | **656.2** | **649.0** |
| 诊断全进程 wall（秒） | 1.200 | 0.708 | 1.180 | 1.220 |

这些阶段是 host wall + launch/sync 的诊断，不是纯 kernel event 时间；正式 benchmark 未启用插桩。
FP32 R2C plan workspace 在本次 batch256 上报告为 0；这不是总显存。
`cudaMemGetInfo` 的设备级 used 约为 default 789 MB、small 636 MB、large/narrow 810 MB，
包含 context、完整输入/输出和 batch 缓冲，不是只算 cuFFT workspace。

## 初始化重叠的机制

default 的 gather+FFT+Mel 只有约 34 ms，而 context/分配约 173 ms，host 准备约 256 ms。
因此继续微优化 FFT 不会解决 default 的主要成本。将 `cudaFree(0)` 放到独立线程，
与 host 输出初始化、窗口、滤波器和 y 转换重叠，能直接消除串行 context 等待。

本地同类诊断中，`device_alloc` 从约 232 ms 降到约 2 ms；正式 215/226 的端到端结果
与此方向一致。初始化重叠只改变启动调度，不改变 FFT、Mel、精度或输出。

## 传输与存储口径（FP32）

- 完整 y 一次 H2D：`24,634,880 × 4 = 98,539,520 B`（约 94.0 MiB）；没有 CPU 重叠帧矩阵。
- default/large/narrow 输出一次 D2H：`197,014,528 B`（约 187.9 MiB）；small_mels 为 49,253,632 B。
- default Mel 权重：256×4097×4 = 4,195,328 B；large/narrow：256×8193×4 = 8,390,656 B。
- host 仍持有 main 的 double y（约 187.9 MiB）、double 输出（约 375.8 MiB）和写盘 float32 缓冲（约 187.9 MiB），
  另外还有 CUDA 输入/输出 staging；不能只统计 GPU 显存。
- fused 不分配 batch 功率矩阵；separate 另需 `batch × n_bins × 4 B`，并有额外功率写/读。

D2H/staging 与 host 转换在 default 约 175 ms，small_mels 约 43 ms，说明输出大小直接决定固定成本。
Pinned 的分配/收益随用例变化，不能把它当作普遍优化；最终默认仍 pageable。

## 下一瓶颈判断

1. **不是继续优化 cuFFT kernel**：它只占 default 全进程很小一部分。
2. **启动/主机初始化是第一阶段重点**：已由初始化重叠解决一大块串行等待。
3. **D2H + host staging/转换 + main 的读写是后续主要固定成本**。
   `src/main.cpp` 不可修改，不能用违规接口规避；pinned/异步只能在实测有效时采用。
4. batch128/256/512 的差异已经小于启动和输出成本，继续增大 batch 没有普遍价值。
5. streams 只有在能证明改善真实端到端结果时才值得增加；多流需要独立 plan/buffer，
   不能共享一个 plan 并发执行，也不能破坏 row-major 输出和尾批正确性。

## 最终候选（作业 227，FP32/b512/fused/初始化重叠）

与官方 226 同源码同参数，另开 `CUDA_PROFILE=1`。四组输出全部 PASS。

| 阶段（ms） | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| host 准备 | 262.6 | 102.6 | 267.4 | 266.5 |
| context/常驻分配 | **0.445** | 41.959 | **0.436** | **0.418** |
| 静态 H2D | 10.0 | 9.8 | 10.2 | 10.2 |
| batch 分配 | 0.224 | 0.182 | 0.242 | 0.227 |
| plan | 3.1 | 2.9 | 3.5 | 3.3 |
| gather | 6.9 | 7.0 | 11.8 | 13.2 |
| cuFFT | 7.5 | 7.4 | 24.6 | 25.1 |
| power + Mel | 10.3 | 5.1 | 11.2 | 10.5 |
| 末尾检查/同步 | 24.8 | 19.6 | 47.6 | 48.9 |
| D2H/staging | 125.9 | 31.8 | 124.5 | 126.1 |
| host 转 double | 47.5 | 11.8 | 47.5 | 47.5 |
| **compute 总 wall** | **481.9** | **222.7** | **508.5** | **512.1** |
| 诊断全进程 wall（秒） | 1.021 | 0.596 | 1.035 | 1.061 |

default/large/narrow 的常驻分配已降到约 0.4–0.45 ms，small_mels 本次仍约 42 ms，
说明重叠是消除可隐藏的等待，不是保证所有场景都零初始化成本。
设备级 used 约 default 806 MB、small 652 MB、large/narrow 843 MB；FP32 workspace 仍报告 0。

**V3 后续瓶颈已明确转为 host 准备、D2H/staging、host 转换和 main 固定读写。**
继续投入 cuFFT 微内核或多流 FFT 的预期收益远小于这些固定成本；如果再做 V4，
应优先验证合法的输出/传输流水和主机端初始化，而不是单纯追求更高理论 GPU 算力。

## 异步输出最终候选（作业 277）

与官方 276 同源码、同默认；`CUDA_PROFILE=1` 独立构建，四组输出全部 PASS。
`resize_wait` 是 join 后残余等待，不包含已与 host 准备并行的时间；`d2h_ms` 是 event 等待，
由于预取，通常已被其他 CPU 工作隐藏。`async_output_total_ms` 包含 staging 分配、输出 resize
（若在异步块内）、传输与 CPU 转换，但部分工作重叠，不能与各子项直接相加。

| 阶段（ms） | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| y double→float | 50.8 | 48.3 | 49.8 | 49.4 |
| 窗口/滤波器 | 8.9 | 2.3 | 18.4 | 18.2 |
| output.resize 等待 | **0.001** | **0.001** | **0.001** | **0.001** |
| CUDA context/常驻分配 | 109.5 | 96.8 | 84.3 | 82.3 |
| 静态 H2D | 9.6 | 8.8 | 10.1 | 11.8 |
| batch 分配 + plan | 3.4 | 3.1 | 3.7 | 3.6 |
| gather | 6.9 | 7.0 | 11.7 | 11.8 |
| cuFFT | 7.5 | 7.4 | 24.5 | 24.9 |
| power + Mel | 10.4 | 5.1 | 11.2 | 10.4 |
| pinned staging 分配 | 10.2 | 10.2 | 10.2 | 10.2 |
| D2H event 等待 | 0.5 | 0.5 | 0.5 | 0.5 |
| CPU float→double | 48.8 | 12.0 | 48.3 | 48.9 |
| 异步输出总区间 | 59.7 | 22.7 | 59.2 | 59.8 |
| **compute 总 wall** | **266.9** | **201.8** | **273.1** | **272.4** |
| 诊断全进程 wall（秒） | 0.823 | 0.571 | 0.807 | 0.831 |

output.resize 的串行等待已从约 160–260 ms 降到约 **1 µs** 残余；D2H 传输与 CPU 转换重叠后，
异步输出总区间约 60 ms（small_mels 约23 ms），而此前单次 D2H 126 ms + 转换 48 ms。
正式 276 的进程总时间低于诊断 wall，是因为诊断每阶段同步、输出校验和日志产生额外开销。

### 剩余瓶颈与是否继续

1. **CUDA context/常驻 buffer 分配仍约 82–110 ms**，是 compute 内最大的单项。
   当前只用 `cudaFree(0)` 提前创建 context，真正的 `cudaMalloc` 和 plan 仍在 host 准备之后。
   下一步可将 device buffer 分配与 plan 构建整体移入后台初始化线程，与 host 准备重叠；
   这是有明确上限的独立实验。
2. **y double→float 约 49 ms、输出 float→double 约 48 ms**。可用独立线程或并行转换，
   但会与输出 resize/设备分配争内存带宽，必须在集群 A/B，不能凭局部 microbench 决定。
3. **main 固定的 PCM 读取与 float32 写盘约 0.18 + 0.22 s**（历史 V2 profile），不可修改。
   当前 0.78–0.80 s 已接近这部分固定成本 + 约 0.2 s 主机/GPU 工作的量级。
4. cuFFT/Gather/Mel 合计只有约 **25–47 ms**，继续优化 GPU kernel 的端到端上限很低。
5. CUDA Graph 最多只能影响 kernel launch 区间（本 profile 中 gather+FFT+Mel 的量级），
   在 host 固定成本占主导时预计收益数毫秒，暂不值得引入生命周期复杂度。
   GPU 端 double 输出会把 D2H 字节数翻倍，仅省掉约49 ms CPU 转换，按当前约60 ms 异步输出区间
   也没有收益证据，不采用。

**结论：本轮达到 large/narrow 稳定领先，四组几何平均也超过 FFTW PATIENT。**
若继续，最值得做的是“设备分配/计划后台化”，然后是多线程主机类型转换；收益空间已从数百毫秒
缩小到约几十毫秒，属于有界收尾优化，不再是必须项。
