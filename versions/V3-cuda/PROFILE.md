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
