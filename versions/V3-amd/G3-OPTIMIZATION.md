# G3 — gfx1201 系统优化

全部数据来自 hepnode0 的普通 Slurm 作业（无 GPU GRES），每次只运行一个短作业。
G2 的官方基线是 **0.904 / 0.663 / 0.988 / 0.991 s**（作业 346）。

## 1. 结论摘要

| 用例 | G2 官方 | G3 冻结（官方） | 变化 |
|---|---:|---:|---:|
| default | 0.904 | **0.689** | −23.8% |
| small_mels | 0.663 | **0.458** | −30.9% |
| large_fft | 0.988 | **0.754** | −23.7% |
| narrow_band | 0.991 | **0.739** | −25.4% |

四项优化按贡献排序：进程级初始化预热（约 190–235 ms/次）、对齐宿主大页提示、
直接 double 输出（去掉 pinned staging 和逐元素转换）、去掉 tail plan。
pinned 输入是明确的负优化，已否决并保留记录。

## 2. 关键发现：190 ms 不是 H2D 带宽

G2 profile 显示 pageable H2D 约 194 ms，但独立微基准（作业 368）中
98.5 MB 拷贝只要 **0.9–5.2 ms（27 GB/s）**，pinned 与 pageable 相同。
继续分解（作业 377–379）后确认它是**每个进程第一次创建 HIP 队列/运行时的一次性成本**：

| 实验 | 现象 |
|---|---|
| probe2 25/49/98/197 MB 连续拷贝 | 1.4 / 2.6 / 5.2 / 7.3 ms，全程无 190 ms |
| probe3 复刻流水线分配顺序 | 首次 H2D **192.4 ms**，同缓冲第二次 **3.8 ms** |
| probe4 先 `hipMemset` 再拷贝 | 首次 memset 173–183 ms，随后拷贝 14 ms |
| probe4 async memset + sync + 拷贝 | 预热 1.8 ms，拷贝 19 ms |
| probe6 先建 stream/malloc 再拷贝 | “setup” 230 ms，随后首次 H2D 16 ms |

即成本落在“进程内第一次真正建立 GPU 队列表”上，和拷贝字节数无关。
`probe3` 里 `resize` 线程被排除后仍是 192 ms，所以也不是 host 侧竞争。

**修复**：在动态初始化阶段（`main` 之前）启动一个 worker 线程，
执行 `hipInit` + 建 stream + 4 KiB async memset，并让主线程在 H2D 前才 join。
这样这笔成本与受保护的 `main.cpp` 读 49 MB PCM、int16→double 归一化重叠。

作业 389 交错 A/B（5 轮最小值，秒）：

| 用例 | prewarm off | prewarm on |
|---|---:|---:|
| default | 0.900 | **0.837** |
| small_mels | 0.653 | **0.537** |
| large_fft | 0.961 | **0.854** |
| narrow_band | 0.965 | **0.852** |

profile 中 `h2d_y_ms` 从 190 ms 降到 **15–17 ms**；`prewarm_wait_ms` 一开始放在函数入口
还有 48–86 ms 的显式等待，移到 H2D 之前后降到 0–6 ms。

另注：早期版本用 `#if` 同步 `hipMemcpy`；现在统一走一条 non-blocking stream 上的
`hipMemcpyAsync` + 一次 `hipStreamSynchronize`，语义不变但不再触发同步路径。

## 3. 宿主大页提示（对齐 MADV_HUGEPAGE）

`output` 是 256×192397 doubles ≈ **394 MB**，`resize` 的逐元素零初始化实测 **208 ms**
（约 96k 次 minor fault，1.9 GB/s），`host_y` 的 98.5 MB 同样受影响。
节点 THP 策略 `always [madvise] never`。

直接对 `vector::data()` 调 madvise 会失败（rc=-1）：glibc 大块分配返回的是页首 +16 字节。
改为把区间**向内对齐到系统页**后再提示，且不覆盖任何分配器元数据，
随后仍然执行完全相同的 `resize`（所有 double 合法初始化，不跳过任何写入）。
这是 CPU 组已经在 `V3-cpu-pipeline-simd` 验证过的同节点结论（0.836→0.701 s）。

作业 392 交错 A/B（5 轮最小值，秒）：

| 用例 | 无提示 | 大页提示 |
|---|---:|---:|
| default | 0.744 | **0.692** |
| small_mels | 0.489 | **0.460** |
| large_fft | 0.805 | **0.741** |
| narrow_band | 0.804 | **0.746** |

profile：`y_convert_ms` 76 → **44 ms**，`resize_wait_ms` 保持 ~0.02 ms。
提示失败时无副作用，非 Linux 走原路径。

## 4. 直接 double 输出

G2 的输出路径是“GPU 写 float → pinned staging 分块 D2H → 宿主 float→double 转换”，
profile 里 `staging_alloc_ms` 约 19–22 ms、`host_convert_ms` 约 48 ms、合计 ~77 ms。
gfx1201 的 D2H 带宽足够，直接让 Mel kernel 写 double、一次 394 MB 拷贝进 `output` 更快：

| 输出路径（default / large_fft 的 async_output_total） | 时间 |
|---|---:|
| pinned staging + 转换 | 76.5 / 73.8 ms |
| 直接 double D2H | **33.8 / 18.7 ms** |

作业 390 的端到端最小值：base 0.774 / 0.480 / 0.784 / 0.792，
direct 0.761 / 0.475 / 0.781 / 0.768。误差与 float 输出完全一致
（float 累加在 kernel 内完成，只在最后一步拓宽成 double）。

## 5. 去掉 tail plan

原来对 `n_frames % capacity` 单独建一个 plan，即使 tail == capacity 也会建，
耗时约 10–15 ms。现在只用 full plan：gather 对超出 `n_frames` 的帧写 0，
FFT 固定跑 `capacity` 个变换，Mel kernel 仍只写 `batch` 个有效帧。
多算的 FFT 数量 ≤ 511 个，代价 < 0.1 ms。

## 6. 否决的实验（保留原始证据）

| 实验 | 结果 | 结论 |
|---|---|---|
| `HIP_PINNED_INPUT=1` | 0.995 / 0.728 / 1.075 / 1.054 | `std::copy` 写入 pinned 只有 ~2.4 GB/s，`y_convert` 49→122 ms，H2D 不变 |
| pinned + batch256 | 与 pinned 相同量级 | 打包后无法单独归因，仍更慢 |
| `HIP_ASYNC_OUTPUT=0` | 1.10 / … / 1.16 | 去掉异步输出立即变慢 |
| 单纯 async H2D（无预热） | 0.837 / 0.537 / 0.854 / 0.852 | 有效但只到这一步，真正原因是进程级初始化 |
| 直接对 `data()` madvise | rc=-1，无效果 | 地址未对齐，必须向内对齐 |
| THP `enabled=always` | 全机策略，不修改 | 只用每进程 `madvise` 提示 |

## 7. 最终 profile（作业 395，default）

`y_convert 44.9 · h2d 17.7 · plan 26.2 · gather 25.7 · fft 30.3 · mel 35.1 ·
d2h 23.8 · window_fb 9.0 · device_alloc 1.1 · prewarm_wait 6.5 ms`，
`compute_total 220.9 ms`（G2 为 509 ms）。

`compute_total` 之外的 ~470 ms 是受保护 `main.cpp` 的读 PCM/int16→double 与
float 写盘、以及进程加载/退出，不属于本后端可改范围；它现在已是主要成本。

## 8. 参数选择

- batch：8192 点用 512，16384 点用 256（G1 微基准 + 端到端复核）。
- 线程块：gather 256、power 256、Mel 256（8 个 wavefront/block），`warpSize=32` 归约。
- 未采用 CUDA 的全局 batch512；16384 点在 AMD 上 512 明显退化。
