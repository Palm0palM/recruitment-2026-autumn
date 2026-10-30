# P1-A：FFTW batched C2C 重组内核

独立实现：`melspec_overlap_fftw_prototype.cpp`，从冻结 half+delta 出发，
不包含 P0 或自研 codelet。链接依赖：double precision `libfftw3`。

## 数据布局与 Plan

每帧 `r=0..B/2` 的输入按自然顺序 `[residue][q]` 写入连续 `fftw_complex`。
每线程一个 `fftw_plan_many_dft`：

- rank=1、长度 128、howmany=B/2+1；
- 输入输出同址，stride=1、dist=128；
- `FFTW_FORWARD | FFTW_ESTIMATE`；
- 数组由 `fftw_malloc` 分配，Plan 在启动 worker 前创建。

因此不再执行 radix-2 位反转填充，每帧只调用一次 `fftw_execute`，
由 FFTW 完成 33 或 65 个独立 C2C。每个线程持有独立 Plan 和数组，
 planner 不在 worker 内并发调用；没有共享可写 FFTW 状态。

额外内存：每线程约 `(B/2+1)*128*16 B`（64/128 KiB）及 FFTW 内部计划。
本地 33/65 组 ESTIMATE 初始化约 0.2–0.5 ms/Plan，16 线程为一次性毫秒级成本，
与全量运行相比很小，但计入端到端时间。

## 数学与正确性

滑动子谱、r/B-r 共轭恢复、完整 m 输出和频域 Hann 均不变。
FFTW 输出自然 m 顺序，直接用于 DC/Nyquist、低频组和高频共轭恢复。
本地 8 项 smoke 与官方四组全量 expected 全部 PASS，见
`raw/fftw-smoke.txt`、`raw/fftw-full-correctness.txt`。

## 对照边界

`microbench_fft128.cpp` 的 FFTW 部分使用同样的 many-DFT 形式，
但微基准使用 `FFTW_UNALIGNED` 适配 vector，正式候选使用 fftw_malloc 对齐数组，
所以微基准只用于筛选，不与端到端数字混用。
组合版 `melspec_overlap_fftw_pruned_prototype.cpp` 是后续独立集成，
不用于归因本文件的单独收益。
