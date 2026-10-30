# P0：按 Mel 支持区间裁剪 Hann + 功率谱

基线：`4f9f87c`，half+delta SHA-256
`cc1b08d7203b68e482645ad8bfeb6cbcc1cb0da4e697aad63734b0a6b4d26efe`。
本实验独立文件：`melspec_overlap_pruned_power_prototype.cpp`，不包含 FFTW/codelet 改动。

## 原理

先按已有 `first[m]/last[m]` 对每个 Mel 三角的非零区间求并集 `power_bins`。
未加窗频谱 `A[k]` 仍完整生成，因此频域 Hann 所需的 `A[k-1]`、`A[k+1]` 始终存在；
只对最终会被稀疏 Mel 读取的 k 执行三点 Hann 和模平方。
DC 与 Nyquist 保留各自的共轭边界分支，未使用 bin 不会被 Mel 读取。

| 用例 | 使用 bin / 全部正频点 | 比例 | 实际支持范围 |
|---|---:|---:|---|
| default | 1881 / 4097 | 45.91% | 1..1881 |
| small_mels | 1881 / 4097 | 45.91% | 1..1881 |
| large_fft | 3763 / 8193 | 45.93% | 1..3763 |
| narrow_band | 1314 / 8193 | 16.04% | 52..1365 |

这不是裁剪重组 FFT，也不改变滑动递推；只减少 Hann/norm 和其频谱访问。
理论上 Hann/power 工作量分别减少约 54%/84%。

## 正确性

- 本地 8 项 smoke 全部 PASS（`raw/pruned-smoke.txt`）。
- 官方四组全量输出全部 PASS，最大误差与冻结版本一致（`raw/pruned-full-correctness.txt`）。
- 没有用 expected 或音频内容决定索引；索引只由公开 Mel 参数计算。

## 128 点微基准

`microbench_fft128.cpp` 固定随机输入，输入重置不计时，避免重复 FFT 数值溢出。
本地 WSL/i7、GCC 16.2、double、2000 次批迭代：

| 组数 | 原 radix-2 ns/FFT | FFTW batched ns/FFT | FFTW ns/batch |
|---|---:|---:|---:|
| 33 | 410.166 | 218.240 | 7201.909 |
| 65 | 409.363 | 209.475 | 13615.896 |

官方全量 128 点调用次数：8192 为 `192397*33=6,349,101`，
16384 为 `192333*65=12,501,645`。
微基准不含相位填充、恢复、Hann 或 Mel，不能当作端到端加速比。

整程序与阶段数据在 `ab_next.py` 和独立阶段插桩中记录；不与微基准混用。
