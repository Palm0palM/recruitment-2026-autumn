# V2-overlap 最终归档

日期：2026-10-30。分支 `v2/overlap`，归档起点 `5bd0c38`。
默认 `src/melspec.cpp` 始终为 V1；所有 Overlap 实现都在独立实验文件中，不参与 Make 构建。
路线主动优化到此结束，不再提交新的性能实验或集群作业。

相关材料：

- [P0 Mel 支持裁剪技术说明](P0-mel-pruning.md)
- [答辩精简材料](DEFENSE.md)
- [half-r 数学与工程记录](half-r/README.md)
- [第三轮 P0/P1 报告](next/README.md) 与 [集群结果](next/cluster-next.md)

## 1. 版本演进

| 阶段 | 实现文件 | 核心变化 | 结果/状态 |
|---|---|---|---|
| V1 | `src/melspec.cpp` | 逐帧半长复数 FFT、稀疏 Mel、16 线程 | 正式对照；四组 PASS |
| 原始 Overlap | `melspec_overlap_prototype.cpp` | 滑动子 FFT、全 B 组 S 点重组、频域 Hann | 本地淘汰；重组成本大于 V1 |
| half-r | `melspec_overlap_half_prototype.cpp` | 实信号共轭，只做 B/2+1 组重组 | 同作业相对 V1 1.10–1.24× |
| half+delta | `melspec_overlap_half_delta_prototype.cpp` | 每帧只预计算一次输入 delta | 同作业相对 V1 1.11–1.25× |
| P0 | `next/melspec_overlap_pruned_power_prototype.cpp` | 只对 Mel 非零支持并集做 Hann/norm | 同作业相对 V1 1.19–1.46× |
| FFTW | `next/melspec_overlap_fftw_prototype.cpp` | batched C2C 替换 128 点 radix-2 | 同作业相对 V1 1.31–1.53× |
| FFTW+P0 | `next/melspec_overlap_fftw_pruned_prototype.cpp` | FFTW batched + Mel 功率裁剪 | **最终推荐；1.42–1.91×** |
| SIMD codelet | `next/fft128_codelet.h`、`next/melspec_overlap_codelet_prototype.cpp` | SoA radix-4、AVX2/FMA | 微基准接近 FFTW，整程序落后；不采用 |

代码提交：

- `23fb9d0`：理论推导与数学原型。
- `bcddea6`：原 Overlap 工程原型及本地负收益。
- `4c38d27`：实信号对称半数组重组。
- `4f9f87c`：half-r 集群 Benchmark/Profile。
- `ac6ba83`：P0 Mel 支持裁剪与微基准。
- `61b372c`：FFTW batched C2C。
- `7b1d911`：radix-4/AVX2 codelet 与失败调试记录。
- `f157a67`：FFTW+P0 组合候选和本地 A/B。
- `5bd0c38`：第三轮集群结果与推荐实现。

## 2. 数学主线

记 `N=n_fft`、`S=hop_length=128`、`B=N/S`、正频谱数 `N/2+1`。
把帧拆成 S 条交错实子序列：

```
z_q[j] = x[start+q+S*j],  j=0..B-1
Z_q[r] = sum_j z_q[j] exp(-2πi*r*j/B)
A[r+mB] = FFT_S(Z_q[r] * exp(-2πi*r*q/N))[m]
```

帧前进 S 个采样时，每条子序列只前移一个元素：

```
Z_{t+1,q}[r] = exp(+2πi*r/B) (Z_{t,q}[r] - x_old + x_new)
```

实输入只保存 `r=0..B/2`。为避免长期漂移，每 1024 帧从原始 PCM 精确重建子 FFT。
周期 Hann 在频域严格为：

```
X[k] = 0.5*A[k] - 0.25*A[k-1] - 0.25*A[k+1]
```

half-r 进一步使用完整帧频谱的共轭对称：

```
A[(B-r)+mB] = conj(A[r+(S-m-1)B]),  1<=r<B/2
```

因此只执行 `r=0..B/2` 共 B/2+1 组 128 点 FFT，但必须保留每组低频侧的
`m=S/2..S-1` 输出用于恢复高频残差。`r=0` 覆盖 DC 与 Nyquist；
`r=B/2` 是自配对残差，只保留 `m=0..S/2-1`。

## 3. 正确性

所有进入正式对照的实现均通过了：

- 本地 8 项 smoke：官方四组、CLI 默认、小 FFT 边界。
- 随机 PCM、低幅度 PCM、1024 帧刷新边界与 V1 输出对照。
- 官方四组全量 expected，容差 `rtol=1e-3, atol=1e-4`。
- 数学原型 44 组：随机、低幅度、零、DC、Nyquist、冲激、不同输入长度和 Hann 边界。

关键误差：

- half-r 数学原型最大 scaled 误差 `6.69e-16`，频域 Hann 后 `4.89e-16`。
- half-r 全量四组最大绝对误差与 V1 一致：`3.12e-2 / 1.17e-2 / 1.25e-1 / 6.25e-2`。
- 完整约 19.2 万帧递推：1024 帧刷新后末帧 scaled 误差约 `8.1e-14 / 1.5e-13`。
- FFTW、P0、FFTW+P0、codelet 的官方四组均 PASS。

## 4. 集群结果与可比性

### 同作业 77（hepnode2）

| 用例 | V1 秒 | half 秒 | half+delta 秒 | V1/half | V1/half+delta |
|---|---:|---:|---:|---:|---:|
| default | 1.543 | 1.403 | 1.387 | 1.100× | 1.113× |
| small_mels | 1.207 | 1.060 | 1.053 | 1.139× | 1.146× |
| large_fft | 2.639 | 2.141 | 2.120 | 1.233× | 1.245× |
| narrow_band | 2.597 | 2.090 | 2.088 | 1.243× | 1.244× |

### 同作业 199（hepnode2）

| 用例 | V1 | half+delta | P0 | FFTW | FFTW+P0 | V1/最终 |
|---|---:|---:|---:|---:|---:|---:|
| default | 1.541 | 1.380 | 1.296 | 1.177 | **1.089** | **1.415×** |
| small_mels | 1.199 | 1.052 | 0.964 | 0.843 | **0.766** | **1.565×** |
| large_fft | 2.644 | 2.134 | 1.959 | 1.729 | **1.537** | **1.720×** |
| narrow_band | 2.591 | 2.080 | 1.779 | 1.690 | **1.357** | **1.909×** |

最终版本相对同作业 half+delta 为 1.267×/1.373×/1.388×/1.533×。
作业 199 中五版本均 4/4 PASS；三轮交错 wall-time 中位比也支持最终版本四组最快。

**可比性声明：**

- 作业 77、199 各自内部可以算同节点加速比，但两个作业间绝对秒数不完全可比。
- 作业 45 的 V1、作业 47 的 V1 Profile 与后续候选不是同一作业，不能与候选直接相除。
- 本地 A/B 只用于筛选，不与集群秒数混合。
- 没有获得同作业的直接 FFTW R2C 或 SIMD Batch8 整程序数据。
  因此本项目**不宣称 Overlap 已超过**这些路线，只确认它在 V1 和冻结 Overlap 前取得了收益。

## 5. Profile

- 作业 47：V1 半长 FFT 占线程内约 76–81%，谱恢复约 13–15%，Mel 约 1.6–5%。
- 作业 85：half/half-delta 中 128 点重组 FFT 约占 43–44%，Hann+功率约 24%，
  相位乘法约 12%，子谱递推约 8–9%。
- 作业 200：P0 将 Hann+norm 时间从 2.917 降到 1.615 秒（default，-45%），
  narrow_band 从 5.824 降到 1.136 秒（-80%）。
- 作业 199 的 128 点微基准：原 radix-2 约 819/820 ns/FFT，
  FFTW batched 约 449/455 ns/FFT。
- 所有 Profile 使用 chrono 插桩或微基准；集群 `perf_event_paranoid=4`，
  没有可用的硬件计数/火焰图，不能据此声称 cache miss 或 IPC 改善。

## 6. 失败与未采用原因

1. **原全 r Overlap**：V1 蝶形数修正后，原重组蝶形反而更多，且每帧全量相位乘法；
   本地稳定慢于 V1，未进入集群端到端测试。
2. **直接全谱 Sliding DFT**：每帧约 52 万/105 万次复数递推，远高于半长 FFT。
3. **Chirp-Z/Bluestein**：理论上可降低块差更新，但需每帧两个完整复数 FFT、
   chirp 乘法及全谱递推，未显示工程优势，未实现。
4. **delta 独立优化**：只占全流程 8–9%，集群额外收益约 0–1%，方向不完全稳定，
   因此不是最终收益的主要来源，但与 P0/FFTW 组合不冲突。
5. **自研 AVX2 codelet**：修正了混合进制逆置换和非连续旋转因子读取，
   直接 DFT 误差约 `7.8e-13`，微基准约 225 ns/FFT；
   但整程序中的 SoA 相位、置换恢复和频谱写回成本使其落后 FFTW，不推荐。
6. **AVX-512 codelet**：未实现专用内核；本地不能执行 AVX-512，
   也没有把“集群可运行 AVX2 回退”冒充 AVX-512 收益。
7. **P0 的 default 阶段收益**：Hann 理论减少约 54%，实际阶段约 45%，
   受索引循环与分支影响；narrow_band 约 80%，比较符合预期。

## 7. 最终推荐与依赖

推荐文件：`next/melspec_overlap_fftw_pruned_prototype.cpp`
SHA-256：`761109fca0a3662be76aaf3341adfffd907b54f937d2489ced19150028f4e761`。

优点：同作业四组最快、全部 PASS；相对 V1 1.415×/1.565×/1.720×/1.909×。
代价：依赖 FFTW 3.3.10；每线程批量 Plan 与 `fftw_malloc` 数组；
一次性 ESTIMATE 初始化；二进制不可脱离 `libfftw3` 独立运行。

不依赖外部库的备选是 P0-only。若只求简洁，half+delta 也保留了主要重叠收益。
是否需要继续追 FFTW R2C/SIMD Batch8，应在同节点加入正式候选后进行，
而不是用不同作业或本地秒数比较。

## 8. 最终状态

- 主动优化与集群 Benchmark 已停止。
- 默认 `src/melspec.cpp` 保持 V1，未修改 `src/main.cpp`、`validate.py` 或数据。
- 所有候选源码、正确性日志、失败记录、微基准、Profile 和集群原始输出均保留。
- 不修改其他 Worktree，不合并 master。
