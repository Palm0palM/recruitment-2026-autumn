# Overlap 路线答辩精简材料

## 1. 一句话结论

利用高重叠帧的可复用子 FFT、实信号对称和频域 Hann，最终 FFTW+P0 在 Xeon 6338
同节点官方四组上相对 V1 达到 **1.415× / 1.565× / 1.720× / 1.909×**，四组全部 PASS；
但没有同作业直接 FFTW R2C/SIMD Batch8 数据，不宣称超过这些路线。

## 2. 算法流程（可直接画流程图）

1. 读取帧后拆成 S=hop 条交错实序列，每序列长度 B=N/S。
2. 缓存每条实序列的 B 点 FFT；下一帧只做一次一元素 Sliding DFT 更新。
3. 实输入共轭：只保存 r=0..B/2 子谱。
4. 对每个残差 r 做 S 点复 FFT；利用完整帧对称
   `A[(B-r)+mB]=conj(A[r+(S-m-1)B])` 恢复另一半正频率。
5. 周期 Hann 在频域用 `0.5A[k]-0.25A[k-1]-0.25A[k+1]` 精确施加。
6. 只对 Mel 非零支持并集做模平方和 Mel 投影（P0）。
7. 每 1024 帧从 PCM 精确重建子谱，限制递推误差。
8. 16 线程按连续帧区间分区，每线程独立子谱/FFTW Plan。

关键公式：

```
Z_q[r] = FFT_B(x[start+q+S*j])
A[r+mB] = FFT_S(Z_q[r] exp(-2πi r q/N))[m]
Z_{t+1,q}[r] = exp(+2πi r/B)(Z_{t,q}[r] - x_old + x_new)
```

## 3. 版本性能表（作业 199，同节点，秒）

| 版本 | default | small_mels | large_fft | narrow_band | 相对 V1 |
|---|---:|---:|---:|---:|---:|
| V1 | 1.541 | 1.199 | 2.644 | 2.591 | 1.00× |
| half+delta | 1.380 | 1.052 | 2.134 | 2.080 | 1.11–1.25× |
| P0 | 1.296 | 0.964 | 1.959 | 1.779 | 1.19–1.46× |
| FFTW | 1.177 | 0.843 | 1.729 | 1.690 | 1.31–1.53× |
| **FFTW+P0** | **1.089** | **0.766** | **1.537** | **1.357** | **1.42–1.91×** |

历史同作业 77（不是 199）：half 相对 V1 约 1.10–1.24×，half+delta 约 1.11–1.25×。
不同作业的绝对秒数不要交叉相除；作业 45/47 的 V1 也不能与后续候选混算。

## 4. 主要优化决策

| 决策 | 原因 | 结果 |
|---|---|---|
| 滑动子 FFT 而非直接 Sliding DFT | 直接全谱每帧约 52/105 万次递推，明显高于 FFT | 建立可复用结构 |
| half-r 共轭恢复 | 只做 B/2+1 组而不是 B 组，但保留 m>S/2 输出 | 相对 V1 1.10–1.25× |
| 频域 Hann | 精确等价，避免每样本再次乘窗 | 降低时域窗成本 |
| P0 支持裁剪 | Mel 三角只读少量 bin，但保留 A[k±1] | Hann 阶段 -45%/-80% |
| FFTW batched C2C | 128 点微基准 819→449 ns，一次执行全部 residue | 四组进一步加速 |
| 1024 帧刷新 | 递推长期误差不消失 | 末帧 scaled 误差约 1e-13 |
| 自研 AVX2 codelet | 微基准接近 FFTW，研究价值高 | **整程序落后，不推荐** |

## 5. 失败与边界（答辩应主动说）

- 原全 r Overlap：V1 蝶形数修正后，其重组成本反而高于 V1，本地淘汰。
- 直接 Sliding DFT/Chirp-Z：运算或常数、内存、漂移不具备明确优势。
- codelet：微基准 225 ns/FFT 不等于整程序收益，SoA 转换、逆置换和恢复写回抵消。
- P0 的 default 实际阶段收益约 45%，低于 bin 比例 54%，不能只凭减少比例推断。
- 最终依赖 FFTW 3.3.10；无外部库时 P0-only/half+delta 是备选。
- 集群 `perf_event_paranoid=4`，没有硬件 PMU；不报告 cache miss、IPC 或 AVX-512 收益。
- 没有同作业 R2C/Batch8 候选，因此最终话术是“缩小差距”而非“已经超越”。

## 6. 证据索引

- 完整归档：`versions/V2-overlap/FINAL.md`
- P0 单页说明：`versions/V2-overlap/P0-mel-pruning.md`
- half 数学/工程：`versions/V2-overlap/half-r/README.md`
- 集群结果：`versions/V2-overlap/next/cluster-next.md`
- 原始日志：`versions/V2-overlap/next/raw/cluster/`
- 最终推荐源码：`versions/V2-overlap/next/melspec_overlap_fftw_pruned_prototype.cpp`

## 7. 建议答辩页顺序

1. 问题与 V1 瓶颈（FFT 约 78%，memory profile）。
2. 重叠帧为何不能直接复用（频谱全局性），引出滑动子 FFT。
3. 实数共轭恢复与完整 m 输出（一张频谱示意图）。
4. 频域 Hann 三频点公式与 DC/Nyquist 边界。
5. P0 支持裁剪与阶段 Profile。
6. FFTW batched + 微基准、版本性能表。
7. 失败路线、可比性边界和最终结论。
