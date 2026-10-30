# P1：跨批次复用原始音频 slab

选项 `-DMELSPEC_PACK_REUSE=1`，默认关闭，仅 B16 路径。

## 等价性与复杂度（实现前约束）

设帧 hop=H，slab 从绝对采样位置 A=t0*H 开始。
对 n=qH+r，定义 R[r,q]=y[A+r+qH]。相对帧 f 的第 n 个样本为
`y[A+fH+n] = R[r,q+f]`。因此每次连续读取 R 行中的 16 个 double，
正好是原 pack 的 16 个帧 lane，无音频内容假设、无近似窗口递推。
仍按 `float(double_sample * double_window[n])` 计算，FFT 位反转位置及浮点模式不变。

slab 在多个 B16 间共享；最多 K=128 帧。按约 16k 个附加 double 的预算自适应 K，
最少16、取16的倍数。行 stride=`ceil(N/H)+K`，所需最大行索引不超过
`floor((N-1)/H)+(K-16)+15 < stride`；最后 slab 的越界原始采样显式补0，
但只执行合法完整批次，填充值不参与输出。其余 0..15 帧仍走冻结 FP64 尾部。
仅 N>=32 且 H<=N/4 启用；非高重叠、小 FFT、其他 ISA/精度路径保持原样。

- 原 pack：FN 次 double Hann 乘法；8FN 字节逻辑音频加载，标量转换/拼装。
- 新 pack：仍 FN 次 double 乘法与8FN缓存加载，改用连续 double SIMD + narrow。
- raw 布局预处理：约 O(S + (F/K)N)，不是每个 frame 的 O(FN) 转置。
- 每 worker slab 大小 `8H(ceil(N/H)+K)`；原 buffer 和 std::vector 都合法初始化。
- 不分配全量 STFT；不改变 FFT、恢复、Mel 和 output 初始化。

潜在收益依据是 pack 约10–17%端到端占比，以及原标量构造指令数；不是声称必须乘法消失。
可能退化：准备成本、额外偏移加载、slab/FFT 同占 L2（大 FFT 原工作集1.5MiB，
目标 L2 实际1.25MiB，不能沿用旧报告“远低于 L2”的笼统描述）、缓存 set/TLB 行为。

本地仅验证未启用 B16 的回退一致性：67组独立合成输入 PASS；B16 已交叉编译，
真正 B16 正确性/性能以集群日志为准。
