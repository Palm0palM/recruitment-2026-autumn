# P1-B：固定 128 点 radix-4 / AVX2 Codelet

独立实现：`fft128_codelet.h`、`melspec_overlap_codelet_prototype.cpp`。
不包含 FFTW 或 P0，其他 hop 规模保留原 radix-2 回退。

## 内核

- 自然顺序 SoA 输入，三级 radix-4 DIF（128→32→8）和一级 radix-2。
- 旋转因子用 `W^(j)、W^(2j)、W^(3j)`，非连续索引必须 gather/构造向量。
- AVX2/FMA 处理 quarter>=4 的蝶形，剩余及小阶段走标量；无 AVX2 时有完整标量路径。
- 输出为混合进制顺序，使用逆置换 `perm[order[position]]=position` 融入恢复。
- 本文件尚未实现 AVX-512 专用路径；本地无 AVX-512，不能执行相关指令。
  集群可执行同一 AVX2 回退，不能把微基准推断成 AVX-512 收益。

## 调试证据（保留失败过程）

初版未通过直接 DFT，不能用性能数字。问题逐项定位为：

1. 混合进制映射需要按子问题逆序构造，且恢复需要 inverse permutation；
2. `W^(2j)`、`W^(3j)` 非连续，普通 `_mm256_loadu_pd` 会读错旋转因子；
3. 修正后标量与 AVX2 的直接 DFT `max_abs_err` 分别为
   `7.81e-13` 和 `7.84e-13`。

本地微基准（5000 次，重置不计时）：

| 路径 | ns/FFT（33 组） | ns/FFT（65 组） |
|---|---:|---:|
| 标量 radix-4 | 283.727 | 275.246 |
| AVX2/FMA radix-4 | **230.619** | **225.022** |
| 原 radix-2（另一微基准） | 410.166 | 409.363 |
| FFTW batched（另一微基准） | 218.240 | 209.475 |

## 端到端结论边界

内核微基准不含 SoA 相位填充、置换恢复、频谱写回、Hann 或 Mel。
整程序中 codelet 虽通过 8 项 smoke 和四组全量 expected，但没有超过 FFTW；
在 `ab_next.py` 的完整 A/B 中落后于 batched FFTW。
因此不能把约 225 ns/FFT 宣称为程序加速，现阶段推荐候选不是自研 codelet。
后续若继续，应先优化恢复与 Hann 数据流，而不是盲目增加 AVX-512 指令。
