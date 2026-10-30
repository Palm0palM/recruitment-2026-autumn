# P0 技术说明：Mel 非零支持频点裁剪

## 1. 目的与适用范围

本优化只减少“频域 Hann + 模平方”的执行范围，不改变 FFT、滑动递推或 Mel 滤波器。
适用于任何已经得到完整未加窗正频谱 `A[k]` 的 Mel 实现，
包括当前 Overlap、FFTW batched 和 SIMD batch 候选。

独立源码：

- `next/melspec_overlap_pruned_power_prototype.cpp`
- `next/melspec_overlap_fftw_pruned_prototype.cpp`

基线为 half+delta，SHA-256
`cc1b08d7203b68e482645ad8bfeb6cbcc1cb0da4e697aad63734b0a6b4d26efe`。
默认 `src/melspec.cpp` 未使用该改动。

## 2. 数学等价性

对单个 Mel 三角 m，代码已计算其精确非零区间 `[first[m], last[m])`。
最终输出为：

```
M[m,t] = sum_k H[m,k] * |X_t[k]|^2
```

其中 `H[m,k]=0` 对所有 `k` 不属于该区间。设所有三角支持区间的并集：

```
P = union_m [first[m], last[m])
```

则对任意 `k not in P`，所有 m 的 `H[m,k]` 都是 0。因此：

- `|X_t[k]|^2` 对最终 Mel 输出的贡献严格为 0；
- 可以完全跳过这些 k 的 Hann 组合和 `std::norm`；
- 仍必须保留完整 `A[k]`，因为某支持 bin 的 Hann 需要 `A[k-1]` 和 `A[k+1]`，
  而相邻 bin 自身不一定属于 P。

这是数学等价裁剪，不是近似、阈值化或对音频内容特化。
`P` 只由公开参数 `sr/n_fft/n_mels/f_min/f_max` 和与 librosa 一致的非零三角边界决定。

## 3. 代码修改

在 `compute_melspectrogram()` 中，原有 `first/last` 计算之后：

```cpp
std::vector<char> used(n_bins, 0);
for (int m = 0; m < n_mels; ++m) {
    while (first[m] < n_bins && mel_fb[m][first[m]] == 0.0) ++first[m];
    while (last[m] > first[m] && mel_fb[m][last[m] - 1] == 0.0) --last[m];
    for (int k = first[m]; k < last[m]; ++k) used[k] = 1;
}
std::vector<int> power_bins;
for (int k = 0; k < n_bins; ++k)
    if (used[k]) power_bins.push_back(k);
```

`overlap_power()` 多接收 `power_bins`，Hann/norm 只遍历该列表：

```cpp
for (int k : power_bins) {
    if (k == 0)          // A[-1] = conj(A[1])
        power[k] = ...;
    else if (k == n/2)   // A[N/2+1] = conj(A[N/2-1])
        power[k] = ...;
    else
        power[k] = std::norm(0.5*A[k] - 0.25*(A[k-1] + A[k+1]));
}
```

`power` 容器仍按完整 `n_bins` 分配并初始化为 0。Mel 投影代码没有改变，
只会读取自身的非零支持区间，因此不会读取未计算值。

### 边界检查

- **DC**：若某个 Mel 支持包含 k=0，使用共轭 `A[-1]=conj(A[1])`；
  当前正式参数 f_min=0 时第一个三角的实际非零从 k=1 开始，但代码仍正确处理。
- **Nyquist**：若支持包含 N/2，使用 `A[N/2+1]=conj(A[N/2-1])`。
- **窄带**：f_min/f_max 改变支持为 52..1365，仍按真实并集裁剪。
- **空 Mel 通道**：不产生支持，视为贡献恒为 0；不扩大任何区间。
- 不依赖 `A` 的频率布局；residue-major 或自然顺序均在 Hann 之前恢复为可索引频谱。

## 4. 覆盖数据

| 用例 | 使用 / 全部正频点 | 比例 | 区间 |
|---|---:|---:|---|
| default | 1881 / 4097 | 45.91% | 1..1881 |
| small_mels | 1881 / 4097 | 45.91% | 1..1881 |
| large_fft | 3763 / 8193 | 45.93% | 1..3763 |
| narrow_band | 1314 / 8193 | 16.04% | 52..1365 |

## 5. 可移植性

### 到 FFTW R2C

FFTW R2C 已输出 `k=0..N/2` 的复频谱。将同一 `power_bins` 作为后处理循环即可；
必须保留完整 R2C 输出（或至少保留每个使用 bin 的左右邻居）。
若 R2C 候选本身已对功率谱全部频点计算，则 P0 可直接减少 norm 和 Mel 前的循环。

### 到 SIMD / Batch8

P0 的数学依赖与 FFT 内核无关，但 SIMD 化时不要只按连续 lane 截断：

1. 先用同一 `first/last` 计算支持并集；
2. 对 power_bins 做 gather/分支或按连续区间分块；
3. 若要批量 norm，必须保留 `A[k-1..k+1]` 的完整输入；
4. 未使用 lane 可写入临时零或丢弃，但 Mel 的读取范围必须与 `first/last` 一致；
5. 不要用浮点阈值替代 `==0`，否则会改变与官方滤波器的等价性。

最稳妥的移植方式：保留 `power` 完整数组和 `power_bins` 列表，
只跳过未使用 bin 的计算；SIMD 主要收益来自 FFT 内核，P0 是独立的尾部裁剪。

## 6. 实测结果与限制

- 作业 199（hepnode2）：P0 组合进 FFTW 后四组 PASS，
  FFTW+P0 相对 V1 为 1.415×/1.565×/1.720×/1.909×。
- 作业 200 P0 阶段 Profile：Hann/norm 时间 default 下降约 45%，
  narrow_band 下降约 80%。
- 当前实现使用 `std::vector<int>` 和每 bin 分支；default 的理论 bin 数减少 54%，
  但索引和分支使阶段实际下降约 45%。若后续移植到 SIMD，
  可改为连续区间循环以改善编译向量化，但这需要单独 A/B，不从 bin 比例推断收益。

## 7. 复现与验证

```bash
source tmp/session-env.sh
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  src/main.cpp versions/V2-overlap/next/melspec_overlap_fftw_pruned_prototype.cpp \
  -lfftw3 -o tmp/binaries/melspec-overlap-fftw-pruned
$PY scripts/smoke.py tmp/binaries/melspec-overlap-fftw-pruned
$PY scripts/bench.py tmp/binaries/melspec-overlap-fftw-pruned --case all --runs 1
```

原始证据：`next/raw/fftw-pruned-smoke.txt`、`next/raw/fftw-pruned-full-correctness.txt`、
`next/raw/cluster/validate-fftw-pruned-199.txt`、`next/raw/cluster/stage-*-200.json`。
