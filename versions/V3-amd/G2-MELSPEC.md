# G2 — HIP Melspec 流水线

> 本文保留 G2 首次可用版本的记录。G3 已把同一份代码优化到
> **0.689 / 0.458 / 0.754 / 0.739 s**（官方作业 395），
> 结论与最终参数见 [G3-OPTIMIZATION.md](G3-OPTIMIZATION.md) 和 [FINAL_AMD.md](FINAL_AMD.md)。

## 1. 实现

`src/melspec_hip.cu` 已从占位 scaffold 替换为与冻结 CUDA 设计对应的完整 HIP 后端：

- 完整 PCM 一次 H2D，GPU kernel 直接 gather hop=128 的重叠帧并施加周期 Hann。
- hipFFT/rocFFT batched R2C（FP32）/D2Z（FP64），自然 `[batch][n_fft]` → `[batch][n_fft/2+1]`。
- fused power+Mel：每个 Mel tap 即时计算 `|X|²`，不落功率矩阵。
- 一个 wavefront 计算一个 (frame, Mel row) 的点积；first/last 稀疏区间；row-major `[n_mels,n_frames]`。
- full batch 和精确 tail batch 分别 plan；n_fft=1 按题意返回零。
- 异步 pinned 输出（2 stream、2 event）与 CPU float→double 转换流水、resize 线程、初始化重叠均可选。
- `src/melspec.cpp` 通过 `MELSPEC_HIP` 分派；`src/main.cpp`、`validate.py`、数据与公共头文件未改。

默认构建参数：FP32、batch512、fused、初始化重叠、async pinned 输出、16 行 staging、resize 线程。
`HIP=0 CUDA=0` 保持 V1 CPU 对照；`HIP=1` 构建 HIP 可执行文件。

## 2. 正确性

本机/集群：
- 8 项 smoke：官方四组小样本、CLI 默认、n_fft=1/2/8，全部 PASS。
- 全量官方四组 `validate.py --runs 3`：**4/4 PASS**。
- 最大绝对误差：0.0546875 / 0.01953125 / 0.15625 / 0.15625，和 CUDA 路线同一量级；
  逐元素仍满足 `rtol=1e-3, atol=1e-4`。

## 3. 首次全量结果（作业 346，hepnode0）

ROCm JIT kernel cache 在同一私有路径预热；正式进程使用 pageable host 输入、batch512。

| 用例 | HIP 秒 | vs 同次 librosa | max abs err |
|---|---:|---:|---:|
| default | 0.904 | 14.811× | 5.47e-2 |
| small_mels | 0.663 | 19.922× | 1.95e-2 |
| large_fft | 0.988 | 25.875× | 1.56e-1 |
| narrow_band | 0.991 | 25.825× | 1.56e-1 |

这不是最终性能：随后 profile 发现 pageable H2D 约 **194 ms**，需 G3 修复。
构建二进制 SHA-256：`9cd11c47862d4f64cac1dc3d1e2f9f32897482f3bb72672f24ac707332a798f1`。

原始输出：[`raw/g2/official-346.txt`](raw/g2/official-346.txt)、[`raw/g2/amd-346.out`](raw/g2/amd-346.out)、
[`raw/g2/smoke-345.txt`](raw/g2/smoke-345.txt)。

## 4. 当前限制

- `ROCFFT_RTC_CACHE_PATH` 预热是性能前提；cold plan 约 0.9–1.8 s，见 G1。
- 首版 batch512 未区分 n_fft；G1 微基准显示 16384 点更优 batch 是 256。
- H2D pageable 仍需 pinned 输入验证。
- amd-cluster 作业通过 hepnode0 普通 Slurm 槽执行，不申请 GPU GRES。
