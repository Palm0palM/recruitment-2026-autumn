# V3-AMD — HIP/ROCm Melspec 最终交付

AMD Radeon AI PRO R9700（`gfx1201`）上的完整 HIP/hipFFT Melspec 后端，
与冻结 CUDA 版本同构：整段 PCM 一次 H2D、GPU gather+Hann、batched R2C、
fused power+Mel、稀疏首尾区间、wavefront 归约、异步/重叠输出。

## 1. 最终成绩（hepnode0，作业 395）

官方 `validate.py --runs 3`，四组全部 PASS（进程 wall time）：

| 用例 | HIP（秒） | 同次 librosa 加速 | max abs err |
|---|---:|---:|---:|
| default | **0.689** | 20.371× | 5.47e-2 |
| small_mels | **0.458** | 30.253× | 1.95e-2 |
| large_fft | **0.754** | 35.741× | 1.56e-1 |
| narrow_band | **0.739** | 36.689× | 1.56e-1 |

- 最终二进制 SHA-256：`9d84f8617568e105f1187125e065512cf9b4c50e4d9d695ae74d3ce24a9e4970`
- 误差与 G2/CUDA 同量级，逐元素满足 `rtol=1e-3, atol=1e-4`。
- 额外 8 项 smoke（官方四组小样本、CLI 默认、n_fft=1/2/8）全部 PASS。

## 2. 同节点、同作业对照（严格可比）

作业 395 内 HIP 与 **SIMD FP32 B16**（冻结源码 `5496622`，`-DMELSPEC_FP32_BATCH=16`）
交错运行 6 轮，每轮都对 expected 校验。中位数（秒）：

| 用例 | HIP | SIMD B16 | HIP 相对 |
|---|---:|---:|---:|
| default | **0.692** | 0.859 | 1.24× |
| small_mels | **0.461** | 0.497 | 1.08× |
| large_fft | **0.745** | 1.081 | 1.45× |
| narrow_band | **0.740** | 1.023 | 1.38× |

最小值口径：HIP 0.679 / 0.454 / 0.730 / 0.722，
B16 0.840 / 0.489 / 1.056 / 1.018。四组几何平均约 **1.28×**。

## 3. 与 NVIDIA CUDA 历史成绩的趋势比较（不同硬件，不冒充严格对照）

| 用例 | HIP R9700（本次） | CUDA L40（历史作业 276） |
|---|---:|---:|
| default | 0.689 | 0.782 |
| small_mels | 0.458 | 0.536 |
| large_fft | 0.754 | 0.800 |
| narrow_band | 0.739 | 0.800 |

两者节点、GPU、驱动完全不同（hepnode0 gfx1201 vs hepnode2 L40 sm_89），
只能说明 AMD 路线已达到同一量级并略快，不能当作同硬件胜负结论。

## 4. Profile（作业 395，default，毫秒）

`y_convert 44.9 · h2d 17.7 · hipFFT plan 26.2 · gather 25.7 · FFT 30.3 ·
Mel 35.1 · D2H 23.8 · window/filterbank 9.0 · prewarm_wait 6.5`
→ `compute_total 220.9 ms`。

进程 wall time 剩余约 470 ms 属于受保护 `main.cpp` 的读 49 MB PCM、
int16→double 归一化、float 输出写盘与进程启动，本后端不可修改。

## 5. 复现方式

```bash
# hepnode0：普通独占 Slurm 作业即可，hepnode0 没有 GPU GRES，不要用 --gres=gpu/--mem
cd ~/melspec-v3-amd
source tmp/amd-gpu/env.sh
export HIP_ARCH=gfx1201
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"   # 必须
make -B -j4 HIP=1 EXEC=tmp/bin/melspec-hip
~/venv/bin/python validate.py tmp/bin/melspec-hip --runs 3
```

或直接提交现成作业：

```bash
sbatch -w hepnode0 --time=00:30:00 --export=ALL,RUNS=6 scripts/amd-final.slurm
```

`scripts/amd-final.slurm` 会构建 HIP 与冻结 B16、交错跑 6 轮、跑官方 4 组、
再跑 profile，并打印源码/二进制哈希。

## 6. 必要条件与已知限制

- **rocFFT kernel JIT 缓存是性能前提**：冷进程首次 plan 0.9–1.8 s，
  配置 `ROCFFT_RTC_CACHE_PATH` 后 5–7 ms。缓存与 ROCm 版本、gfx1201、LLVM 指纹绑定，
  换环境需重新生成；缓存文件不进 Git。
- 私有 rootless ROCm 7.1.1，不修改系统包/容器/共享环境；所有计算都在 Slurm 作业内。
- `hepnode0` 无 GPU GRES，只能用普通节点分配。
- 首次 HIP 队列初始化 ~190–235 ms 已通过进程启动预热与 PCM 读取重叠；
  若换成启动更快的 harness，这部分收益会缩小。
- 大页提示依赖节点 THP 策略为 `[madvise]`；不满足时自动退化为普通 4 KiB 页，
  正确性不变、性能回退到 G3 中间态。
- 输出仍是 `std::vector<double>` 全量初始化（题目接口要求），未跳过任何写入。

## 7. 相关文档

- 环境与私有 ROCm：[ENVIRONMENT.md](ENVIRONMENT.md)
- FFT 可行性与缓存：[G1-FFT.md](G1-FFT.md)
- 完整 HIP 流水线：[G2-MELSPEC.md](G2-MELSPEC.md)
- 优化过程与被否决实验：[G3-OPTIMIZATION.md](G3-OPTIMIZATION.md)
- 原始日志：[`raw/g1`](raw/g1)、[`raw/g2`](raw/g2)、[`raw/g3`](raw/g3)
