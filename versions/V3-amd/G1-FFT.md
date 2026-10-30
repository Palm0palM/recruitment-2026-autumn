# G1 — hipFFT/rocFFT 可行性

## 1. 门禁结论

**G1 PASS，但附加强制条件：必须部署匹配 gfx1201 的 rocFFT kernel JIT 缓存。**
没有缓存时，新进程第一次 plan 初始化约 **0.9–1.8 s**，足以单独否掉端到端性能；
使用项目私有 `ROCFFT_RTC_CACHE_PATH` 后，新进程第一次 plan 约 **5–7 ms**。
这不是可选的微优化，而是本路线能否参赛的前提。

## 2. 设备与软件

- hepnode0，AMD Radeon AI PRO R9700，`gfx1201`，32 CU，wavefront 32，31.86 GiB。
- HIP runtime/driver `7.1.52801-9999`，clang 21.1.8，hipFFT/rocFFT 7.1.1。
- 独立 8 点直接 DFT 对照：`r2c_self_test=PASS`。
- 所有 GPU 运行在 Slurm 作业内，没有使用 GRES（hepnode0 没有 GPU GRES）。

## 3. batched FP32 R2C 微基准（作业 339）

自然布局 `[batch][n_fft]` → `[batch][n_fft/2+1]`，同一 plan 内 batch 个变换。
时间为 warm CUDA-style HIP event 执行区间最小值，不含 H2D/D2H。

| n_fft | batch | plan ms | first exec ms | exec min ms | ns/FFT |
|---:|---:|---:|---:|---:|---:|
| 8192 | 1 | 1836.5 | 0.063 | 0.017 | 17240 |
| 8192 | 16 | 5.7 | 0.112 | 0.041 | 2540 |
| 8192 | 64 | 4.7 | 0.091 | 0.026 | 399 |
| 8192 | 128 | 5.3 | 0.068 | 0.027 | 214 |
| 8192 | 256 | 5.6 | 0.088 | 0.039 | 152 |
| 8192 | 512 | 5.3 | 0.133 | 0.052 | 101 |
| 8192 | 1024 | 4.4 | 0.169 | 0.164 | 160 |
| 16384 | 16 | 7.3 | 0.126 | 0.057 | 3588 |
| 16384 | 64 | 1246.2 | 0.203 | 0.036 | 570 |
| 16384 | 128 | 5.5 | 0.097 | 0.040 | 311 |
| 16384 | 256 | 5.1 | 0.104 | 0.066 | 259 |
| 16384 | 512 | 5.5 | 0.189 | 0.164 | 321 |
| 16384 | 1024 | 5.7 | 0.459 | 0.462 | 452 |

FP64 D2Z batch256：8192 为 0.160 ms（623 ns/FFT），16384 为 0.386 ms（1506 ns/FFT），
明显慢于 FP32，继续以 FP32 为主线。

## 4. 标准任务初始化的真实问题

rocm 7.1 的 hipFFT 实际由 rocFFT 执行，并使用 hipRTC 编译 kernel。
冷进程的第一次 plan 经 hipRTC 编译，实测 1.8 s 左右；第二次 plan 在同一进程约 5 ms。
这意味着每个 Melspec CLI 进程都必须依赖持久 kernel 缓存，否则无法与 CPU 竞争。

已验证方案：

```bash
mkdir -p tmp/rocfft-cache
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/rocfft_kernel_cache.db"
# 第一次运行生成约 396 KiB 的 SQLite kernel cache
# 后续新进程首次 plan: 6.5 ms（对比 cold 1824.7 ms）
```

缓存必须绑定 ROCm 版本、gfx1201、hipRTC/LLVM 生成器指纹；换环境需要重新生成。
它不依赖 PCM 内容。缓存文件不提交 Git，由构建/预热阶段生成。
正式性能必须把这条缓存视为环境的一部分，并在同节点作业内先预热。

## 5. batch 选择

- 8192：batch512 的 ns/FFT 最低；batch1024 明显退化。
- 16384：batch256 最低；batch512/1024 退化。
- 因此不能照搬 CUDA 的全局 batch512；G2/G3 将按 n_fft 选择 512（8192）/256（16384），
  再用完整 Melspec 端到端 A/B 验证。当前先以统一构建参数进入 G2。

原始输出：[`raw/g1/hipfft-probe-339.txt`](raw/g1/hipfft-probe-339.txt)、
冷/热缓存对照：[`raw/g1/hipfft-cache-cold.txt`](raw/g1/hipfft-cache-cold.txt)、
[`raw/g1/hipfft-cache-warm.txt`](raw/g1/hipfft-cache-warm.txt)。
