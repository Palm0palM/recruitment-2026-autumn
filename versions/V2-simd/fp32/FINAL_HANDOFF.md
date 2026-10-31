# V2 SIMD / FP32 Batch16 最终移交

状态：**正式冻结候选**。本轮不再做算法优化。

## 1. 固定身份

| 项目 | 值 |
|---|---|
| 算法提交 | `e448c989c921762884582804efe23d6607678ff9` |
| 最终构建配置提交 | `4df5dc8`（Makefile 自动 ISA 选择） |
| 候选标签 | `v2-simd-fp32-b16` |
| 最终冻结标签 | `v2-simd-fp32-b16-final` |
| 源码 | `src/melspec.cpp` SHA-256 `95aa8e56fb5ca91d89de153bc699a2a2361bb48b174b5076478d258dcf282f3f` |
| 普通 make 产物 | `melspec`，SHA-256 `5514ccb5def7329a5356f2ae85c773c4d06bd6a722945f984b20049581525182` |
| 显式 B16 产物 | `tmp/bin/fp32-b16-explicit`，与普通 make 字节一致 |
| Makefile | `618615b185ced4396ed384dfac183d3f051322ae75ade153bfd455f142215f84` |
| 冻结构建作业 | hepnode3 job 322 |
| 受保护文件 | `src/main.cpp`、`validate.py`、`data`、`include/melspec.h` 未改 |

构建环境：Xeon Gold 6338 @2.00GHz，16 物理核，GCC 15.2.0，`-O3 -march=native -pthread`。

## 2. 自动构建选择（重要）

普通 `make` 不再无条件生成 FP64：

```text
目标 CPU 有 AVX-512F       -> MELSPEC_FP32_BATCH=16（最终候选）
目标 CPU 有 AVX2+FMA       -> MELSPEC_FP32_BATCH=8
其他/无 SIMD               -> MELSPEC_FP32_BATCH=0（strict FP64）
```

显式覆盖：

```bash
make -B -j16                         # 自动选择，目标机为 FP32 Batch16
make -B -j16 MELSPEC_FP32_BATCH=16   # 强制 FP32 Batch16
make -B -j16 MELSPEC_FP32_BATCH=8    # 强制 FP32 Batch8
make -B -j16 MELSPEC_FP32_BATCH=0    # 强制 strict FP64
```

构建目录按精度隔离为 `tmp/build/fp16`、`fp8`、`fp0`，避免配置切换复用旧对象。

## 3. 最终成绩（Job 271）

hepnode3，16 物理核，`--hint=nomultithread`，四程序同作业轮转 6 次，完整进程 median，四组全部 4/4 PASS。

| 用例 | FP64 B8 | FP32 B8 | **FP32 B16** | FFTW float+PATIENT | B16 vs FP64 | B16 vs FFTW |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.865315 | 0.839211 | **0.834009** | 0.855914 | 1.037× | 1.026× |
| small_mels | 0.527069 | 0.495110 | **0.485544** | 0.516943 | 1.085× | 1.065× |
| large_fft | 1.138472 | 1.055000 | **1.035395** | 1.064758 | 1.100× | 1.028× |
| narrow_band | 1.117619 | 1.013177 | **0.999337** | 1.012433 | 1.118× | 1.013× |

冻结作业 322 用普通 `make` 重新构建并执行官方 `validate.py --runs 3`，4/4 PASS：

| 用例 | 官方最小 wall (s) | max_abs_err | 相对同作业 librosa |
|---|---:|---:|---:|
| default | 0.811 | 3.91e-2 | 19.626× |
| small_mels | 0.495 | 1.56e-2 | 30.239× |
| large_fft | 1.061 | 1.25e-1 | 32.731× |
| narrow_band | 1.021 | 9.38e-2 | 34.388× |

原始结果见 `raw/final-freeze-*`。

## 4. Profile 与缓存

集群诊断作业 269，每帧 ns：

| 阶段 | FP64 B8 default | FP32 B8 default | FP32 B16 default | FP64 B8 large | FP32 B8 large | FP32 B16 large |
|---|---:|---:|---:|---:|---:|---:|
| pack | 7,188 | 6,956 | 6,824 | 14,517 | 13,756 | 13,565 |
| FFT | 11,970 | 6,451 | **5,379** | 28,722 | 14,471 | **12,271** |
| recover | 3,103 | 1,916 | **1,401** | 6,301 | 3,796 | **2,781** |
| Mel | 1,359 | 1,008 | 1,000 | 2,743 | 1,799 | 1,885 |

每 worker 工作区（不含尾帧单帧缓冲）：

| 配置 | N=4096 | N=8192 |
|---|---:|---:|
| FP64 B8 re/im | 0.50 MiB | 1.00 MiB |
| FP32 B8 re/im/power | 0.375 MiB | 0.75 MiB |
| FP32 B16 re/im/power | 0.75 MiB | 1.50 MiB |

B16 的工作区与旧 FP64 B8 同级，仍低于 L2；收益来自 FP32 FFT 与更少带宽。

## 5. 正确性与精度

- 官方四组完整 expected，B16 4/4 PASS。
- `smoke.py`：官方四组、CLI 默认、n_fft=1/2/8、tail_2/tail_3 全 PASS。
- `fft_check.py`：1～16384 二次幂，random/zero/impulse/sine 全 PASS。
- 高动态：零值、冲激、满幅交替、1k 正弦、近 Nyquist、DC step、随机、两帧输入全部 PASS。
- 相对 FP64 的最大 tolerance 利用率约 **0.0132（1.32%）**，剩余约 70 倍容差余量。
- Hann 仍 double 计算；谱恢复系数/运算和 Mel 累加保持 double。
- 尾帧 0..15 走 FP64 单帧路径；n_fft=1 走原特殊单帧。
- 浮点输出由受保护 main 统一转 float32。

## 6. AVX-512 与硬件要求

- FP32 Batch16：**AVX-512F**；目标 Xeon Gold 6338 具备。
- FP32 Batch8：AVX2 + FMA。
- FP64 fallback：无 AVX-512 时使用 AVX2 Batch4；generic 使用单帧标量。
- FP32 B16 实现使用 AVX-512 float/double FMA；没有 AVX-512DQ 的强制依赖。目标机同时具备 F/DQ/BW/VL 等完整指令。
- 若编译目标缺乏对应 ISA 且强制 `MELSPEC_FP32_BATCH=16/8`，源码用 `#error` 明确拒绝，不会静默生成错误程序。

## 7. Hybrid CPU Backend 接口

未来 Hybrid 只需调用既有公开入口：

```cpp
void compute_melspectrogram(const std::vector<double>& y,
                            int sr, int n_fft, int hop_length, int n_mels,
                            double f_min, double f_max,
                            std::vector<double>& output);
```

必须移植的生产代码均在 `src/melspec.cpp`：

1. `FFTPlan32` / `Radix4Stage32`；
2. `Lane16f`、`Batch16fBuffer`、`apply_batch16f`、`recover_power_batch16f` 及 worker Batch16 分支；
3. AVX2 回退 `Lane8f`/`Batch8fBuffer`/`apply_batch8f`/`recover_power_batch8f`；
4. support 区间、原位/成对恢复和 FP64 单帧回退。

实验/测试代码，不属于生产依赖：

- `scripts/fft_micro.cpp`、`fft_batch2/4/8_micro.cpp`、`pack8_micro.cpp`、`recover8_micro.cpp`、`dif8_micro.cpp`
- `scripts/precision_check.py`、`batch_profile.py`、`multi_bench.py`、`smoke.py`、`fft_check.py`
- 所有 `.slurm`、`versions/` raw 与实验归档

Hybrid 推荐策略：目标机 AVX-512 → FP32 B16；AVX2/FMA → FP32 B8；否则/严格 double → FP64。若要单二进制运行时选择，需要额外 CPUID dispatch；当前通过 Makefile 编译目标自动选择，不声称已有运行时 dispatch。

## 8. 回退策略

- 精度/环境异常：`make -B MELSPEC_FP32_BATCH=0`，恢复到 FP64 Batch8。
- 上一稳定完整节点：tag `v2-simd-inplace-stable`（`aa2405b`）。
- 早期稳定节点：`v2-simd-batch8-stable`（`09c9995`）。
- 不需要 FFTW、CUDA、BLAS；没有外部动态库和 Wisdom。
- 回退后必须重新跑官方四组和 `smoke.py`，不能只依赖默认构建。

## 9. 已知限制

1. FP32 是显式近似精度，不是全 double 语义；不能对需要严格 double 的调用方宣称等价。
2. Makefile 按编译目标 ISA 自动选择，但没有在单个二进制中做运行时 CPUID 分派。
3. B16 需要 AVX-512F；通用/ARM 构建应显式选0或8（8还需AVX2/FMA）。
4. B16 每 worker 工作区约1.5 MiB（大 FFT），高于 FP32 B8，但实测仍更快。
5. 尾帧会走 FP64 单帧，因此同一输出文件的最后0～15列使用不同精度路径；官方与边界测试均通过，但这是实现细节。
6. 性能结论仅来自 hepnode3/Xeon 6338 同作业测量；不外推到其他 CPU/编译器。
7. FFTW float+PATIENT 对照依赖其私有静态库和 Wisdom，不属于本候选交付物。

## 10. 复现与证据

```bash
source tmp/session-env.sh
make -B -j16
python3 scripts/smoke.py ./melspec
python3 scripts/fft_check.py
python3 scripts/precision_check.py ./fp64-reference ./melspec
python3 scripts/multi_bench.py fp64=./fp64-reference fp32b16=./melspec \
  --cases default small_mels large_fft narrow_band --runs 6
```

最终冻结作业日志：`raw/final-freeze-*.out/txt`。旧实验数据保留在 `raw/`，不删改。
