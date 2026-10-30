# FP32 跨帧 SIMD FFT 实验

基线：`aa2405b`，FP64 Batch8 + Mel 支持裁剪 + 原位功率谱。稳定标签：
`v2-simd-inplace-stable`。本轮只研究 FP32 跨帧 FFT，不重复 DIF、Stockham、
大规模蝶形重写或输入转置。

## 1. 设计

- FFT 工作数据：FP32 AoSoA，时间帧为 SIMD lane。
- Hann：继续用 double 计算 `y * window`，再 cast 到 float。
- 谱恢复：float FFT 输入转 double，recovery 系数和运算保持 double，功率存 float。
- Mel：权重和累加保持 double；FP32 功率转换后以 double 累加，最终写 main 的 double 输出。
- 尾帧：全局最后 0..7（Batch8）或 0..15（Batch16）走原 FP64 单帧路径。
- 默认宏 `MELSPEC_FP32_BATCH=0`，普通 `make` 仍是 FP64。FP32 通过显式编译宏启用。

缓存/每 worker 工作区（N=n_fft/2，不含尾帧单帧缓冲）：

| 配置 | N=4096 | N=8192 |
|---|---:|---:|
| FP64 Batch8 re/im | 0.50 MiB | 1.00 MiB |
| FP32 Batch8 re/im/power | 0.375 MiB | 0.75 MiB |
| FP32 Batch16 re/im/power | 0.75 MiB | 1.50 MiB |

FP32 B16 的工作区与 FP64 B8 同级，仍远低于 L2；没有物化全量功率谱。

## 2. 最终同节点 A/B

hepnode3，16 物理核，`--hint=nomultithread`，同作业四程序轮转 6 次，四组官方 expected 全部 PASS。
表中为完整进程 median，单位秒。

| 用例 | FP64 Batch8 | FP32 Batch8 | FP32 Batch16 | FFTW float+PATIENT | B16 vs FP64 | B16 vs FFTW |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.865315 | 0.839211 | **0.834009** | 0.855914 | 1.037× | 1.026× |
| small_mels | 0.527069 | 0.495110 | **0.485544** | 0.516943 | 1.085× | 1.065× |
| large_fft | 1.138472 | 1.055000 | **1.035395** | 1.064758 | 1.100× | 1.028× |
| narrow_band | 1.117619 | 1.013177 | **0.999337** | 1.012433 | 1.118× | 1.013× |

B8 相对 FP64 改善约 3%～10%，B16 相对 B8 再改善约 0.6%～2%，四组均略快于 FFTW float+PATIENT。
上表取最终提交后的独立作业 271（`raw/fp32-final-271.*`）；作业 268 的同方向结果略高，但不重复引用其倍率。

## 3. 数值误差与安全余量

相对 FP64 Batch8 的官方四组与合成边界比较（`scripts/precision_check.py`）：

| 用例/输入 | max_abs | RMS | max tolerance ratio | max_rel(ref>1e-6) |
|---|---:|---:|---:|---:|
| default | 0.02344 | 2.26e-4 | 0.00983 | 9.77e-4 |
| small_mels | 0.00781 | 1.46e-4 | 0.00222 | 1.48e-4 |
| large_fft | 0.09375 | 8.27e-4 | 0.01320 | 5.17e-4 |
| narrow_band | 0.06250 | 2.99e-4 | 0.00924 | 4.36e-4 |
| zero | 0 | 0 | 0 | 0 |
| impulse | 2.24e-8 | 7.01e-10 | 1.18e-4 | 3.45e-7 |
| alternating ±full | 3.28e-7 | 2.12e-8 | 0.00328 | 4.17e-3 |
| 1 kHz full-scale sine | 0.03125 | 1.13e-3 | 0.00311 | 6.77e-3 |
| near-Nyquist sine | 1.41e-8 | 3.36e-10 | 1.41e-4 | 5.17e-4 |
| DC step | 0.125 | 6.81e-4 | 0.01213 | 1.07e-3 |
| random PCM | 2.44e-4 | 9.61e-6 | 0.00450 | 1.24e-5 |
| two frames | 2.12e-22 | 4.20e-24 | 2.12e-18 | 0 |

`max tolerance ratio = max(|err| / (atol + rtol*|ref|))`，所有输入均小于 0.014，
即最差仍有约 **70 倍**官方容差余量。这个指标比 max_rel 更适合判断 `allclose` 安全边界。

官方 expected 的 max_abs：B8 为 0.0390625 / 0.015625 / 0.125 / 0.09375，
B16 与 B8 相同量级，四组均 PASS。FFTW float+PATIENT 的 max_abs更大但仍 PASS，
说明绝对值本身不是独立判据。

## 4. 阶段成本

集群诊断作业 269，表中为每帧 ns；FP64 数字来自较早作业 134，仅作方向对照。

| 阶段 | FP64 B8 default | FP32 B8 default | FP32 B16 default | FP64 B8 large | FP32 B8 large | FP32 B16 large |
|---|---:|---:|---:|---:|---:|---:|
| pack | 7,188 | 6,956 | 6,824 | 14,517 | 13,756 | 13,565 |
| FFT | 11,970 | 6,451 | **5,379** | 28,722 | 14,471 | **12,271** |
| recover | 3,103 | 1,916 | **1,401** | 6,301 | 3,796 | **2,781** |
| Mel | 1,359 | 1,008 | 1,000 | 2,743 | 1,799 | 1,885 |

主要收益来自 FFT 和恢复的带宽减半；B16 在 FFT 上比 B8 再快约 15%，代价是更大的
re/im/power 工作区。Mel 本身收益有限，且 B16 large_fft 的 Mel 因 16 帧转换/累加略高于 B8，
不影响总体选择。

## 5. 正确性覆盖

- `smoke.py`：官方四组、CLI 默认、n_fft=1/2/8、tail_2/tail_3，默认/B8/B16 全部 PASS。
- `fft_check.py`：公开 `compute_rfft` 的 1～16384 二次幂，random/zero/impulse/sine 全部 PASS。
- 高动态组合：zero、单样本冲激、满幅交替、满幅 1kHz、近 Nyquist、DC step、随机、两帧输入。
- 官方四组完整 expected 与 FP64、FFTW 同作业交错，B8/B16 均 4/4 PASS。
- 未修改 `src/main.cpp`、`validate.py`、`data`；默认构建仍是 FP64。

## 6. 结论：是否值得进入 Hybrid CPU Backend

**值得作为可选 fast path，但不应直接替换无条件默认。**

- Xeon Gold 6338（AVX-512F）：FP32 Batch16 四组均快于 FP64 Batch8 和 FFTW float+PATIENT，
  相对 FP64 中位数改善约 3.7%～11.8%，且有约 70 倍容差余量。
- AVX2/FMA 机器：FP32 Batch8 可提供 5%～17% 改善；B16 不可用。
- 无 AVX-512/AVX2 或必须严格 double 语义：保留 FP64 Batch8（AVX-512）或 FP64 Batch4（AVX2）回退。
- Hybrid 选择应基于 ISA + 可选运行时开关，而不是只凭编译宏；默认安全性优先。
- FP32 仍改变数学精度语义，答辩和部署必须明确说明；不能把 FP32 结果标成全 double。

建议交付形态：

```text
compile-time feature default: FP64 Batch8 (安全默认)
MELSPEC_FP32_BATCH=8  -> AVX2/FMA fast path
MELSPEC_FP32_BATCH=16 -> AVX-512F fast path
runtime/CLI fallback -> FP64 on unsupported ISA or strict-precision mode
```

## 7. 构建与复现

```bash
# FP64 默认
make -B -j16

# FP32 Batch8（AVX2/FMA）
g++ -std=c++17 -O3 -march=native -pthread -DMELSPEC_FP32_BATCH=8 \
  -Iinclude src/main.cpp src/melspec.cpp -o melspec-fp32-b8

# FP32 Batch16（AVX-512F）
g++ -std=c++17 -O3 -march=native -pthread -DMELSPEC_FP32_BATCH=16 \
  -Iinclude src/main.cpp src/melspec.cpp -o melspec-fp32-b16

# 正确性与同节点 A/B
python3 scripts/smoke.py ./melspec
python3 scripts/precision_check.py ./melspec ./melspec-fp32-b16
python3 scripts/multi_bench.py fp64=./melspec fp32b8=./melspec-fp32-b8 \
  fp32b16=./melspec-fp32-b16 fftw=/path/to/fftw-float-patient-wrap --runs 6
```

原始日志：B8 作业 266、B16 作业 268、精度/profile 作业 269，均在 `raw/`。
