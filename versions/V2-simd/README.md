# V2 Agent B — 自研 FFT / SIMD

> 后续集群基准、AVX-512 与跨帧实验见 [advanced/README.md](advanced/README.md)；
> 稳定节点后的支持裁剪与原位融合见 [next/README.md](next/README.md)；
> 最后一轮 FP32 跨帧实验与 Hybrid 结论见 [fp32/README.md](fp32/README.md)。
> 当前最佳为 FP32 Batch16（AVX-512）/ FP32 Batch8（AVX2），FP64 Batch8 保留回退。

- Worktree：`melspec-v2-simd`，分支：`v2/simd`
- 共同 BASE：`15f90558b4ab83ad0e6500e84aa807a942564229`
- 最终算法提交：`e4b1dfa1a95b3554a60f14c949d6ad77db8fcac4`
- 最终 `src/melspec.cpp` SHA-256：`78c968fedcf8d30207af7f19c36dc739275df85a8e733bbb5178535bb60ed552`
- 路线：保持 double、16 线程帧级并行、稀疏 Mel 和 CLI，仅优化手写 FFT 与谱恢复；未使用 FFTW 或其他现成 FFT 实现。

## 1. 最终结果

在同一 WSL/i7-14650HX 环境中，保存 V1 二进制并与最终候选交替运行。每轮交换 A/B 先后次序，排除顺序测量的大部分漂移；每组各运行 9 次。下表为**完整进程 wall time 的中位数**，包括读写，不是纯 FFT 时间。

| 用例 | V1 中位数 (s) | V2 中位数 (s) | 中位数加速比 | V1 / V2 最小值 (s) | 正确性 |
|---|---:|---:|---:|---:|---|
| default | 1.704378 | 1.225720 | **1.3905×** | 1.247274 / 0.952887 | PASS |
| small_mels | 1.438944 | 0.961421 | **1.4967×** | 1.413379 / 0.906744 | PASS |
| large_fft | 3.381666 | 2.469000 | **1.3697×** | 3.127040 / 2.242800 | PASS |
| narrow_band | 3.188539 | 2.363833 | **1.3489×** | 3.080191 / 2.205389 | PASS |

四组全量输出均满足官方 `rtol=1e-3, atol=1e-4`；V1/V2 的最大绝对误差分别为：

| 用例 | V1 max_abs_err | V2 max_abs_err |
|---|---:|---:|
| default | 0.03125 | 0.03125 |
| small_mels | 0.01171875 | 0.01171875 |
| large_fft | 0.125 | 0.125 |
| narrow_band | 0.0625 | 0.0625 |

原始逐次时间、程序路径和误差见 [`raw/final/ab-v1-r9.json`](raw/final/ab-v1-r9.json)。WSL 宿主机负载在测量期间有明显变化，因此同时保留中位数和最小值，**不将最小值比值或本地中位数加速比当作集群成绩**。各独立实验在不同时间测量，其倍率也不能连乘。

## 2. 独立提交与算法设计

| 提交 | 改动 | 设计要点 |
|---|---|---|
| `0deb589ab216bbd05098e79d37619d33d52bbfda` | B1 位反转融合 | 利用位反转表的对合性，在加窗/偶奇打包时直接 `out[reversed[i]]` 写入，删除 FFT 内独立随机交换 |
| `863b7ce06ee810a7d2c254ea969705aead4069de` | B2 radix-4 | 合并相邻两级 radix-2；每四路使用 `W^j/W^2j/W^3j`，预计算第三组因子；奇数 log2 先做一次 radix-2 |
| `d1fb1a7949815c206ddbbaaa221f6e19385b7e99` | B3 SoA+AVX2 | 实部/虚部及旋转因子改为 SoA，跨四个蝶形使用 AVX2/FMA，保留标量回退 |
| `fdfc2cc9293355b1144c0b99ce020e7ee69a5eb9` | B4 谱恢复 | 对称 bin 反向加载，一次计算四个恢复频点和模平方，不再构造临时复数 |
| `84a5d6995aca0d7336826a500c0e48dc3186eb0f` | B5 首级 DFT4 | 4×4 转置同时处理四个独立 DFT4；首级旋转因子全为 1，不读表、不做复数乘法 |
| `e4b1dfa1a95b3554a60f14c949d6ad77db8fcac4` | B6 len=8 阶段 | 四个子块一组，广播相同旋转因子并跨块 SIMD；计算后按原 SoA 位置写回 |

### B1：成本是否只是转移

V1 先做连续打包，再在 FFT 内按表随机交换。B1 改为连续读取音频和窗口、按表散写，FFT 直接从置换后的数据开始。它消除了一个完整交换 pass，但散写本身仍有成本，因此不把 profile 中的位反转比例直接当成收益。

### B2：base-2 位反转下的 radix-4

不能直接套假设 base-4 数字反转的标准公式。按 V1 两级 radix-2 的数据流，四路为：

```text
x0 = a0
x1 = a1 * W^(2j)
x2 = a2 * W^j
x3 = a3 * W^(3j)
```

再计算 `x0±x1`、`x2±x3` 及 `-i(x2-x3)`。与两个独立的 radix-2 阶段相比，每四路复数乘法由 4 次降为 3 次，并减少一级完整读写。`W^3j` 在 plan 构建时计算，热循环没有额外乘法。

### B3/B4：布局与真实 SIMD

`ComplexBuffer` 使用独立的 `re[]/im[]`，每个 radix-4 stage 的六组旋转因子也为 SoA。通用循环用 AVX2 一次处理四个 double 蝶形，复数乘法显式写成 FMA，避免 `std::complex` 特殊值语义带来的控制流。谱恢复用反向加载取得 `a[half-k]`，直接计算 `xr*xr+xi*xi`。

这些路径只在编译器同时定义 `__AVX2__` 和 `__FMA__` 时启用，否则使用相同数学公式的标量回退。未加入无法在本地运行的 AVX-512 专用实现。

### B5/B6：早期小规模 stage

- quarter=1 时，四个 DFT4 通过 4×4 转置组成四个 SIMD lane；首级因子全为 1，省去旋转因子加载和乘法。
- quarter=2 时，四个 len=8 子块共享每个 j 的旋转因子。使用分组装载、向量计算、对齐暂存和写回；该路径只影响 large_fft/narrow_band，default 不执行。

## 3. 独立性能实验

以下均为交替 A/B 的**中位数加速比**，每组 7 次。它们用于判断单步方向，不是同一时刻的累计倍率。

| 实验 | 对比 | default | small_mels | large_fft | narrow_band | 结论 |
|---|---|---:|---:|---:|---:|---|
| B1 | vs V1 | 0.994× | 1.058× | 1.152× | 1.015× | 保留；default 基本持平，其他组改善但需复核 |
| B2 | vs B1 | 1.126× | 1.250× | 1.167× | 1.129× | 保留，radix-4 有独立收益 |
| B3 | vs B2 | 1.108× | 1.117× | 1.058× | 1.037× | 保留，收益较温和但为后续向量化提供布局 |
| B4 | vs B3 | 1.048× | 1.064× | 1.009× | 1.097× | 保留，谱恢复多数用例改善 |
| B5 DFT4 | vs B4 | 1.135× | 1.109× | 0.980× | 0.975× | 保留；后两组不走该分支，差异视为噪声 |
| B6 len=8 | vs B5 | 0.961× | 未测 | 1.050× | 1.078× | 保留；default 不走该分支，大 FFT 改善 |

原始数据在 `raw/b1/` 至 `raw/b6/`。最后两组中“不经过新分支”的理论路径差异，正好说明本地运行噪声不能忽略。

## 4. 失败实验与调试记录

1. **直接套标准 radix-4：错误。** 首次实现假设 base-4 数据布局，n=4 即失败。改为按 base-2 数据流推导后，又发现奇数 log2 尺寸的 `W^3j` 未生成，n=8 失败。最终 `W^3j` 对所有相关 stage 预计算，1～16384 全尺寸通过。
2. **4×4 转置列顺序：错误。** 初版把第 1/2 列交换，n=16 回归失败；修正 unpack/permute 的列映射后全尺寸通过。错误版本均未提交。
3. **B5a：将奇数 log2 的 radix-2 移到最后一级，性能退化且不保留。** 虽然末级可连续向量化，但新增整表读取可能抵消收益。相对 B4 的中位数倍率为 default 0.994×、small_mels 0.964×、large_fft 0.943×、narrow_band 0.849×。该路线没有进入提交。
4. **B5b 仅用于诊断。** 在 B5a 上加 DFT4 转置确有改善，但没有因此保留 B5a；而是将 DFT4 独立移植回 B4，形成提交 `84a5d69`。
5. **顺序 benchmark 曾给出矛盾结果。** small_mels 和大 FFT 在宿主负载变化时出现异常值，因此后续统一使用交替 A/B，并保留完整逐次数据。

## 5. 正确性验证

最终候选完成：

- `scripts/smoke.py`：随机 PCM 的官方四组、CLI 默认值及 n_fft=1/2/8，全部 PASS。
- `scripts/fft_check.py`：公开 `compute_rfft` 对 1～16384 每个二次幂尺寸，分别测试随机、冲激、正弦输入；NumPy 仅作测试参考，全部 PASS。
- 官方参数 0.5 秒切片、3 次运行：四组全部 PASS；切片不作为性能成绩。
- 全量 V1 交替对照：四组全部 PASS。
- 最终 perf 的 default/large_fft 输出再次与全量 expected 比较，均 PASS。
- AVX2 原生构建及 `-march=x86-64` generic 构建均通过 smoke；generic 不启用 AVX2 intrinsic 路径。

复现：

```bash
source tmp/session-env.sh
make -B -j4
"$PY" -B scripts/smoke.py
"$PY" -B scripts/fft_check.py
"$PY" -B scripts/bench.py --case all --slice 0.5 --runs 3
"$PY" -B scripts/ab_bench.py tmp/results/v1/melspec ./melspec \
  --cases default small_mels large_fft narrow_band --runs 9
```

## 6. Perf、火焰图与汇编证据

最终候选使用 `task-clock:u / 499 Hz / DWARF` 采样，两个用例均 `Total Lost Samples: 0`。本地 WSL 没有硬件 PMU，不能报告 IPC、cache miss 或内存带宽。

| 独立符号 self 占比 | default | large_fft |
|---|---:|---:|
| `FFTPlan::apply_permuted` | 32.87% | 40.41% |
| `recover_power` | 9.43% | 7.77% |
| worker `_M_run()` | 53.47% | 48.38% |
| `compute_melspectrogram` | 3.46% | 3.04% |

worker 仍包含打包、Mel 及其他内联代码，**不能将 `_M_run()` 解释为线程管理开销，也不能把独立 FFT 符号占比当成全部 FFT 成本**。函数报告见 `raw/final/*-report.txt`。

- [default 火焰图](default.svg)
- [large_fft 火焰图](large_fft.svg)

`objdump -d -C melspec` 确认最终二进制存在 packed YMM FMA，例如 `vfmsub231pd`、`vfmadd231pd`，见 [`raw/final/packed-fma-evidence.txt`](raw/final/packed-fma-evidence.txt)。手工 intrinsic 不会简单表现为编译器报告中的“loop vectorized”，所以以反汇编为补充证据；generic 反汇编未找到这些 packed FMA。未仅凭指令名称宣称性能，性能结论来自上述 A/B wall time。

## 7. 集群候选与限制

当前 HEAD 就是推荐进入集群验证的候选，无额外构建依赖：

```bash
make -B -j16
mkdir -p tmp/logs
# 后续由统一流程决定远端专属目录和 Slurm 提交，不在本阶段自行执行。
```

集群 Xeon Gold 6338 支持 AVX2/FMA，`-march=native` 会启用本文已本地验证的 AVX2 路径。**没有新增 AVX-512 专用路径，也没有在本地执行或宣称 AVX-512 收益。** 集群上应重新做同节点 V1/V2 对照、四组全量正确性及 profile；不能用本地倍率替代。

本阶段未 SSH、未提交 Slurm、未修改 master/其他 worktree，未 Merge 或 Cherry-pick。`src/main.cpp`、`validate.py` 和题目 `data` 未修改；公共 venv 与数据仅只读使用。大型二进制、perf.data、采样输出和实验 scratch 均留在 `tmp/`，未纳入 Git。
