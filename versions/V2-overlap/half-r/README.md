# V2-overlap 第二轮：利用实信号对称减少重组 FFT

- 分支：`v2/overlap`；仅修改本 worktree 的实验档案，默认 `src/melspec.cpp` 保持 V1。
- 原实现保留为 `../melspec_overlap_prototype.cpp`，不覆盖、不删除历史。
- 新实现：`../melspec_overlap_half_prototype.cpp`。
- 独立 delta 优化：`../melspec_overlap_half_delta_prototype.cpp`。
- 不 SSH、不提交集群任务。

## 1. 修正 V1 与原 Overlap 的成本

V1 对实数帧执行长度 `N/2` 的复数 FFT。radix-2 每级有 `(N/2)/2=N/4` 个蝶形，
共 `log2(N/2)` 级，因此蝶形及其实现中的复数乘法次数为：

```
C_V1 = (N/4) * log2(N/2)
```

上一轮写成 `(N/2)*log2(N/2)`，多算了一倍，现予以更正。

原 Overlap 对全部 `r=0..B-1` 执行 B 个 S 点复数 FFT；新版本只执行 `r=0..B/2` 的
B/2+1 个，但仍保留每个 FFT 的全部 m 输出用于共轭恢复。

| 每帧运算项 | 8192 / S=128 / B=64 | 16384 / S=128 / B=128 |
|---|---:|---:|
| V1 半长 FFT 蝶形/复乘 | 24,576 | 53,248 |
| 原 Overlap 重组蝶形/复乘 | 28,672 | 57,344 |
| 新 Overlap 重组蝶形/复乘 | **14,784** | **29,120** |
| 原重组前相位复乘 `N` | 8,192 | 16,384 |
| 新重组前相位复乘 `(B/2+1)*S` | **4,224** | **8,320** |
| 子谱滑动复乘 `S*(B/2+1)`（不变） | 4,224 | 8,320 |
| 原 Overlap 上述复乘合计 | 41,088 | 82,048 |
| 新 Overlap 上述复乘合计 | **23,232** | **45,760** |

新方案比 V1 的半长 FFT 复乘少约 5.5%/14.1%，但仍需承担频谱恢复写回、Hann/功率谱和
Mel 投影；V1 也有加窗打包、实数谱恢复等成本。该表只用于形成实验假设，不代表 wall-time 加速比。
原 Overlap 的总复乘显著多于 V1，与上一轮负收益一致。

delta 预计算不改变复乘数量：把每帧重复进行的进入/离开样本读取与减法从
`S*(B/2+1)` 降为 `S`，即 4,224/8,320 降为 128；复数递推本身不变。

## 2. r 与 B-r 的严格对称推导

定义：

```
Z_q[r]  = sum_j x[start+q+S*j] exp(-2πi*r*j/B)
C_r[q]  = Z_q[r] exp(-2πi*r*q/N)
A[r+mB] = FFT_S(C_r)[m],  N=S*B
```

原始子输入为实数，所以 `Z_q[B-r]=conj(Z_q[r])`。但 `C_r` 与 `C_{B-r}` 的相位并不相同，
**不能直接写 `C_{B-r}=conj(C_r)`**。必须使用完整帧频谱的实信号对称性。

对 `1<=r<B/2`、`0<=m<S/2`，令高频残差为 `B-r`：

```
N - ((B-r)+mB) = r + (S-m-1)B
A[(B-r)+mB]  = conj(A[N-((B-r)+mB)])
              = conj(A[r+(S-m-1)B])
```

右侧恰好是低频组 r 的 `S-m-1` 输出，范围为 `S/2..S-1`。
因此只计算低频组的正 m 会丢信息，**必须保留完整 S 点 FFT 输出**，再按下式恢复：

```
low[r,m]  = C_FFT_r[m]
high[B-r,m] = conj(C_FFT_r[S-m-1])
```

### 边界

- `r=0`：子谱和重组输入都是实数。保留 `m=0..S/2`，其中 m=0 是 DC、
  `m=S/2` 是 Nyquist（k=N/2）；其余高频由实序列 FFT 自身的共轭对称覆盖。
- `r=B/2`：是自配对残差，不存在另一个 `B-r` 组。正频率要求
  `B/2+mB <= N/2`，所以保留 `m=0..S/2-1`，不包含 m=S/2。
- `1<=r<B/2`：低频正频率使用 `m=0..S/2-1`；对应高频正频率用
  `conj(z[S-m-1])` 恢复。不会与 Nyquist 重复，Nyquist 仅属于 r=0、m=S/2。

输出元素数验证：r=0 提供 S/2+1 个，r=B/2 提供 S/2 个，
其余 `(B/2-1)` 对残差各提供 S 个，总计 `N/2+1`，没有缺失或重复。

## 3. 实现与正确性

数学原型新增 `assemble_real_half()`，与原来的全 r `assemble()` 和 NumPy 直接 FFT 同时对照。
44 组测试覆盖两种 N、不同输入长度、随机、低幅度、零、DC、Nyquist、冲激和 Hann 边界，
全部通过；见 `../raw/prototype-half-correctness.json`。

C++ 半数组版本与 delta 版本均通过：

- 8 项 smoke；
- 官方四组全量 expected，误差与 V1 一致；
- 随机/低幅度/1024 刷新边界的独立输入回归：12 项比较全部 PASS，
  且三种 Overlap 输出与 V1 的 float32 结果逐元素相同（`../raw/input-regression.json`）。

delta 版本只把 `y[old+q+N]-y[old+q]` 预计算到每帧数组，随后供所有 r 使用，
与半数组对称优化分开测量。

## 4. 性能与阶段计时

测试程序：V1、原全 r Overlap、新半数组 Overlap、新半数组+delta。
`ab_variants.py` 使用轮转顺序的多次全量交错测试，同时记录 wall time 和子进程 CPU time。
单线程诊断副本强制线程上限为 1，避免 taskset/hardware_concurrency 陷阱。
`stage_profile.py` 独立插桩 delta 版本，阶段为：

- 子频谱递推（含 delta 与周期精确刷新）；
- 重组相位乘法与位反转填充；
- 128 点 FFT 蝶形；
- 高低频恢复写回；
- 频域 Hann 与模平方（当前代码中的融合阶段）；
- Mel 投影。

### 4.1 单线程全量交错结果

每个用例三轮，顺序轮转并反转；四个程序的输出均再次通过 expected。
以下为 CPU-time 中位数（秒）和每轮 `variant/V1` 比值的中位数；wall-time 趋势一致。

| 用例 | V1 | 原 full | 新 half | half+delta | full/V1 | half/V1 | delta/V1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 7.267 | 9.273 | **6.188** | **6.135** | 1.293 | **0.859** | **0.859** |
| large_fft | 15.799 | 17.750 | **11.632** | 11.781 | 1.123 | **0.740** | 0.747 |

对应 wall-time 中位比：default 的 half/half-delta 为 0.862/0.860，
large_fft 为 0.740/0.747。半数组优化在两种 FFT 规模上都显示稳定收益。

delta 是独立的次要实验：default 的 `delta/half` CPU 比中位数为 **0.993**（约快 0.7%），
large_fft 为 **1.009**（反而慢约 0.9%）。因此不把 delta 宣称为普遍收益，
最终选择以多线程结果和各阶段成本为依据。

### 4.2 多线程全量交错结果

保留各实现的 16 线程连续帧分区，每个用例三轮。下表分别给出累计 CPU time 与
进程 wall time 的逐轮比值中位数；两者分母不同，不用 CPU 总秒数代替 wall time。

| 用例 | full/V1 CPU | full/V1 wall | half/V1 CPU | half/V1 wall | delta/V1 CPU | delta/V1 wall |
|---|---:|---:|---:|---:|---:|---:|
| default | 1.337 | 1.234 | **0.868** | **0.926** | **0.857** | **0.906** |
| large_fft | 1.244 | 1.211 | **0.781** | **0.816** | **0.786** | **0.824** |

对应 wall-time 中位数：default 的 V1/half/half-delta 为 1.284/1.184/1.158 秒；
large_fft 为 2.374/1.969/1.919 秒。四个程序均再次通过全量 expected。

半数组优化在单线程和多线程、两种 FFT 规模下均有稳定收益。delta 的额外影响较小且不稳定：
default 中略有益；large_fft 单线程略慢，多线程直接 `delta/half` 比值又有正有负。
由于逐轮中位数在不同负载下不具备传递性，不拿 `delta/V1` 与 `delta/half` 的中位数做代数推导。
主结论是半数组对称优化，delta 不作为必要组成。

### 4.3 分阶段成本

单线程 half-delta 诊断副本的全量结果如下。包含 `chrono` 插桩开销，只用于分析构成，
不替代上面的未插桩 A/B。Hann 与模平方在当前实现中融合，未人为拆成额外数组阶段。

| 阶段 | default 秒 / 占比 | large_fft 秒 / 占比 |
|---|---:|---:|
| 子谱递推/刷新/delta | 0.484 / 8.09% | 0.985 / 8.20% |
| 重组相位乘法/位反转填充 | 0.825 / 13.79% | 1.670 / 13.91% |
| 128 点 FFT 蝶形 | **2.638 / 44.09%** | **5.251 / 43.73%** |
| 高低频恢复写回 | 0.420 / 7.02% | 0.851 / 7.09% |
| Hann + 功率谱 | 1.330 / 22.23% | 2.712 / 22.58% |
| Mel 投影 | 0.287 / 4.79% | 0.539 / 4.49% |
| 合计（插桩） | 5.984 / 100% | 12.008 / 100% |

128 点 FFT 仍是最大单项，但相位乘法与 Hann/功率谱合计已超过它的约八成。
delta 只优化约 8% 的递推阶段，因此收益小且容易受数据布局影响是合理结果。
两次诊断输出也都通过全量 expected，原始数据见 `../raw/stage-default.json` 和
`../raw/stage-large.json`。

## 5. 最终结论

1. 上一轮 V1 蝶形数确实多算了一倍，已同步修正主报告。
2. 实信号对称可以将重组 FFT 从 B 组降为 B/2+1 组，但必须保留低频组全部 m 输出，
   再用共轭关系恢复高频残差；DC、Nyquist 与自配对组均已单独验证。
3. 新 half 版本在本地获得稳定收益：多线程 wall-time 相对 V1，
   default 约快 **7.4%**，large_fft 约快 **18.4%**；单线程与 CPU-time 也同向。
4. half-delta 在 default 多线程下约快 **9.4%**，large_fft 约快 **17.6%**。
   delta 的独立影响很小且方向不完全稳定，不作为必要优化；保留 half 作为更简洁的主结果。
5. **值得在后续获准后做同节点集群 A/B，但本轮不 SSH、不提交集群任务。**
   不继续扩大该路线，也不修改默认 V1 或 master。

所有实现均只存在于实验档案：原 full、新 half、half-delta 分别独立保存。
主 `src/melspec.cpp` 未改变。

## 6. 复现入口

```bash
source tmp/session-env.sh
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  src/main.cpp versions/V2-overlap/melspec_overlap_half_delta_prototype.cpp \
  -o tmp/binaries/melspec-overlap-half-delta

$PY versions/V2-overlap/input_regression.py
$PY versions/V2-overlap/ab_variants.py --case default --rounds 3 --single-thread \
  --out versions/V2-overlap/raw/ab4-default-single.json
$PY versions/V2-overlap/stage_profile.py --case default \
  --out versions/V2-overlap/raw/stage-default.json
```
