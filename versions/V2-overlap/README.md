# V2-overlap：高重叠 STFT 的计算共享研究

> 第二轮更正：V1 半长复数 FFT 的蝶形数应为 `(N/4)*log2(N/2)`，本文表格已修正。
> 利用完整 m 输出的实信号共轭对称，可将重组 FFT 从 B 组降为 B/2+1 组；
> 见 [half-r 专项实验](half-r/README.md)。

- Worktree：`melspec-v2-overlap`，分支 `v2/overlap`
- 共同 BASE：`15f90558b4ab83ad0e6500e84aa807a942564229`
- V1 算法 SHA-256：`3b00a2d777a20de397c3a2e407b60f6a21258940470851cbeb82c4c21c64f957`
- 路线：滑动子 FFT + 频谱重组 + 频域周期 Hann；与 FFTW、自研 SIMD 路线独立。
- 本轮不针对音频内容，不改变接口，不使用近似频谱。

## 1. 为什么不能由 98% 重叠率直接推断加速

`hop=128`，8192/16384 点帧分别重叠 63/127 个帧间隔。重叠只说明输入数据重复，
不说明任意窗函数的 FFT 可以廉价更新。频谱是全局量：一个新采样会通过相位项影响所有频点。
真正的收益必须来自可复用的代数分解，而不是跳过旧样本或重复使用旧频谱。

以下记 `N=n_fft`、`S=hop_length`、`B=N/S`、`K=N/2+1`，所有运算均为 double。

## 2. 直接 Sliding DFT：复杂度和访存均不合适

未加窗帧 `A_t[k] = sum_{n=0}^{N-1} x[tS+n] exp(-2πikn/N)`。
每次前进一个采样可使用：

```
A_{t+1}[k] = exp(+2πik/N) (A_t[k] - x_old + x_new)
```

每移动 128 个采样，若逐点直接更新，全部 `K` 个频点需要 `128*K` 次复数递推：

| N | K | 每帧直接 Sliding DFT 更新 | V1 半长复数 FFT 蝶形循环（约） |
|---:|---:|---:|---:|
| 8192 | 4097 | 524,416 | 24,576 |
| 16384 | 8193 | 1,048,704 | 53,248 |

即使用分块离开/进入样本，离开块与进入块的差仍需在全部目标频率上求 DFT，
直接计算仍是 `O(S*K)`。每次更新都要读写 `K` 个复数，并施加相位乘法；
不仅运算次数高，递推状态还会长期保留舍入误差。因此不采用直接全谱 Sliding DFT。

## 3. 可复用的滑动子 FFT 分解

取 `N=S*B`。对每个相位 `q=0..S-1`，定义长度 B 的实数交错子序列：

```
z_{t,q}[j] = x[tS + q + S*j],       j=0..B-1
Z_{t,q}[r] = sum_j z_{t,q}[j] exp(-2πi*r*j/B)
```

由 `k=r+m*B` 可得未加窗完整帧频谱：

```
A_t[r+m*B] = sum_q Z_{t,q}[r] exp(-2πi*r*q/N) exp(-2πi*m*q/S)
           = FFT_S( Z_{t,q}[r] exp(-2πi*r*q/N) )[m]
```

这给出了严格的计算共享：下一帧 `t+1` 的每个子序列仅前进一个元素：

```
Z_{t+1,q}[r] = exp(+2πi*r/B)
               (Z_{t,q}[r] - x[tS+q] + x[tS+q+N])
```

子输入为实数，只保存/更新 `r=0..B/2`；负频率用共轭恢复。
随后对每个 `r` 做一次 S 点复数 FFT，重组 `A_t[k]`。

### 3.1 主要运算量

| 项目 | 8192 / 128 | 16384 / 128 |
|---|---:|---:|
| 子 FFT 长度 B | 64 | 128 |
| 每帧重组 FFT 数量 × 长度 | 64 × 128 | 128 × 128 |
| 重组蝶形循环 `B*(S/2)*log2(S)` | 28,672 | 57,344 |
| 子谱递推 `S*(B/2+1)` 次复数更新 | 4,224 | 8,320 |
| 重组前相位乘法 | 8,192 | 16,384 |
| V1 半长 FFT 蝶形循环 `(N/4)*log2(N/2)`（对照） | 24,576 | 53,248 |

固定 S=128 时，每帧渐近成本由 V1 的 `O(N log N)` 降为
`O(N log S + N)`，即相对 N 的近线性；但正频谱和相位访问本身有 `Ω(N)` 下界，
不可能借重叠进一步变成亚线性。初版全文误把 V1 蝶形数写成了两倍；修正后可见，
**原全 r 重组的蝶形数反而比 V1 多约 16.7%/7.7%**，还另有相位乘法与子谱更新。
第二轮的 B/2+1 组重组才将该项降至 14,784/29,120，详见 half-r 报告。
每帧仍有 `N` 次重组前复数乘法、子谱读写、频谱写入及功率/Mel 投影。
小 FFT 的循环/位反转开销和不自然顺序的频谱访问也可能抵消理论收益。

### 3.2 内存

每个工作线程主要状态：

- 子谱 `(B/2+1)*S` 个 complex<double>，约 64 KiB / 128 KiB；
- 重组后的正频谱约 64 KiB / 128 KiB；
- 长度 B/S 的临时 FFT 缓冲区及 double 功率谱；
- 所有线程只读共享 `N` 个重组相位，约 128 KiB / 256 KiB。

状态能放入单核 L2，但每帧多次遍历，不能忽略 L1/L2 流量。
各线程处理连续帧区间并持有独立子谱，无跨线程递推状态或写共享。

## 4. 周期 Hann 可在频域精确施加

```
w[n] = 1/2 - 1/4 exp(+2πin/N) - 1/4 exp(-2πin/N)
X_t[k] = 1/2 A_t[k] - 1/4 A_t[k-1] - 1/4 A_t[k+1]
```

索引按 N 周期处理。实输入只需保存正频谱：

```
A[-1]     = conj(A[1])
A[N/2+1] = conj(A[N/2-1])
```

所以 DC、Nyquist 也可精确计算。该关系不是近似，也不是三次 FFT；
只需一次未加窗频谱及每频点三点组合。代价是频域相减可能比时域先加窗更容易发生抵消，
因此必须做实际误差验证，不能只证明代数相等。

## 5. 数值漂移与周期精确重建

子谱递推的相位模长理论为 1，长期运行仍会累积舍入误差；误差不通过后续 FFT 自动消失。
采用每 1024 帧从原始 PCM 精确重算全部子 FFT 的策略：

- 每个线程以自己的首帧为第 0 帧，在 `frame % 1024 == 0` 时精确重建；
- 最多连续 1023 次递推，重建成本按 1024 帧摊销很小；
- 不递推完整输出频谱，避免把一帧误差继续带入全部后续帧。

原型同时检查不刷新与 1024 帧刷新。以下是不同时刻重组频谱相对 NumPy 直接 FFT 的检查点误差，
不是“所有帧最大值的穷举证明”：

| N / 帧数 | 末帧不刷新 max_abs | 不刷新 scaled | 1024 刷新 max_abs | 刷新 scaled |
|---|---:|---:|---:|---:|
| 8192 / 192397 | 2.41e-10 | 1.06e-12 | 1.84e-11 | 8.06e-14 |
| 16384 / 192333 | 4.89e-10 | 2.15e-12 | 3.41e-11 | 1.50e-13 |

scaled 的分母为 `max(1, max(abs(reference)))`。原始记录见
`raw/prototype-drift-8192-full.json` 与 `raw/prototype-drift-16384-full.json`。
这些数据说明 double 下误差可控，但仍须以最终 Mel 输出的官方容差为验收依据。

## 6. 原型正确性

`prototype.py` 直接实现上述公式，不依赖正式 C++ 代码。44 组测试全部通过：

- N=8192/16384，第 0/1/7/1024 帧及多种输入长度（含非整帧尾长）；
- 随机 PCM、2^-20 低幅度、全零、DC、Nyquist、帧首尾冲激；
- 未加窗频谱与 NumPy FFT 对照；
- 频域 Hann 与时域周期 Hann FFT 对照，包含 DC/Nyquist；
- 20,000 帧和完整约 192k 帧递推。

判定阈值为缩放后最大误差 `<1e-10`，实测通过；未加窗/Hann 的最大 scaled 误差分别为
`6.69e-16` 和 `4.89e-16`，DC/Nyquist 最大绝对误差为 `2.84e-14`。
新增长度矩阵见 `raw/prototype-correctness.json`。命令：

```bash
source tmp/session-env.sh
$PY versions/V2-overlap/prototype.py --drift-frames 192397 --drift-nfft 8192 \
  --out versions/V2-overlap/raw/prototype-drift-8192-full.json
$PY versions/V2-overlap/prototype.py --drift-frames 192333 --drift-nfft 16384 \
  --out versions/V2-overlap/raw/prototype-drift-16384-full.json
```

## 7. 其他方向及未采用原因

### 7.1 每次只更新 Mel 需要的频点

`f_max=22050` 只覆盖约 23% 的正频点，但直接滑动更新仍需 `O(S*K_used)`；
 Hann 还需要相邻频点。8192/16384 下仍约 24 万/48 万次复数更新，不能优于 V1。

### 7.2 Chirp-Z / Bluestein 更新离开/进入块

块差频谱可写成 Chirp-Z，用卷积把 `O(S*K)` 降为约 `O(M log M)`，
`M >= S+K-1`。但每帧通常需正向、逆向两个完整复数 FFT、chirp 乘法和全谱递推，
其常数、临时内存及长期漂移均不优于本方案的短 FFT 重组。它是有效数学路线，
但在本工作负载没有明确的工程优势，本轮不作为主线。

### 7.3 只缓存相邻帧的小 FFT/旧输出

普通 Cooley-Tukey 的交错子序列在帧移动后相位和序列位置会改变，
不能直接缓存任意层中间结果；本文使用的是上述可证明的特定分解。
也不能跨时间帧复用功率谱，因为 Hann 权重和非线性模平方都随帧变化。

## 8. 参考资料

- Eric Jacobsen, [Understanding and Implementing the Sliding DFT](https://www.dsprelated.com/showarticle/776.php)：单样本 SDFT 递推及其计算特点。
- Zafar Rafii, [Sliding DFT with Kernel Windowing](https://zafarrafii.com/Documents/Presentations/Rafii%20-%20Sliding%20DFT%20with%20Kernel%20Windowing%20-%202021.pdf)：频域窗核的一般背景；本文使用周期 Hann 的三点精确形式。
- 本文的交错子谱重组公式由 `N=S*B` 直接展开 DFT 推导，`prototype.py` 用 NumPy 独立验证，不声称来自上述资料。

## 9. 工程实现与正确性

C++ 工程原型保持了 double、16 线程连续帧分区和稀疏 Mel，不支持分解时回退 V1。
另外实现了重组输入位反转融合、residue-major 连续频谱写入，避免额外交换和跨步写回。
完整实验代码归档为 `melspec_overlap_prototype.cpp`，SHA-256：
`d6ab95e8c23ab562a5718d1cfd85420be747e2b26ff24114ee0c5417b9bb3d92`。

正确性结果：

- 本地 8 项 smoke（官方四组、默认 CLI、小 FFT）全部 PASS；
- 3 秒随机 PCM 与 V1 对照均满足官方容差，并覆盖 4096 点及刷新边界；
- **官方输入四组全量输出全部 PASS**，见 `raw/full-correctness.txt`：

| 用例 | max_abs_err | 结果 |
|---|---:|---|
| default | 3.12e-2 | PASS |
| small_mels | 1.17e-2 | PASS |
| large_fft | 1.25e-1 | PASS |
| narrow_band | 6.25e-2 | PASS |

这些误差与 V1 的官方结果一致。`raw/full-correctness.txt` 中的一次顺序计时只用于运行记录，
不作为性能结论。

## 10. 本地性能与否定结论

测量脚本每对交替改变 V1/V2 的先后顺序，同时记录 wall time 和
`getrusage(RUSAGE_CHILDREN)` 的 user+system CPU time，使用配对中位数而不是只取有利最小值。
真正的单线程对照在 `tmp/` 生成把线程上限替换为 1 的诊断副本；**仅使用 taskset 不够**，
因为 `std::thread::hardware_concurrency()` 在本环境不会随 cpuset 降为 1。
早期未意识到这一点产生的“单核”数据已废弃，不用于结论。

| 用例 | 三组 CPU-time 比值 V2/V1 | 配对中位数 | 解释 |
|---|---|---:|---|
| default | 1.370 / 1.237 / 0.697 | **1.237** | 方向不完全一致，中位数为慢约 24%，不能宣称加速 |
| large_fft | 1.413 / 1.114 / 1.136 | **1.136** | 三组均较慢，中位数为慢约 14% |

原始每次 wall/CPU time 见 `raw/ab-default-cpu-time.json` 和 `raw/ab-large-cpu-time.json`。
default 第三组出现反向结果，说明共享宿主机的频率/负载扰动无法完全由 CPU time 消除；
因此不把任何单次最快值当作加速证据。多线程切片也曾出现方向不一致，同样不用于宣称收益。

**原全 r 实现结论：不推荐进入集群正式 Benchmark。** 该分解在数学上正确，
固定 hop 时具有更好的渐近形式，但全量短 FFT 重组、相位乘法、子谱更新和频谱访存的
常数成本抵消了理论收益。第二轮利用实信号对称将重组 FFT 减半后，结论发生变化；
新结果见 [half-r 专项实验](half-r/README.md)，不再沿用这里的性能否定结论。

最终 `src/melspec.cpp` 已恢复为 V1，SHA-256 与 BASE 完全一致：
`3b00a2d777a20de397c3a2e407b60f6a21258940470851cbeb82c4c21c64f957`。
慢速原型只留档于本目录，不参与 Make 构建，避免影响 V1 或其他路线的对照。
恢复后 8 项 smoke 再次全部 PASS，见 `raw/final-v1-smoke.txt`。

## 11. 复现工程原型与 A/B

```bash
source tmp/session-env.sh
mkdir -p tmp/binaries
make -j4
cp melspec tmp/binaries/melspec-v1
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  src/main.cpp versions/V2-overlap/melspec_overlap_prototype.cpp \
  -o tmp/binaries/melspec-v2-overlap

# 四组全量正确性
$PY scripts/bench.py tmp/binaries/melspec-v2-overlap --case all --runs 1

# 真正单线程的交错 wall/CPU-time 对照；V1 从共同 BASE 生成
$PY versions/V2-overlap/ab_benchmark.py --case default --pairs 3 --single-thread \
  --candidate-source versions/V2-overlap/melspec_overlap_prototype.cpp \
  --out versions/V2-overlap/raw/ab-default-reproduce.json

# 多线程对照
$PY versions/V2-overlap/ab_benchmark.py --case default --pairs 5 \
  --baseline tmp/binaries/melspec-v1 --candidate tmp/binaries/melspec-v2-overlap \
  --out versions/V2-overlap/raw/ab-default-multithread-reproduce.json
```

V1 对照保留在 `tmp/binaries/melspec-v1`，公共 PCM/expected 只读；本轮未执行 SSH 或集群任务。
