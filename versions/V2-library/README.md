# V2 Library-First — 极端科学计算库驱动实验

- Worktree：`melspec-library-first`；分支 `exp/library-first`。
- 固定起点：`b4552f3962751b5a02d2aa56798907c5a1c29afc`（FFTW A2 ESTIMATE，已集群验证）。
- 本实验独立于 FFTW/SIMD/Overlap/MPI 主线，不修改 master、其他 worktree、`src/main.cpp`、`validate.py` 或 `data`。
- AI：OpenCode / GPT-6.1 Sol（实现、测试、测量、文档）。

## 0. 核心回答

**如果不再自己写具体数值内核，只组合成熟科学计算库，最接近可用的是 Python A3（NumPy + SciPy FFT + SciPy CSR）和 C++ sparse（FFTWf + Eigen sparse）；但它们都没有超过 FFTW A2。**

原因不是“库不够快”，而是每个库只覆盖一个局部：

- FFTW 确实可以完整替代手写 FFT，并且 A2 已经证明收益；
- BLAS 的稠密 GEMM 在数学上会做 `n_mels × n_bins` 的稠密乘法，而 Mel 矩阵只有约 4k 个非零，稠密路线多做了约 256 倍工作；
- SciPy/Eigen 的通用稀疏乘法能正确完成投影，但每帧/每块的调度、转置和 CSR 迭代开销吞掉了低 FLOPs 的优势；
- Python 解释器、导入和逐块 API 调用是固定成本，题目的总计算只有亚秒级，固定成本占比很高；
- GPU 的 cuFFT/cuBLAS 计算本身很快，但本机进程 wall time 仍被 CuPy 导入、H2D/D2H、dense 投影启动和 197 MB 输出传输拖慢；
- 本题最有价值的库替换仍是 **FFTW 的 R2C**，不是把整个流水线交给高层 Python。

因此，**Library-First 是一条有价值的“上界/可维护性”参考路线，但不是本轮的最终性能路线**。最接近理想的组合是：FFTW 负责 FFT，自写或库稀疏投影负责 Mel，外层自己做分块/线程/缓冲复用。完全“库编排、零手写内核”会失去后两项控制权。

## 1. 题目语义与正确性契约

程序必须保持：

- 16-bit LE 单声道 PCM → `/32768` 归一化；
- `center=False`，不补零；
- 周期 Hann（`fftbins=True`），`n_fft=1` 按题目公式 `w[0]=0`；
- 非归一化前向 real FFT，模平方；
- HTK Mel 尺度 + Slaney 面积归一化；
- float32、小端、`[n_mels, n_frames]` 行优先；
- 官方 `rtol=1e-3, atol=1e-4`。

所有实现均未调用 librosa/essentia/torchaudio，也没有调用现成 Mel API；Mel 三角滤波器和 Slaney 归一化由本实验定义，仅为构造滤波器使用 NumPy 向量运算。

## 2. 实验 A：Python 科学计算栈

### A1：原始 `melspec.py`

提供的参考实现先物化整个 `(n_frames, n_fft)` 加窗矩阵、整个复数谱和功率谱，再转置做 dense `mel_fb @ S`。这是“库调用基线”，但内存随帧数平方级增长：

- default：约 192k × 8192 float64 加窗矩阵约 **12.6 GB**，复数谱约 **12.6 GB**，功率谱约 **6.3 GB**；
- large_fft：对应约 **25/25/12.6 GB**；峰值数十 GB。
- 本地 15 GiB 机器无法全量运行；5 s 切片实测峰值约 **0.78 GB**，只用于阶段观察。

A1 的原始 `melspec.py` 保留不改；`library_melspec.py --mode a1` 是带阶段计时的等价副本，便于取得分阶段数据。全量 A1 只在集群计算节点运行。

### A2：分块 NumPy + SciPy FFT + dense BLAS

- 用 `as_strided` 构造滑窗视图，但**不物化全量窗口矩阵**；
- 每块 `B` 帧：float32 加窗 → `scipy.fft.rfft(..., workers=W)` → 直接计算 `real²+imag²`；
- `power.T` 变连续，`np.matmul(mel_dense_float32, power_t)` 走 NumPy/OpenBLAS；
- 输出全量 `out[n_mels, n_frames]`，按块写入。

A2 的 dense GEMM 是正确的，但做的是 256×4097 的稠密乘，而不是三角滤波器的稀疏乘。它是最直观的“科学库组合”，也是 dense 对照。

### A3：分块 NumPy + SciPy FFT + SciPy CSR

- 与 A2 相同的分块/FFT/功率谱；
- 预构造 `scipy.sparse.csr_matrix(mel_fb)`（仅约 3,736 个非零，default）并做 CSR × dense；
- 尝试了左乘 `CSR @ power.T` 和右乘 `power @ CSR.T` 两种布局；左乘默认更快/更省峰值（右乘反而增加内存和转置）。

### A4：批大小/线程数

`--block-size` 与 `--fft-workers` 独立；`--blas-threads` 控制 BLAS。SciPy FFT 使用内部 worker，BLAS 只在 A2 的 GEMM 阶段使用；没有把 FFT 和 BLAS 嵌套到同一时刻，避免双重 oversubscription。

## 3. 实验 B：C++ 科学计算库栈

实现见 `impl/cpp/library_melspec.cpp`：

- FFTWf `plan_many_dft_r2c` 做 batched R2C（也支持 batch=1 的 `plan_dft_r2c_1d`）；
- 共享一个只读 plan，worker 使用独立 `fftwf_malloc` 输入/输出缓冲；
- 外层最多 16 个 `std::thread`；没有调用 FFTW threads/OpenMP 后端；
- power 由 `real²+imag²` 控制循环完成；
- Mel 两种模式：
  - `dense`：OpenBLAS `cblas_sgemm`；
  - `sparse`：Eigen `SparseMatrix<float, RowMajor>` × dense；
- 输出直接 float32，不经 protected `main.cpp` 的 double 向量。

**关键失败/限制**：

1. FFTWf 的共享 plan 在本机 16 线程表现出明显的扩展损失：单线程 FFT 约 1.75 s，16 线程线程内 FFT 累计约 14.6 s（虽然 wall 仍在 ~0.9 s）；A2 的 double R2C 在同一机器上线程累计约 5.4 s。不能把“库调用”当作自动高并行。
2. 单 plan per worker 更差；共享 plan 是正确方向，但 float FFTWf 仍不如 double A2。
3. Eigen sparse × dense 比 V1 的紧致三角循环慢；CSR 的逐行迭代和临时转置抵消了稀疏 FLOPs。
4. OpenBLAS dense 在数学上多算约 256× 的 Mel 乘，实际也明显更慢。
5. 因此 C++ 库路线没有达到“稳定本地优势”，**没有提交集群正式作业**；只在本地完成正确性/性能/阶段诊断，避免占用受限节点。

`cpp/library_melspec_double.cpp` 是同一代码的 double 试验副本，用于确认精度/类型不是唯一原因；不作为候选。

C++ sparse 的独立 profile（default，block=1，16 workers，线程内累计而非进程 wall）：
window 1.98 s、FFTw execute 14.66 s、power 0.89 s、Eigen Mel 3.16 s、store 1.11 s；
`compute_s` wall 1.40 s。FFT 线程累计约 14.7 s 是网格/缓存/调度综合结果，不能除以 16 当作单线程 FFT；
它说明“FFTW 调用”并没有自动带来 A2 的扩展效率。Eigen Mel 线程累计 3.16 s，也高于 V1 的紧致三角循环。
C++ dense 的 OpenBLAS SGEMM 在数学上多算约 256×，实际更慢；原始阶段日志在 `raw/local/cpp-stage.txt`。

## 4. 实验 C：GPU（可选）

`impl/gpu_melspec.py` 使用 CuPy：

- H2D 复制整段 PCM 一次；
- 每块在 GPU 上建帧、加窗、cuFFT R2C、模平方；
- dense: cuBLAS `mel @ power.T`；sparse: `cupyx.scipy.sparse.csr_matrix`。

GPU 需要独立 `tmp/gpu-venv`（`cupy-cuda13x==14.2.0`），没有修改公共 venv，也没有把 GPU 结果和 CPU 结果混算。设备是 RTX 5060 Laptop 8 GB；结果见硬件表，包含 CuPy 导入、H2D/D2H、计算和写出。

本地 GPU 进程 wall（每次 1 run，CuPy 导入计入）：

| 用例 | GPU dense b512 | 设备 |
|---|---:|---|
| default | 2.124 s | RTX 5060 Laptop 8 GB |
| small_mels | 1.820 s | 同上 |
| large_fft | 2.497 s | 同上 |
| narrow_band | 2.510 s | 同上 |

default 的 device 阶段：H2D 0.05 s、cuFFT 0.14 s、dense Mel 0.19 s、D2H 0.05 s、compute 约 0.98 s；CuPy 导入/解释器启动是 2.9 s 进程 wall 的主要部分。GPU 比 Python CPU A3 快，但仍明显慢于 FFTW A2；GPU sparse 因 CuPy CSR 极慢（default 10.7 s）被保留为失败对照。GPU 结果只代表本地 RTX 5060 + WSL，不与集群 CPU 相除。


## 5. 正确性

| 实现 | 验证 |
|---|---|
| A1 reference copy | 12 组确定性随机/边界输入与官方 expected 对照 PASS；短输入安全拒绝 |
| A2 dense | 12 组 PASS；短输入安全拒绝 |
| A3 sparse | 12 组 PASS；短输入安全拒绝 |
| C++ sparse | 12 组 PASS；3 个短输入拒绝 |
| C++ dense | 12 组 PASS；3 个短输入拒绝 |
| GPU dense/sparse | 官方 default 全量 PASS；短输入拒绝 |

边界输入覆盖 `n_fft=1/2/8/32/2048`、零值、满幅正负、冲激、Nyquist、精确帧边界和官方四组参数。官方全量 expected 由 `scripts/library_bench.py` 每次输出后检查。

A1 的原始 `melspec.py` 在 `n_fft=1` 时使用 scipy 单点 Hann=1 的特例，与题目公式 `w[0]=0` 不同；本实验的所有新实现按题目/V1 语义返回零，不会复制该特例。

## 6. 本地性能

硬件：WSL2，i7-14650HX（24 逻辑 CPU，AVX2/FMA，无 AVX-512），15 GiB RAM + 4 GiB swap；CPU 基准会受宿主机负载影响，表内使用同轮交错运行的最小值，不跨环境比较。

同轮交错、四组全量、各 3 次取最小（local-final；不是官方 validate 作业）：

| 用例 | FFTW A2 | Python A2 dense | **Python A3 sparse** | C++ sparse | C++ dense |
|---|---:|---:|---:|---:|---:|
| default | **0.932** | 9.222 | 5.355 | 1.912 | 4.626 |
| small_mels | **0.715** | 8.302 | 5.067 | 1.604 | 2.370 |
| large_fft | **1.698** | 15.999 | 8.349 | 2.871 | 9.316 |
| narrow_band | **1.647** | 16.608 | 8.850 | 2.819 | 9.827 |

C++ 的 sparse/dense 都不是 A2 对手；Python A3 是 Python 栈中最好的，但仍比 A2 慢约 5–6 倍。每轮输出均与官方 expected 比较，全部 PASS；原始逐次数据见 `raw/local/local-final.json`。

### 阶段拆解（default，A3 空载代表值）

| 阶段 | 时间 |
|---|---:|
| import + 解释器启动（进程 wall 减 internal） | ~1.5–2.5 s |
| PCM 读取/归一化 | 0.05–0.10 s |
| 分块加窗 | 0.33–0.62 s |
| SciPy R2C FFT（16 workers） | 0.69–1.06 s |
| 模平方 | 0.85–1.47 s |
| power 转置 | 0.24–0.51 s |
| CSR Mel | 0.11–0.47 s |
| 输出写盘 | 0.15–0.22 s |

Python 的固定启动/导入成本约 1.5–2.5 s；即使计算全部免费，进程也只能到 ~1.5 s，无法接近 A2 的 0.75–0.93 s。A3 的实际计算约 3.2–3.6 s，其中 FFT 与逐元素 power/加窗合计占绝大多数；Mel 不是主要瓶颈。

### 内存

- A1 5 s 切片峰值约 **0.78 GB**；集群全量 default/large 峰值 **34.3 / 68.1 GB**。
- Python A3 全量 default 峰值约 **0.59 GB**，A2 约 **0.95 GB**（输入 + 输出 + 块内临时）；块处理没有物化全量功率谱。
- C++ sparse 的独立统计约 **0.4 GB**；C++ dense 约 1.0 GB（工具/OpenBLAS 影响，见原始 stats）。
- GPU 进程 host RSS 约 1.2 GB；设备内存池峰值约几十 MB（每块 512×8192 输入与 512×4097 输出）。

### A4 调参

集群 default（同作业，1 次）A3：b64/w16 **6.385 s**，b128/w8 6.452 s，b128/w16 **6.096 s**，b256/w16 6.746 s；A2 dense b128/w16 9.942 s。本地 `a4-clean` 结果见 raw/。结论：block=128、FFT workers=16 是稳定折中；更大 block 降低调用次数但提高工作集/转置成本；更少 worker 在本题（16 物理核）没有优势。BLAS 线程只作用于 A2 dense，不能改变稀疏 Mel 的结论；本地 BLAS 1/4/16 的扫描受宿主机负载影响，保留原始记录而不据此挑一个“最佳 BLAS 线程数”。


## 7. 哪些库替代了什么

| 计算 | 库方案 | 结论 |
|---|---|---|
| PCM 读取/归一化 | NumPy / C++ I/O | 可替代，但 I/O 固定；C++ 更快 |
| 周期 Hann | SciPy `get_window` / NumPy | 可替代；单点语义需按题目修正 |
| FFT | SciPy pocketfft / FFTW R2C | **最成功的替代**；FFTW A2 已证明 |
| 功率谱 | NumPy 向量运算 | 可替代，但中间数组/循环开销显著 |
| Mel 投影 | NumPy dense BLAS | 可替代但数学上多算约 256×；不推荐 |
| Mel 投影 | SciPy CSR | 可替代；Python/CSR 开销使其仅约 0.1–0.5 s，仍不如紧致循环 |
| Mel 投影 | Eigen sparse | 可替代但比 V1 手写三角循环慢 |
| GPU FFT/GEMM | CuPy | 单步快，导入/传输/启动使全进程不占优 |
| oneMKL | 未测试 | 本机和集群没有可确认的许可/安装/runtime；不在未验证时假设可用 |

## 7.1 代码量、依赖与构建难度

| 实现 | 代码量（本实验新增） | 依赖 | 构建难度 |
|---|---:|---|---|
| A1 原始参考 | 提供代码约 174 行；诊断副本仅加计时 | numpy/scipy | 无构建，最简单 |
| A2/A3 共用 | 约 217 行 + 226 行评测/检查脚本 | numpy 2.5.4、scipy 1.18.1 | 无编译；导入/解释器是固定成本 |
| C++ 库栈 | 约 333 行（double 试验另 335 行） | FFTWf 3.3.11、OpenBLAS 0.3.34、Eigen 5.0.1 | 中高：头文件/库/链接/线程/对齐；需独立前缀 |
| GPU 可选 | 约 134 行 | CuPy 14.2.0、CUDA 13.4、独立 venv | 中：wheel 约 66 MB，无编译但 runtime/驱动敏感 |

依赖构建：Python 不需要编译；C++ 的本地系统库版本如上，无 sudo 构建脚本见 `scripts/build_deps.sh`（FFTW float、Eigen 3.4、OpenBLAS 0.3.34；因 C++ 无本地优势未在集群执行）。GPU 的 CuPy 只装在 `tmp/gpu-venv`，共享 `tmp/.venv` 未修改。oneMKL 未安装，不能在本机/集群确认许可、安装和 runtime，故没有把 oneMKL 作为已验证方案。

## 8. 失败与有价值的负面结果

- A1 全量在本地不可行，是内存模型问题，不是算法错误；必须分块。
- A2 dense 虽然正确，但做了什么工作：稠密 256×4097 × 192k ≈ 800 GFLOP；A3 稀疏约 0.8 GFLOP 量级。dense 结果证明“有 BLAS”不等于“该用 BLAS”。
- A3 右乘 CSR 布局没有更快，反而增加内存；保留为失败实验。
- C++ batched FFTWf 的 16 线程扩展明显差于 FFTW double per-frame A2；共享 plan 比 per-worker plan 正确/快，但仍不足以赢。
- GPU sparse 在 CuPy 上极慢（>10 s），dense 的 cuBLAS 好得多，但仍被 Python/CuPy 启动和传输拖累。
- 所有结论都受 WSL 宿主机负载影响；本地表用于方向，不作为集群成绩。

## 9. 集群验证

集群作业 198，hepnode3，`COMPLETED/0:0`，1 node / 1 task / 16 CPU / nomultithread / no mem。作业内 Python 3.14.4、numpy 2.5.4、scipy 1.18.1；cpuset `0-15`；Xeon Gold 6338，503 GiB 内存。先做 A1/A2/A3 四组全量交错（各 1 次），再做 default 的 A3/A2 batch/worker sweep，最后对 A3 跑完整官方 `validate.py --runs 3`。

### 9.1 官方 `validate.py`（A3，3 次取最小）

| 用例 | 时间 s | vs 同作业 librosa | max_abs_err | 状态 |
|---|---:|---:|---:|---|
| default | **4.792** | 2.895× | 6.25e-02 | PASS |
| small_mels | **4.660** | 2.928× | 1.56e-02 | PASS |
| large_fft | **9.555** | 3.401× | 1.25e-01 | PASS |
| narrow_band | **9.375** | 3.463× | 9.38e-02 | PASS |

**四组 4/4 PASS。** 这是 Python A3 的正式集群结果。随后在同一 hepnode3 作业 210 中重新编译 FFTW A2 并与 A3 交错运行三次，得到严格同作业对照：A2 最小 0.981 / 0.665 / 1.352 / 1.299 s，A3 最小 4.949 / 4.640 / 9.623 / 9.383 s，即 A3 慢 **5.05× / 6.98× / 7.12× / 7.23×**。将 A3 的官方三次最小（4.792/4.660/9.555/9.375）与同作业 A2 相除也一致，约 **4.88–7.22×**。不再依赖跨作业的旧 A2 数字。

### 9.1b 同作业 A2/A3 交错（3 次取最小）

| 用例 | FFTW A2 s | Python A3 s | A3 / A2 |
|---|---:|---:|---:|
| default | 0.980714 | 4.948507 | 5.05× |
| small_mels | 0.665246 | 4.640409 | 6.98× |
| large_fft | 1.351756 | 9.623378 | 7.12× |
| narrow_band | 1.298746 | 9.383332 | 7.23× |

同节点、同 16 CPU、同输入、同输出位置；A2 由 BASE `b4552f3` 的 `src/melspec.cpp` 在本作业内重新编译，使用本实验自建 FFTW 3.3.11 double 静态库（库 SHA-256 `31a905e895886a593d4c88d98243fee28b446da9b870a36e200b158a3f3ce9f5`，二进制 `a9805f8c227d1faf471c081eaf2bcd176a659a0edd31efcf6878f1c65a63f05a`）。A3 仍为 b128/w16。

### 9.2 四组全量交错（各 1 次，非官方最小口径）

| 用例 | A1 ref s / RSS | A2 dense s | **A3 sparse s** |
|---|---|---|---|
| default | 30.028 / 34.3 GB | 10.284 | **6.260** |
| small_mels | 29.304 / 34.1 GB | 7.136 | **5.955** |
| large_fft | 65.172 / 68.1 GB | 15.101 | **10.998** |
| narrow_band | 65.096 / 68.1 GB | 14.727 | **10.965** |

A1 的内存规模（34/68 GB）说明“原始 NumPy 全量物化”在本题不可取；A3 的块处理把峰值降到约 **1.0 GB**。A1/A2/A3 的 12 个全量输出均与 official expected 对照 PASS。原始 JSON 包含每次运行、阶段和峰值 RSS。

A3 default 阶段（集群，单次）：import 2.34 s，read 0.077，frame 0.622，FFT 0.695，power 1.468，transpose 0.514，Mel 0.175，store 0.067，compute 3.567，write 0.148；进程 wall 6.26。A2 dense 同轮 compute 7.61，其中 dense Mel 1.64 s，FFT 2.94 s；阶段受宿主机/作业状态影响，不能把两行当作精确同条件微基准，但 dense Mel 的额外成本方向明确。

### 9.3 A4

A3 default 的 batch/worker sweep（集群，1 run）：b64/w16 6.385，b128/w8 6.452，b128/w16 **6.096**，b256/w16 6.746；A2 dense b128 9.942。因此选择 b128/w16。A3 CSR right-layout 在本地反而更慢（5.10 vs 5.00 s），失败数据保留。


## 10. 复现

```bash
git clone <repo> melspec-library-first
cd melspec-library-first
# 只使用本 worktree 的独立环境；不要 setup.sh 修改共享 venv
source tmp/session-env.sh
$PY versions/V2-library/scripts/library_check.py --programs ...
$PY versions/V2-library/scripts/library_bench.py --runs 3 ...
```

Python 依赖：numpy 2.5.4、scipy 1.18.1；无编译。C++：GCC 16.2.1、FFTW 3.3.11、OpenBLAS 0.3.34、Eigen 5.0.1（系统 include），构建见 `scripts/build_cpp.sh`；无 sudo 依赖构建见 `scripts/build_deps.sh`（未在集群实际执行，C++ 未晋级）。GPU：CuPy 14.2.0 / CUDA 13.4，独立 venv。

原始数据、日志和阶段 JSON 归档在 `raw/`；大二进制、venv、perf.data 不提交。上述所有测量均保留每次运行，不篡改单位或把线程累计当作进程 wall。
