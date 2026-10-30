# V3 CPU-Pipeline 独立实验记录

## 研究审查与采用顺序

三个 Kimi K3.1 Pro Highspeed 子任务已完成静态研究，原文归档于 `research/`。
主 Agent 采用以下隔离顺序，不把未测量的改动合成一个“优化包”：

1. S1：直接压缩 Mel 构造，保留原稀疏点积。
2. B：每 worker 单份 FFT scratch + B 行 power，扫描 B=1/2/4/8/16。
3. P：双前缀矩查询，只有数值与端到端收益均成立才采用。
4. M：准备阶段细分、power scratch 和线程/NUMA；不违法规避 output 初始化。

研究明确否定两项过快结论：双前缀和的 `O(B+M)` 不保证优于现有稀疏 taps；
`output.resize` 在固定 `vector<double>&` 与受保护 main 下不能用 UB 跳过初始化。

## S1：直接压缩三角滤波器

公开 `build_mel_filterbank` 保持原稠密接口；compute 内部改用私有
`CompressedMelFilter {first,last,weights}`。每个滤波器：

- 用 lower/upper bound 找到包含左右端点的候选区间；
- 对候选 bin 执行与原实现相同的 rising/falling/min/max/enorm 运算顺序；
- 仅按 `==0.0` 从两端裁零，不使用 epsilon，不重新归一化；
- 空滤波器规范为 `first==last==n_bins`、空 weights；
- 每帧仍按相同 k 顺序执行 `acc += weights[k-first] * power[k]`。

目标只包括减少 `O(n_mels*n_bins)` 稠密准备和权重跨度；不改变 FFTW、窗、输出布局或求和顺序。
验证使用 8 项 smoke，以及相对固定 A2 的 39 组随机/边界/满幅/冲激/Nyquist 对照和短输入拒绝。
性能采用 A2/S1 交错五次全量进程计时和独立阶段插桩。结果在后续小节补齐。

### S1 本地验收结果

- smoke 8/8 PASS；与 A2 的 39 组合法输入输出 **逐元素完全相同（max_abs_err=0）**，3 组短输入拒绝。
- 首轮遇到宿主机高负载，A2 自身 median 从先前约 0.80 s 变为 1.63 s，原始数据保留但不作决策。
- 第二轮交错五次（min / median，秒）：

| 用例 | A2 min | S1 min | A2 median | S1 median | median 比值 A2/S1 |
|---|---:|---:|---:|---:|---:|
| default | 0.743597 | 0.721554 | 0.782846 | 0.752934 | 1.040× |
| small_mels | 0.526625 | 0.521432 | 0.540131 | 0.537411 | 1.005× |
| large_fft | 1.725985 | 1.639888 | 1.815768 | 1.682165 | 1.079× |
| narrow_band | 1.581933 | 1.547803 | 1.632437 | 1.552680 | 1.051× |

四组全部 PASS。S1 作为低风险候选保留，仍需在集群同节点复测；small_mels 的收益很小，不宣称所有用例都有显著提升。
原始逐次结果见 `raw/s1-*`，高负载数据见 `raw/s1-local-noisy.txt`。

## B：小帧块与输出局部性（待实施）

只保留 B 行 frame-major power；每个 b 仍复用同一 input/fft_out 并调用原单变换 Plan。
然后按 `m → b → k` 计算 Mel，使同一 Mel 行内 B 个输出连续。不是 plan_many/Batch FFT。

每 worker 核心 scratch：`S(N,B)=(16+4B)N+8B+16` 字节，不随 n_frames 增长。
B=1 应退回基线行为；B=2/4/8/16 独立扫描，每 worker 分区尾块单独处理。
风险包括推迟 power 消费、L2/TLB 压力和额外循环，只有 compute/端到端改善才保留。

### B 本地正确性与当前结论

五种 B 均通过 8 项 smoke、39 组对照和 3 组短输入拒绝，且与 A2 输出逐元素相同。
两轮本地性能受宿主机负载影响明显：default 多次偏向 B=2/4，large_fft 多次偏向 B=1/2；
small_mels 和 narrow_band 没有跨轮稳定的全局赢家。阶段插桩也显示 read/write 和系统负载扰动，
因此本地只保留候选框架，**不把任何 B 宣称为稳定收益，也不从最小值挑一个参数**。
默认仍先设 B=1，集群同节点交错对照后再决定。原始逐次数据见 `raw/frame-block-*`。

### B 集群结果（作业 131，hepnode2）

固定 A2 与各 B 在同一作业交错三次；median（秒）：

| 用例 | A2 | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.997651 | **0.986654** | 1.006925 | 1.000184 | 0.997991 | 1.007012 |
| small_mels | 0.674159 | 0.665229 | **0.660255** | 0.663142 | 0.669333 | 0.677444 |
| large_fft | 1.369392 | **1.358000** | 1.375441 | 1.365065 | 1.375139 | 1.391057 |
| narrow_band | 1.315782 | **1.303558** | 1.319319 | 1.311275 | 1.325339 | 1.331077 |

B=1 完整官方验证 4/4 PASS：`0.965 / 0.646 / 1.349 / 1.289 s`。
**采用 B=1，不采用更大的帧块**；B=2 在 small_mels 的微小优势不足以抵消其他用例回退。
原始逐次结果见 `raw/cluster/cpu-131.out`、`compare-131.json`。

## P：双前缀矩查询（待独立验证）

`P[k+1]=P[k]+power[k]`，`Q[k+1]=Q[k]+k*power[k]`；
仿射权重段查询为 `a*(Q[r]-Q[l])+b*(P[r]-P[l])`。仅消费 power，可与 FFT backend 解耦。
中心用 lower_bound 切分，空段/空滤波器显式处理。每帧成本由总 taps T 变为 `O(n_bins+n_mels)`，
但新增递推、两次 scratch 写入、分段端点访问及前缀相减抵消，不能用渐近复杂度直接宣称更快。
不得默认 clamp 负值来掩盖抵消；必须保留直接 sparse 回退及压力测试。

### P 本地结果：不采用

P1 通过 smoke 和 39 组对照，但与 sparse 的最大输出差为 `3.0517578125e-05`，不再 bitwise 相同。
官方形状 T/(B+M) 分别约 0.858/0.879/0.884/0.309，说明窄带尤其不适合额外全 bins 前缀。
本地 median：default 1.747→1.531 s（变快），但 small_mels 0.799→1.246 s、
large_fft 2.362→2.885 s、narrow_band 2.558→3.055 s，整体回退，不能按 default 单例采用。

主动压力测试（实际 C++ 公式）还显示：
- `dynamic_prefix` 在 `power[0]=1e12`、其余仅约 1e-6 时，max_rel≈67.67、168 个负 Mel 结果；
- smooth 输入 max_rel≈3.82e-12；zero 和末 bin 冲激无误差。
这证明“官方样例通过”不能排除前缀抵消。**默认保留 sparse，不增加 clamp，不采用 P1。**
代码保留为 `MEL_PROJECTION=moments` 的独立失败实验；原始结果见 `raw/p1-*`、`moment-pressure.txt`。

## M：内存与线程（待独立验证）

- setup 需要拆成 filter、output.resize、FFTW scratch/plan、power、线程生命周期。
- `reserve+resize`、assign、并行清零、raw storage 再复制都不能合法消除 output 的现有初始化成本。
- power scratch 与 output 接口不同，可在证明每分支写后读后评估合法免零初始化/worker first-touch。
- `OMP_*` 不直接控制 std::thread；8/12/16 与绑核必须按实际 cpuset 做独立 A/B。
- 任何诊断 affinity/chrono/日志都不能混入正式二进制和正式计时。

### M1：setup 细分结果

新增的三个计时宏在默认构建中为空；插桩前后 B1 二进制 SHA-256 完全相同
（`887f92fe43040d0251588741a034c5883915b9bed0b92724c2cd35c84b6aa5c2`），smoke 8/8 PASS。

本地 default / small_mels：
- `mel_setup_s`：56.9 / 50.7 μs，压缩构造已经不是可优化的大项；
- `output_resize_s`：0.173689 / 0.031118 s，与输出 double 容量相符，是 setup 的主要部分；
- `scratch_setup_s`：0.001230 / 0.001043 s，power/FFTW scratch 合计仅约 1 ms。

因此不为约 1 ms 的 scratch 引入未初始化 owner 或额外 NUMA 复杂性；output 初始化虽贵，
但在受保护 main 与 vector 接口下没有合法“跳过”路线，保持现状并转向线程实验。
原始插桩见 `raw/m1-profile.txt`。
