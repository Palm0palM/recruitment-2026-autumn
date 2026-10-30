# V3 MPI 端到端性能模型（1 / 2 / 4 节点）

## 0. 范围与结论边界

本文仅依据以下固定材料建立可手算模型：

- `versions/V3-mpi-spike/README.md` 中 V1 与 FFTW A2 ESTIMATE 正式基线；
- `src/main.cpp` 的 PCM 读取、int16 → double 转换、float32 转换与 root 单文件写出语义；
- `src/melspec.cpp` 的 A2 worker、16 线程上限、frame 独立计算和 row-major `[n_mels, n_frames]` 输出；
- `validate.py` 的四组参数、shape、三次运行取最小 wall time 及正确性口径；
- `scripts/fftw-cluster.slurm` 的单节点 `1 task × 16 CPU` 对照口径。

**没有进行集群测量，也没有可声称的 MPI 实测带宽、延迟、初始化时间或扩展效率。** 下文所有 `B_eff`、`α`、launcher/init、I/O 和阶段计算时间均为待主 Agent 实测参数。文中的带宽数值代入仅是量纲/算术敏感性示例，不是集群数据或性能预测。

本次只写本文档；未运行 shell、测试、Git、SSH 或 Slurm，故无测试命令和结果可报告。

---

## 1. 固定基线与比较口径

正式表中的数字是每组三次的最小 wall time，包含普通 CLI 进程内的读取、计算、转换和写出。

| 用例 | V1，hepnode3（s） | A2 ESTIMATE，hepnode2（s） | 表内 V1/A2 时间比 |
|---|---:|---:|---:|
| default | 1.537 | 0.985 | 1.560× |
| small_mels | 1.196 | 0.650 | 1.840× |
| large_fft | 2.634 | 1.356 | 1.942× |
| narrow_band | 2.578 | 1.300 | 1.983× |

记同节点、同构建、同 I/O 条件下的单进程 A2 时间为 `T0`。MPI 候选的强扩展量定义为：

```text
S(p) = T0 / T_MPI(p)
E(p) = S(p) / p = T0 / (p × T_MPI(p)),  p ∈ {1, 2, 4}
```

注意：

1. 上表 V1 与 A2 原本就在不同节点测得，不能把这个时间比当作 MPI 强扩展收益。
2. MPI 正式比较还需要 **同节点单进程 A2** 对照；同时保留 `p=1` 的 MPI+launcher 版本，用来分离 MPI 封装开销。
3. launcher 内的应用 wall time 应包含 `mpirun`/等价启动器、MPI 初始化和退出。Slurm 排队时间另记为作业周转指标，不混入应用 wall time，也不能悄然删掉 launcher 成本。

---

## 2. 数据量公式与四组输出量级

符号：

```text
p = 节点数 = MPI rank 数（每节点 1 rank、16 CPU）
S = PCM sample 数
K = n_fft
H = hop_length
F = n_frames = 1 + floor((S-K)/H)
M = n_mels
```

`src/main.cpp` 和输出语义给出：

```text
原始 int16 PCM：             P = 2S bytes
完整 double y：              Y = 8S = 4P bytes
完整 double 输出（不汇集）： D = 8MF bytes
最终 float32 输出：          O = 4MF = D/2 bytes
```

四组用例由相同 PCM 得到。按 frame 公式反推、忽略 PCM 末尾不足一个 hop 的余数，`S ≈ 24,634,880`，所以：

```text
P ≈ 49,269,760 B  = 49.270 MB = 46.987 MiB
Y ≈ 197,079,040 B = 197.079 MB = 187.949 MiB
```

实际模型应使用文件实际大小 `P_file`，而不是用 frame 数反推值替代；两者最多相差一个不足 hop 的尾部。

| 用例 | `M × F` | float32 输出 `O` | 若错误地 Gather double 的流量 `D=2O` |
|---|---|---:|---:|
| default | 256 × 192397 | 197,014,528 B ≈ **197.015 MB / 187.888 MiB** | 394.029 MB |
| small_mels | 64 × 192397 | 49,253,632 B ≈ **49.254 MB / 46.972 MiB** | 98.507 MB |
| large_fft | 256 × 192333 | 196,948,992 B ≈ **196.949 MB / 187.825 MiB** | 393.898 MB |
| narrow_band | 256 × 192333 | 196,948,992 B ≈ **196.949 MB / 187.825 MiB** | 393.898 MB |

因此 default 中三个容易混淆但都约 197 MB 的量是：

- double `y`：约 197.079 MB；
- float32 输出：约 197.015 MB；
- PCM 只有约 49.270 MB，而 double 输出则约 394.029 MB。

**输出必须先在 worker 转成 float32 再 Gather。** 若先汇集 double，default/large/narrow 的输出集体通信量会翻倍，且 root 还要串行完成全量 cast。

---

## 3. 帧划分、重叠 Scatter 和输出顺序

### 3.1 确定性的连续 frame 分块

rank `r` 使用：

```text
begin_r = floor(F × r / p)
end_r   = floor(F × (r+1) / p)
f_r     = end_r - begin_r
```

每帧独立，worker 内仍保持 A2 的 FFT、power、mel 累加顺序。正常同构建下，跨 rank 分块本身不需要改变单帧浮点运算顺序。

### 3.2 带重叠的输入 Scatter

rank `r` 需要的 sample 区间是：

```text
[begin_r × H, (end_r-1) × H + K)
```

非空分块的本地 sample 数为：

```text
L_r = (f_r - 1)H + K
```

覆盖全部 `F` 帧时，所有分块之和为：

```text
S_used = (F-1)H + K
Σ L_r  = S_used + (p-1)(K-H)

Scatter 原始 PCM 总 payload = 2[S_used + (p-1)(K-H)]
Scatter double y 总 payload = 8[S_used + (p-1)(K-H)]
```

default 的 `K-H=8064`，即使 `p=4`，全部重叠仅多 `3×8064=24,192` 个 sample：raw PCM 多 48,384 B，double 多 193,536 B。相对 49/197 MB 总量很小，但首尾 rank、空分块和短输入仍必须测试。

### 3.3 float32 Gather 不能直接拼接本地 row-major 缓冲区

本地计算自然产生 `[M, f_r]` row-major；全局目标是 `[M, F]` row-major。按 rank 直接拼接会变成：

```text
rank0 的全部 mel × 本地 frames，rank1 的全部 mel × 本地 frames，...
```

这并不是全局 `[M,F]`。可选实现及其成本必须进入模型：

1. 每个 mel 使用 `Gatherv`/派生类型写入 root 的正确位移；可能引入最多 `M` 次 collective 的延迟项；
2. 本地转为 frame-major 后一次 Gather，root 再转置；增加本地/root pack、unpack 与内存流量；
3. 其他确定性布局方案，但必须把 pack 和地址转换时间单独测量。

下文用 `J_layout(p)` 表示这一不可省略的布局成本，用实际测量的 `T_gather` 表示所选方案的总时间，不能假定一次理想的大块 Gather 就足够。

---

## 4. 1 / 2 / 4 节点端到端模型

### 4.1 单进程 A2 的阶段分解

先把正式基线拆成待测阶段：

```text
T0 = R + X0 + U + C + J0 + W

R  = root 读取完整 PCM
X0 = 单进程完整 int16 → double 转换
U  = window/filterbank/FFTW plan 等 setup
C  = 单节点 A2 worker 计算（16 线程，含其线程调度成本）
J0 = 单进程 double → float32 转换及输出准备
W  = float32 文件 write + close
```

`U` 和 `C` 可用 A2 已有 setup/worker 计时辅助区分，但正式 wall time 与阶段 profiling 分开运行。`R/X0/J0/W` 均须实测，不能从总时间猜测。

### 4.2 MPI 的通用串行阶段模型

对输入方案 `v`，无重叠的保守端到端式为：

```text
T_MPI,v(p) = L(p) + I(p)                         launcher + MPI_Init
           + R                                   root 完整 PCM 读取
           + X_v(p) + A_in,v(p)                  转换 + Bcast/Scatter
           + U + C_p                             每 rank setup + 最慢 rank worker
           + J_cast(p) + J_layout(p)             worker float32 cast + 布局/pack
           + A_out(p)                            float32 输出 Gather/Gatherv
           + W                                   root 写出完整输出
           + Z(p) + Q(p)                         最终同步/状态归约 + MPI_Finalize
```

其中：

- `L(p)`：外部 launcher 到应用可开始执行的启动成本；
- `I(p)`：MPI 初始化、通信器建立；
- `C_p ≥ 0`：实际最慢 rank 的 worker 时间；理想负载时才近似 `C/p`；
- `A_out`：只传 float32，root 接收量约 `O(1-1/p)`；
- `Z`：所有 rank 成功/失败状态的同步，不允许某 rank 失败而 root 仍报告成功；
- `Q`：退出/finalize 成本；
- `p=1` 时输入/输出 collective 数据项应为零或接近零，但 `L/I/Z/Q` 不一定为零。

理想强扩展近似只替换 `C_p ≈ C/p`，其他项不能自动除以 `p`。setup 在每个 rank 重复，root I/O 不随节点数减少。

### 4.3 四种输入方案

| 方案 | root 工作 | 输入 collective | worker 转换 | 主要额外流量 |
|---|---|---|---|---|
| BC-PCM | 读 PCM | 完整 `P` Bcast | 每 rank 都转换完整 `S`，关键路径仍约 `X0` | 小 PCM，但重复转换和完整 y 内存 |
| BC-Y | 读 PCM、完整转 double | 完整 `Y=4P` Bcast | 无完整重复转换 | 输入通信是 BC-PCM 的 4 倍 |
| SC-PCM | 读 PCM | 带重叠 Scatter，总量约 `P` | 各 rank 只转换本地重叠区间 | 输入总量最小，cast 可近似分摊 |
| SC-Y | 读 PCM、完整转 double | 带重叠 Scatter，总量约 `Y` | 无 | root cast 不扩展，输入通信约为 SC-PCM 的 4 倍 |

对应的转换/通信关键路径为：

```text
BC-PCM: X = X0,             A_in = Bcast(P)
BC-Y:   X = X0（root）,     A_in = Bcast(Y)
SC-PCM: X ≈ max_r X_local,r A_in = Scatter(P + overlap)
SC-Y:   X = X0（root）,     A_in = Scatter(Y + overlap)
```

完整 PCM Bcast 后每个 rank 仍需完整 double `y`，default 仅这一项就约 197 MB/rank；Scatter 的本地 double 输入才近似按 `1/p` 下降。

### 4.4 带重叠 Scatter 的理想重叠下界

只有输入 Scatter 与 worker 计算重叠时，可把串行的

```text
A_in + X_local + U + C_p
```

理想化为下界：

```text
T_input+compute ≥ max(A_in, X_local + U + C_p)
```

实际还应加首块到达前的启动时间、chunk 调度、最后一块以及负载不均；root 完整读 PCM 若未做流式读取，也不能与 Scatter 重叠。输出 Gather、root 写出和最终同步在此简单模型中仍显式串行计入。

所以定义两档情景：

- **保守情景**：launcher/init、读、转换、Scatter、计算、cast/layout、Gather、写出、同步依次计入；`C_p` 使用实测最慢 rank，而非 `C/p`。
- **理想重叠情景**：仅输入 Scatter 与本地转换/setup/worker 按上式重叠；其余通信和 I/O 仍使用实测值。
- **绝对下界（非可实现预测）**：令 launcher、所有通信和同步为零、负载完美，得到 `U + C/p + 固定 I/O/cast`。它只用于判断理论余量。

---

## 5. 集体通信的可手算近似

### 5.1 `α–β` 模型必须注明算法口径

可使用最简单的 root 串行注入近似：

```text
T_msg ≈ n_msg × α + L / B_eff
```

- `α`：该消息尺寸和 MPI/UCX 路径下的有效延迟；
- `B_eff`：collective 端到端有效带宽，不是网卡标称带宽；
- `L`：root 串行发送/接收模型中的关键路径总字节；
- `n_msg`：该简化模型的消息数，root 与每个非 root 对端交互时约为 `p-1`。

对优化后的 tree Bcast、流水线 Scatter 或单个 Gather，实际 `α` 项和关键路径字节不同。**优先直接测目标 size 的 `T_bcast/T_scatter/T_gather`，不要从标称网络参数推这些时间。** root 串行式只是保守手算工具。

### 5.2 default 的非 root 聚合通信量

下表忽略 Scatter 的很小重叠附加量；Bcast 的总量按 `(p-1)×完整消息`，Gather/Scatter 按非 root 部分。它是通信量表，不是实测时间表。

| 操作 | p=2 | p=4 |
|---|---:|---:|
| 完整 PCM Bcast | 49.270 MB | 147.809 MB |
| 完整 double y Bcast | 197.079 MB | 591.237 MB |
| 重叠 PCM Scatter，非 root 合计 | ≈24.635 MB | ≈36.952 MB |
| 重叠 double Scatter，非 root 合计 | ≈98.540 MB | ≈147.809 MB |
| float32 输出 Gather，root 接收 | ≈98.507 MB | ≈147.761 MB |

忽略延迟时，输入加输出的手算通信时间为：

| 方案 | p=2：秒 | p=4：秒 |
|---|---:|---:|
| BC-PCM + float Gather | `0.14778 / B_GB` | `0.29557 / B_GB` |
| SC-PCM + float Gather | `0.12314 / B_GB` | `0.18471 / B_GB` |
| BC-Y + float Gather | `0.29559 / B_GB` | `0.73899 / B_GB` |
| SC-Y + float Gather | `0.19705 / B_GB` | `0.29557 / B_GB` |

这里 `B_GB` 是以 **GB/s（十进制）** 表示、且为便于手算而假定两种 collective 相同的有效带宽。实际应分别使用 `T_in` 和 `T_out`，并加延迟项。

纯算术敏感性示例：若仅假设 `B_GB=1`，上表数值分别就是约 0.123–0.739 s；若 `B_GB=10`，全部除以 10；若为 0.1，全部乘以 10。**这些假设不是集群测量，也不包含 launcher、MPI init、α、pack、I/O 或同步。**

root 文件系统的独立量级为：

```text
T_read  ≥ P / R_GB = 0.04927 / R_GB 秒（还不含转换和文件系统延迟）
T_write ≥ O / W_GB = 0.19701 / W_GB 秒（default）
```

`R_GB/W_GB` 同样必须实测。root 写出不随 `p` 减少，不能把它算作可扩展计算。

---

## 6. 强扩展余量与“约 1 秒时是否值得”

### 6.1 理想计算表与绝对下界

令不可扩展/未分摊部分为：

```text
V = T0 - C
```

则零通信、零 launcher、完美负载的理想式为：

```text
T_ideal(p) = V + C/p
```

| 用例 | p=1 | p=2 | p=4 |
|---|---:|---:|---:|
| default | `V+C = 0.985` | `V+C/2`；绝对下界 `0.4925` | `V+C/4`；绝对下界 `0.24625` |
| small_mels | `V+C = 0.650` | `V+C/2`；绝对下界 `0.325` | `V+C/4`；绝对下界 `0.1625` |
| large_fft | `V+C = 1.356` | `V+C/2`；绝对下界 `0.678` | `V+C/4`；绝对下界 `0.339` |
| narrow_band | `V+C = 1.300` | `V+C/2`；绝对下界 `0.650` | `V+C/4`；绝对下界 `0.325` |

“绝对下界”列甚至令 `V=0`，因此不是预测；真实时间一定还需计入不可扩展 I/O 和 MPI 成本。理想效率为：

```text
E_ideal(p) = T0 / [p(V+C/p)] = T0 / (pV+C)
```

只要 `V>0`，就有 `E_ideal(p)<1`；实际 `C_p>C/p` 和额外通信会进一步降低效率。

### 6.2 盈亏平衡条件

在完美 `C/p` 近似下，把所有相对单进程 A2 新增的 launcher、init、通信、布局、同步、负载不均等成本合并为 `H_p`：

```text
T_MPI(p) ≈ T0 - (1-1/p)C + H_p

只有满足：H_p < (1-1/p)C，MPI 才比同节点 A2 快。
```

即使极其乐观地令 `C=T0`，额外成本预算也不得过下表；真实 `C<T0`，真实预算更小。

| 用例 | p=2 的绝对收益上限 `T0/2` | p=4 的绝对收益上限 `3T0/4` |
|---|---:|---:|
| default | 0.4925 s | 0.73875 s |
| small_mels | 0.3250 s | 0.48750 s |
| large_fft | 0.6780 s | 1.01700 s |
| narrow_band | 0.6500 s | 0.97500 s |

结论只能写成条件判断：

1. 当前单节点 default 约 1 秒时，**多节点并非天然不值得，也绝不能据 FFT 可并行就声称值得**。两节点最多从计算分摊中省不到 0.4925 秒；所有新增成本必须低于实际 `C/2`。
2. small_mels 的两节点预算更紧，绝对上限只有 0.325 秒；large_fft/narrow_band 的 A2 总时间较长，可能有更多余量，但仍须分解 `C`。
3. default 每次约有 197 MB float 输出必须回 root 并写出。Scatter 只能降低输入流量，不能消除输出 Gather 和 root 写出。
4. 完整 double y Bcast 会把输入从约 49 MB 放大到约 197 MB；raw PCM Bcast 减少网络量但每个 rank 重复完整转换。两者要实测，不可仅按网络量判优。
5. 若 launcher/init、collective、pack、同步和负载不均之和超过上表实际预算，继续增加节点只会降低效率。只有端到端 `T_MPI<T0` 且四组验证通过，才算端到端价值。

---

## 7. 主 Agent 测量方案

### 7.1 必须先测的参数

| 参数 | 测量方法/注意事项 |
|---|---|
| `R, X0, U, C, J0, W` | 在同节点 A2 上分阶段计时；正式 wall 与 profiling 分开 |
| `L(p), I(p), Q(p)` | 外部 timer 包住完整 launcher；rank 内用 `MPI_Wtime` 记录 init/finalize，外部总 wall 不可被内部阶段和替代 |
| `C_p` | 每 rank 单独测 worker，取最大值；同时记录 frame 数和 16 线程实际 cpuset |
| `T_bcast(P/Y)` | 对四组实际输入尺寸测目标 collective，不用标称带宽换算 |
| `T_scatter` | 使用真实重叠区间和非均匀计数，覆盖首尾 rank |
| `T_gather` | 使用真实 float32 数量和最终布局方案，包含必要的派生类型或多次 Gatherv |
| `J_cast/J_layout` | 与 collective 分开计时，防止把 pack 隐藏成“带宽” |
| `Z` | 测最终状态归约/同步；profiling barrier 自身成本单独列出 |
| `R_GB/W_GB` 或原始 I/O 时间 | 在作业实际使用的节点本地 scratch 和文件路径上测；记录文件系统与缓存条件 |

所有阶段并行段以 **最慢 rank** 为准，不能只打印 rank0。外部总 wall、各阶段最大值和原始每 rank 日志同时保留。

### 7.2 运行矩阵与资源一致性

1. 先做 `p=1` 普通 A2 与 `p=1` MPI+launcher 对照，分离 MPI 封装成本。
2. 对 `p=1/2/4`、四组官方用例分别测：BC-PCM、SC-PCM；若需要判断转换与流量取舍，再测 BC-Y、SC-Y。
3. SC-PCM 至少比较“完全串行输入→计算”和“chunk Scatter 与 worker 重叠”两种，验证理想重叠是否兑现。
4. 每节点保持 `1 task × 16 CPU`、`--hint=nomultithread`；多节点即每节点 1 rank，总 rank 数等于 `p`。显式记录 rank mapping、cpuset、CPU 型号和实际线程数，不写 `--mem`。
5. 输出写 root 所在计算节点的本地 scratch；记录路径和文件系统。不能用 RAM 输出候选对比普通磁盘基线而不说明。
6. 正式口径每例三次并保留所有原始时间；报告最小值以匹配现有表，同时报告中位数/波动避免只凭一次快运行判断。
7. 串行作业，记录 commit、源码/二进制 hash、FFTW/OpenMPI/UCX 版本、节点列表和完整 launcher 配置。历史 wrapper/runtime 故障须先用小作业验证，不默认可用。

### 7.3 正确性门槛先于扩展结论

- 先随机 PCM、边界输入、smoke，再完整四组 expected；官方要求 `rtol=1e-3, atol=1e-4`，四组 PASS。
- 覆盖 `F<p`、空 rank、首尾 rank 的重叠区间、短输入异常和尾帧。
- 检查输出字节数、float32 小端、确定性 `[M,F]` row-major 顺序，重点抓“按 rank 直接拼接”的布局错误。
- 任一 rank 失败必须让整体失败；root 不得写出貌似成功的部分结果。
- 用普通单进程 A2 输出辅助逐字节/逐值定位布局问题，但最终仍以完整 `validate.py` 为准。

### 7.4 结果表建议与决策规则

每个方案、每个用例填写：

| p | 外部总 wall | launcher+init | read+cast | input comm | setup | worker max | cast+layout | output Gather | root write | sync+finalize | `S(p)` | `E(p)` | PASS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 待测 | 待测 | 待测 | 待测/零 | 待测 | 待测 | 待测 | 待测/零 | 待测 | 待测 | 待算 | 待算 | 待测 |
| 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待算 | 待算 | 待测 |
| 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待算 | 待算 | 待测 |

主 Agent 最终应同时回答：

1. 相对 **同节点普通 A2** 是否端到端变快，而非只快于 MPI p=1 或只看 worker；
2. 实测 `H_p` 与 `(1-1/p)C` 的盈亏平衡关系；
3. Scatter 重叠节省了多少，是否被 Gather/layout/root 写出抵消；
4. 从 2 到 4 节点时效率如何变化，是否出现通信/I/O 主导；
5. 四组是否全部保持正确性，以及收益是否只在个别用例成立。

在这些实测完成前，最稳妥的结论是：**A2 单节点约 0.65–1.36 秒，多节点可用收益预算很小；float32 Gather 降低了输出流量，但 launcher、collective 和 root I/O 必须全部计入。是否值得由上述端到端条件决定，不能由理论 FFT 加速或假设带宽决定。**
