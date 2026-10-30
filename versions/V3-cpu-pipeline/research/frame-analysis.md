# V3 子任务 B：帧分块与数据布局研究

状态：仅静态阅读与循环/内存设计；未修改正式源码，未运行编译、测试或 Benchmark。
所有性能判断都是待实测假设，不构成收益承诺。

## 0. 范围与证据

- 固定 BASE 为 FFTW A2 原生 R2C + ESTIMATE，保持 double、周期 Hann、HTK/Slaney、稀疏 Mel、行主序输出和外层线程分区。
- 只允许逐帧执行现有单变换 `Plan`；本方案不使用、也不需要 `plan_many`、Batch FFT、Wisdom 或 Codelet。
- 依据：`src/melspec.cpp`、`include/fftw_support.h`、本 worktree 的 `versions/V3-cpu-pipeline/BASELINE.md` 与 `versions/V3-cpu-pipeline/README.md`；后两者作为集群 profile 摘要。
- 补充本地阶段构成参考：本 worktree 的 `versions/V2-fftw/README.md` 与 `versions/V2-fftw/raw/local-profile-r2c-estimate.txt`。
- 任务原指定的 `versions/V2-fftw/PROFILE.md` 在当前固定 BASE worktree 中不存在，因此没有虚构其内容；未读取任何其他 worktree。
- 具体伪代码另见同目录 `prototype-loop.md`。

## 1. 当前逐帧循环及跨行写入

### 1.1 当前顺序

`src/melspec.cpp:173-236` 的每个 worker 拥有连续帧区间 `[begin,end)`，核心顺序为：

```text
for t = begin .. end-1:
    pack 当前帧
    对当前帧执行一次共享的单变换 R2C Plan
    计算当前帧 power[0..n_bins)
    for m = 0 .. n_mels-1:
        for k = first[m] .. last[m]-1:
            acc += mel_fb[m][k] * power[k]
        output[m*n_frames + t] = acc
```

每 worker 当前只有一份 `input[n_fft]`、一份 `fft_out[n_bins]` 和一份 `power[n_bins]`。当前帧的功率计算完后立刻做 Mel，因而 `power` 很可能仍热在低级缓存中；这是基线不能被忽略的优点。

### 1.2 输出布局的实际问题

输出是必需的行主序矩阵 `[n_mels,n_frames]`，逻辑元素为：

```text
output[m][t] == output[(size_t)m*n_frames + t]
```

当前固定 `t`、递增 `m`，相邻两次输出存储的逻辑地址相差 `n_frames` 个 double，即字节跨距为：

```text
output_store_stride = 8 * n_frames
```

因此每处理一帧，会同时推进 `n_mels` 条相距很远的输出行写流。对大 `n_frames`，这些写入通常落在不同 cache line 和不同页；普通可写回缓存上的写分配还可能带来 RFO。到下一帧时，每条流只前进 8 字节，但中间已经遍历了其他所有 Mel 行，并执行了下一帧的 pack/FFT/power，较早的输出行缓存线不保证仍然驻留。

这不是“输出总量变大”：输出矩阵大小始终为 `8*n_mels*n_frames` 字节。问题是单帧阶段的写局部性和活跃写流数量。

### 1.3 不能只看输出存储

V2 原始 R2C ESTIMATE 插桩中，worker 阶段时间加总比例为：

| 用例 | pack | FFT execute | power | Mel |
|---|---:|---:|---:|---:|
| default | 10.33% | 71.17% | 6.31% | 12.20% |
| large_fft | 9.27% | 74.87% | 5.35% | 10.51% |

上表只是本 worktree 保留的 V2 本地诊断。集群口径以 V3 摘要为准：`BASELINE.md` 记录 A2 诊断作业 59 的 default 读取 0.180045 s、compute 准备 0.210458 s、compute 全部 0.565350 s、转换/写出 0.222535 s；`README.md` 总结 A2 后 worker FFT 仍约 68–75%，default 读入/准备/写出约占诊断 wall 61%。正式四组 A2 ESTIMATE 最小 wall 分别为 default 0.985 s、small_mels 0.650 s、large_fft 1.356 s、narrow_band 1.300 s，但这些是绝对基线，不是分块后的收益数据。

帧分块不改变 FFT 点数、执行次数或 FFT 数学工作，主要只能影响数据局部性及非 FFT 阶段。default 的非 compute-worker 部分占比很高，所以 worker 内局部收益不能直接换算成相同比例的端到端收益；阶段计时之和也不能代替进程 wall。

## 2. 推荐的 B 帧块：单份 FFT scratch + B 份 power

### 2.1 关键布局选择

推荐每 worker 保留当前已有的单份 FFT 工作区：

```text
input[n_fft]                  // fftw_malloc，double
fft_out[n_bins]               // fftw_malloc，fftw_complex
power[B][n_bins]              // 普通 double，frame-major
```

逻辑索引：

```text
power_b[k] == power[(size_t)b*n_bins + k],  0 <= b < block_frames
```

**不为 B 帧复制 B 份 input/fft_out。** 每个 `b` 仍依次完成“复用同一 input/fft_out 的 pack → 一次 `plan.apply` → 写入该 b 的 power 行”。这样：

1. 满足“B 个连续帧依次执行现有单变换 Plan”，并非批量 FFT；
2. B=1 时数据量和循环顺序可回到当前 BASE；
3. 只增加功率块，不让 FFT 工作区随 B 线性膨胀；
4. 保留原有两块 FFTW `fftw_malloc` 基址，不引入帧子指针的对齐风险。

第 4 点很重要：`fftw_support.h:67-69` 明确依赖 new-array execute 的尺寸、对齐和 in/out placement 一致。如果把一个大连續 slab 的 `input+b*n_fft`、`fft_out+b*n_bins` 直接交给 plan，小 `n_fft` 甚至一般对齐下都不应未经验证就假定等价；例如仅按元素大小计算，8192 点的相邻复数输出帧相差 65552 字节，并不保持 64 字节对齐。复用原始两块 buffer 可完全绕开这个问题，也不要求 `FFTW_UNALIGNED`。

### 2.2 每 worker 的具体循环

对每个 worker 自己的 `[begin,end)` 独立分块：

```text
for block_begin = begin; block_begin < end;:
    block_frames = min(B, end-block_begin)

    // 阶段一：仍是 B 次独立单帧变换
    for b = 0 .. block_frames-1:
        t = block_begin+b
        pack 当前 t 帧到唯一 input
        plan.apply(input, fft_out)              // n_fft > 1
        power[b][k] = norm(fft_out[k])           // 所有 k

    // 阶段二：复用一条 Mel 权重行，连续处理块内帧
    for m = 0 .. n_mels-1:
        for b = 0 .. block_frames-1:
            acc = 0
            for k = first[m] .. last[m]-1:
                acc += mel_fb[m][k] * power[b][k]
            output[m*n_frames + block_begin+b] = acc

    block_begin += block_frames
```

该布局有三个待验证的局部性目标：

- 对固定 `m`，`mel_fb[m][first:last]` 在块内复用 B 次，而不是每帧都把整个 filterbank 轮换一遍；
- 固定 `m` 时，块内输出地址随 `b` 连续，单条 Mel 行产生 `8*block_frames` 字节的连续逻辑存储；
- pack/FFT/power 与 Mel 分成短阶段，但不跨块保存功率。

同时存在反向因素：当前 BASE 的单个 `power` 在 Mel 前最热，而 B>1 会推迟第 0 帧的 Mel，并在多个 power 行之间跳转。因此“权重和输出更局部”不保证净收益。

### 2.3 数值运算顺序不变

对任一 `(m,t)`，内层 `k` 仍严格从 `first[m]` 到 `last[m]-1` 顺序累加，权重和 power 的计算公式不变；不同 `b`、不同 `m` 之间没有归约重排。每帧仍使用同一个共享只读 plan 的 new-array execute。因而该设计的目标不仅是容差等价，而是保持同一组 double 运算；集成后仍应实际比较输出，不能仅凭循环推断 bitwise 相同。

## 3. scratch 字节公式、cache 与 TLB

### 3.1 推荐布局的精确数据载荷

令 `N=n_fft`，`K=N/2+1`，`fftw_complex` 为两个 double。每 worker：

```text
input  = 8N
fft    = 16K = 8N+16
power  = 8BK = 4BN+8B

S(N,B) = 8N + 16K + 8BK
       = (16+4B)N + 8B + 16   bytes
```

这里包含每 worker 的 input、fft_out、power，不包含必需的 `y/output/window/mel_fb/first/last`、FFT plan 内部分配、线程对象、allocator 元数据或其他全局数据。所有 worker 的总核心 scratch 为 `n_threads*S(N,B)`。

| B | N=8192，字节/ worker | KiB/worker | N=16384，字节/worker | KiB/worker |
|---:|---:|---:|---:|---:|
| 1 | 163,864 | 160.023 | 327,704 | 320.023 |
| 2 | 196,640 | 192.031 | 393,248 | 384.031 |
| 4 | 262,192 | 256.047 | 524,336 | 512.047 |
| 8 | 393,296 | 384.078 | 786,512 | 768.078 |
| 16 | 655,504 | 640.141 | 1,310,864 | 1,280.141 |

B=1 与当前每 worker 三类 scratch 的数据载荷完全相同。16 workers 时，最大档 B=16 的核心 scratch 分别约为 10.00 MiB（8192）和 20.00 MiB（16384），仍须另算全局输出、滤波器与 plan。

作为反面对照，若把 input/fft_out/power 都复制 B 份，公式会变为 `20BN+24B`；B=16 时每 worker 达 2,621,824 字节（8192）或 5,243,264 字节（16384），还引入上述 FFTW 子指针对齐问题。不推荐该布局。

### 3.2 L1/L2 关系

不能在没有目标机实测的情况下把“小于某缓存”直接等同于快，但可先做容量筛选：

- 表中所有档的完整核心 scratch 都明显大于常见 32–48 KiB L1D。即便 B=1，BASE 本来也不是“所有 scratch 都在 L1”。
- FFT 阶段活跃的是固定 input+fft_out，加当前正在写的一行 power。8192/16384 点的 input+fft_out 数据本身约为 128/256 KiB，主要已是 private L2 量级，而不是 L1 量级。
- Mel 阶段新增的关键容量是 B 行 power：B=16 时仅 power 就约 512 KiB（8192）或 1 MiB 多一点（16384）。后者再加固定 FFT scratch 约 1.25014 MiB，已经超过或至少紧贴常见 1.25 MiB private L2 的容量，且还没计 plan、窗口、权重和输出，因此 B=16/16384 是高风险档。
- B=2/4 的总占用较温和，但不能据此预言最快：B 越大，Mel 权重行复用越多，同时首帧 power 等待越久、跨 power 行访问越远。
- 当前 BASE 每帧立即消费 power，可能拥有更好的 L1 命中；新循环主要试图用 B 帧权重复用和连续输出抵消这一损失。

### 3.3 4 KiB 页与 TLB 量级

仅作下限量级估算，假设三类大数组分别按页对齐、没有透明大页，数据页数近似为：

```text
pages(N,B) = ceil(8N/4096) + ceil(16K/4096) + ceil(8BK/4096)
```

实际 allocator 页共享和额外数据会改变精确值。

| B | 8192 核心 scratch 页 | 其中 power 页 | 16384 核心 scratch 页 | 其中 power 页 |
|---:|---:|---:|---:|---:|
| 1 | 42 | 9 | 82 | 17 |
| 2 | 50 | 17 | 98 | 33 |
| 4 | 66 | 33 | 130 | 65 |
| 8 | 98 | 65 | 194 | 129 |
| 16 | 162 | 129 | 322 | 257 |

B 增大显然增加 power 的页工作集，`m→b→k` 还会在 B 个相距 `8K` 字节的区域之间切换，可能加重 DTLB/STLB 压力。另一方面，当前 `t→m` 输出在一帧内触及很多远端 Mel 行，新顺序在一个 `m` 内只推进一个连续输出段，可能减少同时活跃的输出页/写流。两种效应方向相反，必须用端到端与阶段计时判定；不得假设大页或伪造 PMU 结论。

### 3.4 不物化全量功率谱

任何时刻每 worker 只保留最多 B 帧功率，总功率 scratch 为：

```text
n_threads * B * n_bins * sizeof(double)
```

与 `n_frames` 无关，不会形成 `[n_frames,n_bins]` 全量功率谱。输入 `y` 和最终 `output` 本来就是接口要求的完整数组，不计作新增功率物化。块完成后，下一块覆盖同一 power scratch。

## 4. 正确性边界与失败路径

### 4.1 块尾与线程分区

- 保留现有 `begin=floor(n_frames*id/n_threads)`、`end=floor(n_frames*(id+1)/n_threads)`，先分区、再在每个分区内部成块；不能按全局块再把块分给线程，否则容易改变边界或产生重叠。
- 每块长度必须为 `min(B,end-block_begin)`。每个 worker 都可能有不同尾块，不能只对全局 `n_frames%B` 做一次处理。
- `B` 大于某分区的帧数时仅产生一个短块；`begin==end` 时直接无操作。推荐用 `block_begin += block_frames`，避免固定 `block_begin += B` 在极端 int 边界溢出。
- 输出仍写 `output[(size_t)m*n_frames+t]`，每个 `(m,t)` 恰好写一次，worker 间无共享 scratch、无跨分区写。

### 4.2 `n_fft=1`

必须保留现有专门分支：周期 Hann 的 `window[0]` 为 0，当前实现计算：

```text
sample = y[start]*window[0]
power[0] = sample*sample
```

不能改成通用 R2C 后直接取模平方，也不能采用其他库的单点 Hann 特例。块内每个 b 都执行该分支并写 `power[b][0]`；仍可保持现有的 plan 创建行为，但 worker 中不执行 plan。随后统一走 `m→b→k`。

### 4.3 短输入与音频尾部

- 保留在整数除法前检查 `n_samples<n_fft`、非法参数和 `hop_length<1` 的行为；短于一个 FFT 的输入仍安全拒绝，不做零填充。
- `n_samples==n_fft` 时 `n_frames==1`，任意 B 都退化为一个单帧块。
- 保持 `n_frames=1+(n_samples-n_fft)/hop_length`，最后不足 `n_fft` 的音频尾部仍忽略；“音频尾部不足一帧”和“计算块尾不足 B 帧”是两个不同概念。
- 每个合法 t 仍只读取 `y[t*hop_length ... t*hop_length+n_fft)`，不能为了凑满 B 读取越界帧。

### 4.4 分配、plan 与线程创建失败

- 所有 B-power buffer 必须在启动线程前分配完成，保持“worker 不抛异常”的现有约束；索引用 `size_t` 做 `B*n_bins` 等乘法。
- plan 仍在 worker 外串行创建/销毁，worker 只执行；不能因分块给每个 b 创建 plan。
- 保留当前 `std::system_error` 回退：已经成功启动的线程处理自己的分区，`worker(0)` 后，主线程依次执行所有未启动 id 的完整分区，最后 join 已启动线程。块尾逻辑必须只依赖 id 的分区，不能依赖“线程是否并行运行”。
- 分配或 plan 创建失败可继续按现有异常语义向外传播，不必新增运行时降级。若未来要做 B→1 的内存失败重试，必须在 plan 创建和任何线程启动前完整重建布局，不能在 worker 中隐式换 B。
- B=1 是实验回退控制，而不是运行中改变数学路径的 fallback；若 B 扫描无收益，主 Agent 应直接保留 BASE。

## 5. 是否值得独立 B 扫描

### 5.1 值得，但应以 B=1 控制组开始

值得做一次**只改帧循环和 power 块大小**的独立扫描，因为收益/损失由 `n_fft`、`n_mels`、`n_frames`、目标 cache/TLB 和编译器共同决定，静态分析无法给出可靠最优 B。B=1/2/4/8/16 已覆盖从 BASE 到较大 private-L2 压力的范围。

扫描必须满足：

1. B=1 先验证确实回到 BASE 的数据量与运算顺序；若 B=1 自身明显变慢，先查实现，不解释成“分块收益”。
2. 每次只改变 B，固定 R2C ESTIMATE、编译参数、线程分区、权重表示和其他优化；不得与压缩 Mel、线程绑定、NUMA 或 FFT backend 改动混入同一对照。
3. 每个 B 都跑相同正确性集合和四组性能输入，记录失败/回退，而不是只报告最优档。
4. 插桩分别汇总 pack、FFT execute、power、Mel，并同时记录 compute wall 与端到端 wall；worker 时间加总不是实际 wall。
5. 主 Agent 负责编译、正式测试和性能测量；本子任务没有执行任何测试或 Benchmark。

### 5.2 可能回退的原因

- FFT 占 worker 阶段约 71–75%，本改动不减少这部分工作，理论可影响范围有限。
- B>1 破坏当前“power 刚算完立即 Mel”的 L1 局部性，首帧功率等待整个块的 FFT 完成。
- `m→b→k` 在多条 power 行间跳跃；虽然每条行的 k 区间连续，但 B 较大时 cache/TLB 工作集变大，并可能出现 cache set 冲突。
- 更大的 power 首次触页/写回成本增加；总元素量不变，不代表缓存流量不变。
- B<8 时单个 m 的连续输出不足 64 字节，且基址不保证 cache-line 对齐；B=8/16 也不保证整条线在切换 m 后仍驻留。
- 输出矩阵的写分配/RFO 和最终总写流量并未因分块消失；全局输出仍可能远大于末级缓存。
- 权重行复用收益受 `first/last` 稀疏区间长度影响；支持区很窄或 filterbank 本来已热时，收益可能很小。
- 额外 b/块循环可能妨碍编译器优化或增加分支；尾部块还会引入短循环。
- 线程创建回退时主线程串行处理多个分区，更大的 scratch 可能进一步影响其 cache 行为。

### 5.3 可量化验收建议

以下应作为预登记工程门槛，而非预设能达到的速度：

**正确性硬门槛**

- 官方四组全部 PASS，继续满足 `rtol=1e-3, atol=1e-4`；不能用“最大绝对误差很小”替代逐元素容差。
- 随机 PCM、零值/满幅/冲激、精确帧边界、hop 不整除、`n_fft=1/2/8`、n_frames 为 1、n_frames 不能整除 B、线程分区尾块均覆盖。
- 短输入与非法 hop 仍拒绝，无越界；线程创建失败回退路径由主 Agent 用可控故障注入或等价审查验证。
- 因该方案不改变每个输出的 k 累加顺序，可额外记录与 BASE 的逐元素/bitwise 差异；若编译器重排导致非 bitwise，仍必须满足官方容差并说明原因。

**性能与资源门槛**

- 同节点、同二进制构建体系下 B=1 与各 B 做交错重复 A/B，保留每次结果；本地开发至少 5 次，正式口径按项目 `validate.py --runs 3`，不能跨节点混算加速比。
- 建议只有当中位 compute wall 至少下降 3%、且下降大于重复运行噪声，同时四组端到端中位 wall 没有超过 1% 的回退时，才进入正式候选复核。3%/1% 是筛选阈值，不是收益承诺；噪声更大时判为不确定而不是 PASS。
- 分别报告四组的 compute 与端到端 min/median、pack/FFT/power/Mel 阶段变化。若只有 Mel 下降而 compute/端到端不变或回退，不集成。
- 报告实际 B、每 worker 和 16-worker 核心 scratch 字节；可用 GNU time 的环境再记录 RSS/缺页，不可用本地 WSL 数据推断集群 PMU。
- 若最佳 B 随用例变化，先固定一个全局 B 复核；不要按 PCM/expected 内容特化。若必须依参数选择 B，也只能使用 `n_fft/n_mels/n_frames` 等公开参数，并重新验证全部边界。

## 6. 建议结论

优先实现“单份 input/fft_out + B 行 frame-major power”的最小原型，以 B=1 为严格对照，独立扫描 1/2/4/8/16。该设计直接针对当前跨 Mel 行输出和权重轮换，同时避免全量功率谱、B 倍 FFT scratch 及 FFTW 帧子指针对齐风险。

不过，当前单帧 power 的即时消费是强基线，且 FFT 仍占主要 worker 时间；B=8/16 尤其可能受 L2/TLB 压力影响。因此研究结论只能是“值得小范围实测，失败即回退”，不能预先宣称任何 B 有收益。

## 7. 本任务执行记录

- 仅使用只读文件工具阅读项目资料，并写入本目录两个 Markdown 文件。
- 未运行 shell、编译、功能测试、Benchmark、SSH、Slurm 或 Git 命令；因此没有命令输出或实测性能结果可报告。
