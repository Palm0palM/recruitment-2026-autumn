# V3 MPI Subagent A：全局时间帧划分与 PCM 数据分发

## 0. 范围与结论

本文只研究数据划分和分发，不修改正式源码，也不作性能承诺。依据为固定 BASE 中的：

- `src/melspec.cpp`：`compute_melspectrogram` 不 padding、不 center，局部帧数严格为 `1 + (n_samples-n_fft)/hop_length`；局部输出为行主序 `[n_mels, local_frames]`。
- `src/main.cpp` / `include/melspec.h`：PCM 是 int16 小端 mono，读入后除以 `32768.0` 转为 double；最终输出为无文件头 float32 小端行主序 `[n_mels,n_frames]`。
- `validate.py`：官方四组检查形状和数值容差；普通 CLI 与输出文件语义不能改变。

**推荐划分：先均分全局帧，再由帧区间反推出 PCM 区间。每个非空 rank 的局部 PCM 长度取 `(count-1)*hop+n_fft`，即可不增删帧地复用现有 `compute_melspectrogram`。`count=0` 必须作为独立分支，不读取 PCM、不调用该函数。**

推荐默认边界公式（`P` 为 rank 数）：

```text
t_begin(p) = floor(F*p/P)
t_end(p)   = floor(F*(p+1)/P)
count(p)   = t_end(p)-t_begin(p)
```

它给出连续、不重叠、有序且完整的全局帧区间；每个 rank 的帧数至多相差 1。零帧 rank 集中在末尾，便于输入分发和汇集。

---

## 1. 记号与全局帧定义

令：

- `N`：全局 PCM 样本数；
- `n = n_fft >= 1`；
- `h = hop_length >= 1`；
- `M = n_mels >= 1`；
- `P = MPI communicator size >= 1`；
- 全局 PCM 为 `x[0..N)`，所有样本区间均为左闭右开。

现有实现要求 `N >= n`，否则抛出“参数无效或音频过短”异常。在合法短音频边界上：

```text
F = 1 + floor((N-n)/h),  N >= n
```

全局第 `t` 帧（`0 <= t < F`）恰好使用：

```text
frame_start(t) = t*h
frame_PCM(t)   = x[t*h : t*h+n]
```

最后一个完整帧的结束位置（exclusive）为：

```text
used_end = (F-1)*h+n
R        = N-used_end
```

由 `F` 的下取整定义可得：

```text
0 <= R < h
```

因此全局末尾有 `R` 个样本不足以再组成一帧，必须忽略；不能为了让各 rank 数据等长而 padding 或多读一帧。

### 1.1 `hop > n_fft` 也须保留的语义

常见官方参数为 `n > h`，相邻帧有重叠。但通用公式也允许 `n < h`：此时相邻帧之间有 `h-n` 个未被任何帧使用的样本。它们不是“负重叠”，也不能为了制造连续 PCM 区间而填补或改变帧起点。

---

## 2. 从全局帧区间到局部 PCM 切片

给 rank 分配全局帧区间：

```text
[t_begin,t_end),  0 <= t_begin <= t_end <= F
count = t_end-t_begin
```

### 2.1 非空区间：`count >= 1`

定义：

```text
sample_begin = t_begin*h
L            = (count-1)*h+n
sample_end   = sample_begin+L
             = (t_end-1)*h+n
y_local      = x[sample_begin:sample_end]
```

所以：

- **全局起始样本**：`sample_begin = t_begin*h`；
- **全局末样本（inclusive）**：`sample_end-1 = (t_end-1)*h+n-1`；
- **局部 PCM 样本数**：`L = (count-1)*h+n`；
- 局部第 `u` 帧（`0 <= u < count`）从 `y_local[u*h]` 开始，对应全局第 `t_begin+u` 帧，二者访问的是完全相同的 `n` 个 double 样本。

### 2.2 恰好产生 `count` 帧的证明

因为 `count >= 1`、`h >= 1`，有 `L >= n`，满足现有函数的长度检查。代入其帧数公式：

```text
local_F = 1 + floor((L-n)/h)
        = 1 + floor(((count-1)*h+n-n)/h)
        = 1 + (count-1)
        = count
```

整除是精确的，不依赖任何凑整。并且：

1. 第 0 帧从局部样本 0 开始，覆盖 `[0,n)`；
2. 第 `count-1` 帧从 `(count-1)*h` 开始，结束恰好是 `L`；
3. 若再多一帧，其结束位置为 `count*h+n = L+h > L`，故不可能多出一帧。

因此无需 padding，也无需在调用后裁掉额外帧。现有函数每帧独立加窗、FFT、power 和 mel 累加，局部计算不会改变帧索引对应的 PCM 内容。

### 2.3 小例子

取 `n=8,h=4`，分配 `[t_begin,t_end)=[2,5)`，则 `count=3`：

```text
sample_begin = 2*4 = 8
L            = (3-1)*4+8 = 16
sample_end   = 24
局部帧        = 全局帧 2、3、4
全局 PCM      = [8,16)、[12,20)、[16,24)
```

最后一个使用样本是全局 23，局部切片没有多余样本。

---

## 3. 相邻切片的重叠、空洞与总量

对两个相邻非空区间，前者的 `t_end` 等于后者的 `t_begin`：

```text
prev.sample_end   = (t_end-1)*h+n
next.sample_begin = t_end*h
差值               = prev.sample_end-next.sample_begin = n-h
```

因此：

- `n > h`：相邻 rank 重叠 `n-h` 个样本；重叠区间为
  `[next.sample_begin,prev.sample_end)`；
- `n == h`：切片首尾相接，重叠为 0；
- `n < h`：两切片之间有 `h-n` 个样本空洞，这些样本本来就不会被完整帧使用。

每个非空 rank 的 `count*h` 个“按帧推进”样本之外，还需：

```text
L-count*h = n-h
```

在通常的 `n>=h` 情况下，这就是该 rank 的右侧 halo；**不是左右各一个 halo**。首帧自身已经覆盖 `n` 个样本。

令 `K=min(P,F)` 为非空 rank 数（采用上述连续均分时），则所有局部切片长度之和为：

```text
S = sum_p L_p = F*h + K*(n-h)
```

当 `n>=h` 时，也可写成全局有效样本加跨 rank 重复：

```text
S = used_end + (K-1)*(n-h)
```

总跨边界重复量为 `(K-1)*(n-h)` 个样本。对于 int16 文件读取，每样本 2 字节；对于 double 的 MPI 消息，每样本 8 字节。重叠很大而每 rank 帧数很少时，halo 成本不可忽略。

---

## 4. 非整除帧数与零帧 rank

### 4.1 两种等价的均衡计数方式

设：

```text
q = floor(F/P)
r = F mod P
```

可给前 `r` 个 rank 分配 `q+1` 帧，其余分配 `q` 帧：

```text
t_begin(p) = p*q+min(p,r)
t_end(p)   = (p+1)*q+min(p+1,r)
count(p)   = q + (p<r ? 1 : 0)
```

也可使用推荐的 floor-boundary 形式：

```text
t_begin(p) = floor(F*p/P)
t_end(p)   = floor(F*(p+1)/P)
```

二者计数均为 `q` 或 `q+1`，差别只是多出来的一帧落在哪个 rank。正式实现只能固定一种，避免不同 rank 对边界理解不一致。

例：

```text
F=10,P=3：count = 4,3,3（q/r 形式），区间为 [0,4)、[4,7)、[7,10)
F=2,P=4： count = 1,1,0,0
```

### 4.2 `count=0` 是空任务，不是短 PCM 任务

当 `P>F` 时必有空 rank。对空 rank：

- 只设置 `count=0`、局部输出为空；
- 不计算或解释 `sample_begin/sample_end/L`；
- 不读取、不接收任何 PCM；
- 不调用 `compute_melspectrogram`；
- 仍须参与后续必要的 MPI 集合通信或汇合协议。

尤其不能直接套用 `L=(count-1)*h+n`：`count=0` 会得到 `n-h`，在 `n>=h` 时可能被误当成非空切片，且现有函数对非空短切片至少会产生一帧或抛错，均不等于零帧。

### 4.3 极短音频

- `N<n`：没有合法帧，不能把它表示成“所有 rank 都 count=0 并成功退出”；应保持现有失败语义，并确保所有 rank 一致退出、无集合通信死锁。
- `N=n`：`F=1`，只有 rank 0 获得一帧，其 `L=n`，其余 rank 为空。
- `n <= N < n+h`：仍为 `F=1`；尾部不足 `h` 的样本忽略。
- `N=n+h`：`F=2`。
- `N=0`：属于 `N<n` 的失败路径。

---

## 5. 三种 PCM 输入分发方式比较

以下只列数据量和结构性成本。实际快慢必须由主 Agent 在同节点、同 rank 映射下测量；不能把通信量较少直接等同于端到端更快。三种方式的最终输出写出都仍是 root 写 `4*M*F` 字节，输出汇集成本另计。

记 rank 0 的非空局部长度为 `L_0`；若 rank 0 为空则 `L_0=0`。`S=sum L_p`。

| 方案 | 输入 I/O / 显式通信 | PCM 内存 | 优点 | 主要代价与风险 |
|---|---|---|---|---|
| root 广播完整 `y` | root 从文件读 `2*N` 字节并转换；逻辑上向其他 rank 广播约 `8*N*(P-1)` 字节的 double 数据 | **每个 rank 都持有 `8*N` 字节完整 y**，另加局部输出 | 控制简单；所有 rank 用统一切片公式；无需共享文件随机读 | 最大输入通信和内存；发送大量本 rank 不用的数据；root 先读完整文件形成串行前置阶段 |
| root `Scatterv` 带重叠局部切片 | root 读 `2*N` 字节；非 root 接收总量 `8*(S-L_0)` 字节（不含 rank 0 自拷贝） | root 持有完整 `8*N`；非 root 仅 `8*L_p` | 非 root 内存最小；只传有效帧及必要 halo；输入通信随实际工作量增长 | root 仍是读文件和发数据的集中点；变长 counts/displs；halo 重复；不是等长普通 `Scatter` |
| 共享文件系统各 rank 局部读取 | 各 rank seek 到 `2*sample_begin`，读 `2*L_p`；全 job 文件读取请求总量 `2*S` 字节（缓存可能改变实际物理 I/O）；无显式 MPI PCM 消息 | 每 rank 仅保存局部 int16 缓冲及 double `y_local`，可边转换边释放 int16 | root 无完整 y 输入内存和广播前置阶段；各 rank 只读所需数据 | 依赖共享 FS 可见性与一致性；大量小读/seek 和元数据竞争；缓存使 I/O 估算不等于实测；每 rank 须先得到一致的 `N` 和参数 |

### 5.1 负载均衡

三者使用同一帧区间时，帧计算量都至多相差一帧，比按 PCM 样本数直接均分更正确：按样本均分会切断帧、产生重复或遗漏，且在 `n!=h` 时各 rank 的有效帧数不易控制。

但“帧数均衡”不等于端到端均衡：

- 每 rank 都重复 window、mel filterbank、FFTW plan 等 setup；
- 现有函数局部线程数为 `min(16, hardware_concurrency, max(1,count/256))`，所以 `count<256` 的 rank 只用单线程；
- 空 rank 没有 DSP 工作，但仍需正确参与通信；
- root 通常还承担输入和/或输出，可能成为瓶颈；
- `n-h` 很大时，小任务的输入 halo 占比很高。

### 5.2 实现建议

- spike 若以最少新文件 I/O 代码为先，可优先评估 root `Scatterv`；它避免每 rank 完整 y，同时不必依赖共享 FS 的扩展属性。
- `MPI_Scatterv` 可从 root 的完整 double `y` 以 `sample_begin` 为 displacement、`L` 为 sendcount 发送，无需先构造等长 packed 块；空 rank 的 count 必须为 0，并使用安全 displacement（例如 0），不要把无意义的帧坐标转为越界指针。
- 若 MPI 实现/封装要求独立 send buffer，则打包会额外增加 root 的 `8*S` 临时内存和复制，应计入方案，不可忽略。
- 共享 FS 方案中，int16 到 double 的转换必须保持 `static_cast<double>(int16_value)/32768.0`，避免不同分发路径产生输入差异。
- 所有 `N`、帧索引、样本索引和乘法先以至少 64 位无符号/有符号安全类型计算，再检查是否满足现有 `int` 接口与 MPI `int` count/displacement 限制。直接 `(count-1)*h+n` 的 int 运算可能溢出。

---

## 6. 输出汇集是不可省略的布局问题

局部结果是：

```text
local[m*count+u],  0<=m<M, 0<=u<count
```

它对应全局：

```text
global[m*F + (t_begin+u)]
```

即每个 rank 在每个 mel 行中贡献连续区间 `[t_begin,t_end)`，但**整个局部 row-major 缓冲区并不是全局 row-major 缓冲区的一段连续区间**。因此不能直接把所有 rank 的局部 buffer 用一次普通拼接式 `Gatherv` 得到最终结果。

可行方案：

1. root 接收各 rank 的局部 row-major double buffer，再按上式逐 mel/逐帧放置；
2. 每个 mel 行分别做 `Gatherv`，counts 为 `count_p`、displacements 为 `t_begin_p`，共 `M` 次集合通信；
3. 局部先转置为 frame-major，再一次汇集，root 转成最终 mel-major；须显式计入转置内存和成本。

最终 root 输出长度必须是 `M*F`，顺序固定，最后写 float32 小端、无文件头。若考虑先转 float 再汇集以减少通信，应由主 Agent 单独验证数值和集成语义；本文公式按现有 double 局部输出描述。

---

## 7. 集成风险清单

1. **空 rank 误调用 DSP**：`count=0` 若调用现有函数，无法表示零帧；必须绕过。
2. **短音频错误语义改变**：`N<n` 不能静默输出空文件；root 检测后须让其他 rank 同步知悉失败，避免有的 rank 等待 `Bcast/Scatterv/Gatherv`。
3. **把帧区间当成样本区间**：不同 `n,h` 下会丢帧或增加帧。
4. **漏掉右 halo**：只给 `count*h` 样本通常不足，除非特殊的 `n=h`；正确长度始终是 `(count-1)*h+n`。
5. **多算左 halo**：局部第 0 帧必须从全局 `t_begin*h` 开始，不能向前偏移，否则所有帧错位。
6. **`hop>n_fft` 被错误禁止或填补空洞**：现有全局算法允许这种参数，切片之间可不连续。
7. **尾部 padding**：局部和全局都只能保留完整帧；末尾 `R` 个样本忽略。
8. **输出直接拼接**：会破坏 `[M,F]` 行主序；必须使用 `m*F+t_begin+u` 映射或逐行汇集。
9. **整数溢出与 MPI 窄类型**：`F*p`、帧到样本转换、`M*F`、字节偏移和 MPI counts/displs 都须检查；文件 seek 的字节偏移为 `2*sample_begin`。
10. **重复 setup 与局部线程缩放**：每 rank 的 plan/filterbank setup 和 `count/256` 线程策略可能削弱多节点收益，必须纳入端到端测量。
11. **root 集中 I/O 与输出内存**：root 可能同时持有完整 y、全局 double 输出、float 输出副本和通信临时区；比较方案时不能只报非 root 内存。
12. **共享 FS 假设**：需确认所有计算节点看到同一路径、同一文件及一致长度；不能假定本地缓存或吞吐。
13. **集合通信形状不一致**：所有 rank 必须基于同一 `N,F,P,n,h` 得出同一 counts/displacements；建议 root 计算并广播元数据，或所有 rank 用同一 64 位纯函数计算后交叉检查。
14. **FFTW 初始化并发/线程安全**：属于主 Agent 集成审查项；数据划分本身不证明多进程内多线程 plan 初始化安全。
15. **计时口径**：输入读取、转换、通信、setup、同步、输出转置/汇集和写出均须计入，不能只计局部 DSP。

---

## 8. 建议测试清单（供主 Agent 执行）

本子任务未编译、未运行任何测试。建议先测独立划分纯函数，再跑随机 PCM/smoke，最后跑正式四组 `validate.py`。

### 8.1 划分不变量与边界

- [ ] 对 `N>=n`、`P>=1`，所有 `[t_begin,t_end)` 连续、有序、互不重叠，并集恰为 `[0,F)`。
- [ ] `sum(count)=F`，且任意两 rank 的 `count` 差不超过 1。
- [ ] 每个非空 rank 均满足 `sample_begin=t_begin*h`、`sample_end=(t_end-1)*h+n`、`L>=n`。
- [ ] 用独立实现计算 `1+(L-n)/h`，逐一断言等于 `count`。
- [ ] 所有非空局部切片均在全局有效范围内：`sample_end<=used_end<=N`。
- [ ] 相邻非空 rank：`n>h` 时重叠恰为 `n-h`；`n=h` 时为 0；`n<h` 时空洞恰为 `h-n`。
- [ ] 全局尾部：`R=N-((F-1)*h+n)` 且 `0<=R<h`；分别覆盖 `R=0` 和 `R=h-1`。
- [ ] `F%P==0` 与 `F%P!=0`；`P=1`、`P=2`、非 2 的幂 `P=3/6`。
- [ ] `F<P`，检查只有前 `F` 个 rank 非空、其余为空；空 rank 不读、不调用 DSP，输出计数为 0。
- [ ] `N=n`、`N=n+h-1`、`N=n+h`，分别验证 `F=1,1,2`。
- [ ] `N=0`、`N=n-1` 保持失败语义，所有 rank 一致退出且无死锁。
- [ ] `n=h`、`n>h`、`n<h`、`n=1`；至少一组 `h=1` 和一组大 `n-h`。
- [ ] 非整除例子 `F=10,P=3`，以及极小例子 `F=2,P=4`，逐 rank 断言区间与长度。

### 8.2 数据内容与端到端正确性

- [ ] 用带可识别样本序号的随机/递增 int16 PCM，检查局部第一帧和最后一帧与全局对应帧逐样本一致，尤其是首尾 rank 和 halo 两侧。
- [ ] 对每种输入方案（Bcast、Scatterv、共享 FS 局部读）使用相同帧计划，局部输出形状均应为 `[M,count]`。
- [ ] `P=1` 与普通单进程 BASE 比较完整输出；预期帧划分本身不引入任何数值变化。
- [ ] 多 rank 汇集后检查 `global[m*F+t_begin+u]==local[m*count+u]`，防止 mel 行错位。
- [ ] 覆盖 count 小于 256、等于 256、跨多个 256 的局部任务，检查线程数变化不影响帧归属。
- [ ] 随机 PCM、多组 `n/h/M/P` 先与单进程结果比较；再运行项目 `scripts/smoke.py`。
- [ ] 最后运行官方四组：default、small_mels、large_fft、narrow_band；全部满足形状及 `rtol=1e-3,atol=1e-4`，任何 rank 失败都判失败。
- [ ] 模拟/检查任一 rank 读文件、分配或 DSP 失败时的错误传播，不留下其他 rank 阻塞。

### 8.3 集成性能记录（不是本子任务的实测结论）

- [ ] 分阶段记录 root 输入、PCM 通信/局部读取、局部 setup、局部 DSP、输出汇集、float 转换和写出。
- [ ] 记录节点数、rank/CPU 映射、每 rank count/L、总 halo 样本、root 峰值内存及输出临时副本。
- [ ] 与同节点单进程基线及可比多节点配置比较，包含全部通信和同步；不只报告 DSP 时间。

---

## 9. 可直接交给主 Agent 的划分伪代码

以下仅是无副作用伪代码，不是已编译原型。正式实现须加入参数检查、64 位溢出检查和 MPI 类型范围检查。

```cpp
struct Partition {
    int64_t t_begin = 0;
    int64_t t_end = 0;
    int64_t count = 0;
    int64_t sample_begin = 0;  // 仅 count>0 有意义
    int64_t sample_end = 0;    // exclusive；仅 count>0 有意义
    int64_t local_samples = 0; // 仅 count>0 有意义
};

Partition make_partition(int64_t F, int64_t n, int64_t h,
                         int64_t P, int64_t rank) {
    Partition d;
    d.t_begin = F * rank / P;          // 实际代码须防乘法溢出
    d.t_end   = F * (rank + 1) / P;
    d.count   = d.t_end - d.t_begin;
    if (d.count > 0) {
        d.sample_begin  = d.t_begin * h;
        d.local_samples = (d.count - 1) * h + n;
        d.sample_end    = d.sample_begin + d.local_samples;
    }
    return d;
}

// 非空 rank：
// y_local = global_y[sample_begin : sample_end]
// compute_melspectrogram(y_local, ...) -> [M,count]
// global_output[m*F + t_begin+u] = local_output[m*count+u]

// 空 rank：
// local PCM 为空、local_output 为空；绝不调用 compute_melspectrogram。
```

核心验收条件可概括为：

```text
count>0  => local_samples=(count-1)*h+n 且现有函数恰好输出 count 帧
count=0  => 无 PCM、无 DSP、空输出
所有 rank => 全局帧不重不漏，输出按 m*F+t_begin+u 汇集
```
