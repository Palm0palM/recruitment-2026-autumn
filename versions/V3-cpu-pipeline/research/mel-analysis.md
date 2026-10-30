# V3 子任务 A：Mel 三角滤波器与双前缀和分析

## 0. 范围与证据

本报告只分析 Mel 滤波器构造和每帧 `power -> mel` 投影，不涉及 FFT planner、`plan_many`、Wisdom、Batch FFT 或 Codelet，也不修改接口、CLI、窗口和 FFT backend。

已阅读：

- `src/melspec.cpp`
- `include/melspec.h`
- `scripts/smoke.py`
- `versions/V3-cpu-pipeline/BASELINE.md`
- `versions/V3-cpu-pipeline/README.md`

按补充说明，集群 profile 摘要只采用当前固定 BASE（`b4552f3`，FFTW A2 原生 R2C + ESTIMATE）worktree 中的上述 V3 文档，不依赖其他 worktree。下述历史数据不是本任务的新测量。

本任务未运行 shell、编译、smoke、benchmark 或任何其他测试，无新增命令或测试结果。所有性能判断均为**待实测**。

记号：

- `B = n_bins = n_fft / 2 + 1`
- `M = n_mels`
- 第 `m` 个滤波器实际循环区间为 `[first[m], last[m])`
- `T = sum_m(last[m] - first[m])`，即当前每帧总循环 taps；对正常三角滤波器也就是保留的总非零 taps
- `x_k = k * sr / n_fft`，`p_k = power[k] >= 0`

## 1. 当前稀疏实现的真实特征

### 1.1 构造阶段仍是稠密算法

`build_mel_filterbank`（`src/melspec.cpp:69-105`）：

1. 生成 `B` 个 FFT bin 频率和 `M+2` 个 HTK-mel 等距点对应的 Hz 点。
2. 分配 `M x B` 个 double，并先全部置零。
3. 对每个 `(m,k)` 都计算 `rising`、`falling`、`min/max` 和 Slaney `enorm`，包括支撑区间外的 bin。

所以构造的时间复杂度是 `O(MB)`，稠密权重存储是 `8MB` 字节（这里 `MB` 表示乘积，不是兆字节），另有频率和 Hz 点数组。它不是“直接构造稀疏滤波器”。

随后 `src/melspec.cpp:132-138` 对每一行分别从两端扫描，只跳过**恰好等于 `0.0`** 的权重，得到 `first/last`。没有使用 epsilon，不会丢弃小的正权重；空滤波器最终为 `first == last == B`。

### 1.2 每帧热点是总 taps 为 T 的连续小段点积

`src/melspec.cpp:227-235` 在 power 刚计算完后立即做 Mel：

```cpp
for (int m = 0; m < M; ++m) {
    double acc = 0.0;
    for (int k = first[m]; k < last[m]; ++k)
        acc += mel_fb[m][k] * power[k];
    output[m * n_frames + t] = acc;
}
```

每帧特征：

- 算术为 `T` 次乘加，另有 `M` 次输出存储；空滤波器不进入内层循环，结果严格为 `0.0`。
- 源代码层面每个 tap 请求一个权重 double 和一个 power double，即 `2T` 次 double 元素加载、约 `16T` 字节的加载请求量。**这不是实际 DRAM 流量**：相邻滤波器的 power 会重叠，power 又刚写过，权重也由 workers 只读共享，cache 命中情况取决于工作集和调度。
- 每行被访问的权重区间连续，power 下标也递增，内层循环具备连续访问和自动向量化/FMA 的可能；是否发生及收益均**待实测**。
- 权重行的物理跨度仍是 `B`，虽然只访问 `[first,last)`；稠密分配带来的总占用和构造写入不会因 `first/last` 而消失。
- 外层按 frame、内层按 mel，输出在一帧内跨 `n_frames`  stride 写入；这与两种 Mel 投影算法相同，不是双前缀和的独立收益来源。
- 当前点积按 k 递增逐项累加，权重非负、power 非负，因此有限正常输入下 `acc` 不会因前缀相减而产生负值。

V3 集群 profile 摘要指出：A2 后 worker FFT 仍约占 68–75%，default 的读入、准备和写出合计已占诊断 wall 约 61%；`BASELINE.md` 另记录 default 的 compute 准备约 0.210458 s、compute 全部约 0.565350 s。worker 阶段比例不是全进程比例，setup 是 compute 的子集，不能相加或混算。这些记录说明“仅改 Mel 的整体收益需要先受 Amdahl 约束”，同时使低风险压缩构造值得独立试验；任何实际加速均**待实测**。

## 2. 零阶/一阶矩双前缀和，以及与 FFT backend 的解耦

### 2.1 公式

每帧在 `power[0:B]` 完整生成后，构造长度 `B+1` 的前缀数组：

```text
P[0] = 0,  Q[0] = 0
P[k+1] = P[k] + p_k
Q[k+1] = Q[k] + k * p_k
```

`k` 与频率 `x_k` 只差固定比例 `sr/n_fft`，所以 `Q` 就是一阶频率矩；把比例吸收进线性系数即可，无需另存每帧频率矩。

对任意仿射权重 `w_k = a*k + b`：

```text
sum_{k=l}^{r-1} w_k p_k
  = a * (Q[r] - Q[l]) + b * (P[r] - P[l])
```

第 `m` 个三角滤波器使用：

```text
d = sr / n_fft
enorm = 2 / (f_r - f_l)

上升侧 w_k = aL*k + bL
aL =  enorm*d / (f_c-f_l)
bL = -enorm*f_l / (f_c-f_l)

下降侧 w_k = aR*k + bR
aR = -enorm*d / (f_r-f_c)
bR =  enorm*f_r / (f_r-f_c)
```

令 `c = lower_bound(fft_freqs, f_c)`，并夹到 `[l,r)` 内，即 `c` 是第一个 `x_k >= f_c` 的 bin。中心 bin 只归一个分段；上述选择让上升侧为 `[l,c)`、下降侧为 `[c,r)`，在 `x_k == f_c` 时两侧数学权重都为 `enorm`，不会重复或漏掉中心。

```text
segment(l,r,a,b) = a*(Q[r]-Q[l]) + b*(P[r]-P[l])
mel_m = segment(l,c,aL,bL) + segment(c,r,aR,bR)
```

空段直接贡献零，空滤波器直接写零。每帧前缀构造为 `O(B)`，所有滤波器求值为 `O(M)`，与 `T` 无关。一个普通双侧滤波器可按约 4 次前缀差、4 次系数乘法及若干加法理解；单侧/空滤波器可更少。常数因子和实际访存决定速度，不能只按渐近复杂度判断。

### 2.2 backend 解耦边界

双前缀和只消费长度 `B` 的 double `power`，不消费复数 FFT 输出，也不关心 R2C、半长 C2C 或其他实现。安全解耦方式是把每帧流程保持为：

```text
backend-specific FFT -> backend-independent power[B]
                       -> { sparse dot | prefix-moment projection }
```

- 滤波器几何、`P/Q` 算法及 `power` 的布局不进入 FFT plan 或 FFT 内部 buffer。
- 每线程复用独立的 `P/Q` scratch，长度 `B+1`，避免逐帧分配和数据竞争；只读滤波器系数可共享。
- 保留现有 `n_fft == 1` 分支，先生成语义正确的 `power[0]`，再走统一 Mel 投影。
- 不改变 `compute_rfft` 公共接口、plan 类型、线程划分或 FFT 调用方式。
- 编译期或运行期 A/B 开关只替换 power 后面的 Mel 投影，便于单独归因。

这使同一 Mel 实验可以原样用于当前 A2 ESTIMATE，也可用于其他 backend；但“解耦”不等于性能可移植，仍须分别实测。

## 3. 直接生成压缩三角滤波器的安全建议

建议私有表示：

```cpp
struct CompressedMelFilter {
    int first;                 // 与当前 first 相同
    int last;                  // 半开区间，与当前 last 相同
    std::vector<double> weights; // weights[k-first]
};
```

也可采用统一 flat weights 加 offset，减少每行小分配；这是另一个实现选择，收益**待实测**。

### 3.1 保持公共接口和热点语义

- `include/melspec.h` 已公开稠密 `build_mel_filterbank` 返回 `[M][B]` 的契约。不要让热点压缩优化悄悄破坏它。
- 推荐新增私有 compressed builder 供 `compute_melspectrogram` 使用；公共函数可继续原实现，或由同一安全的逐 bin 权重生成逻辑展开成稠密结果。
- 第一阶段只替换“构造和索引”，内层仍按 `acc += weights[k-first] * power[k]` 计算。这样可独立验证压缩没有改变每帧求和顺序。

### 3.2 候选区间要包含边界，再按原规则裁零

在 `sr > 0` 且 `f_l < f_c < f_r` 的正常定义域内，`fft_freqs` 单调，正支撑只可能位于 `f_l <= x_k <= f_r`。安全构造是：

```text
lo = lower_bound(fft_freqs, f_l)  // 第一个 >= f_l，包含左端点
hi = upper_bound(fft_freqs, f_r)  // 第一个 > f_r 的位置，包含右端点
```

只对 `[lo,hi)` 中的 bin 用**原来的表达式和运算顺序**计算：

```cpp
double rising = (fft_freqs[k] - f_l) / (f_c - f_l);
double falling = (f_r - fft_freqs[k]) / (f_r - f_c);
double val = std::max(0.0, std::min(rising, falling));
val *= enorm;
```

然后仍用 `val == 0.0` 从两端裁零，复制/移动剩余区间到压缩权重。包含端点后再裁剪，可避免 `floor/ceil` 或 epsilon 造成端点 bin 被提前漏掉，并保持以下语义：

- HTK：`2595*log10(1+f/700)` 及其逆变换不变。
- `M+2` 个 mel 点、Hz 点的生成顺序不变。
- Slaney：`enorm = 2/(f_r-f_l)` 不变。
- `max(0,min(rising,falling))` 不变。
- DC、Nyquist、恰落在 `f_l/f_c/f_r` 的 bin、单侧支撑、单 bin 支撑均由同一权重公式决定。
- 不用近似阈值删除小权重，不自行归一化或 renormalize 离散三角形。

正常输入下，候选区间外数学上恰为零，且候选内权重计算与稠密版一致，因此压缩结果展开后应可逐元素与稠密参考比较，而不必先接受一个近似误差。

### 3.3 空、异常和退化情况

- 空滤波器规范化为当前形式的 `first == last == B`、空 weights，点积结果直接为零；不得访问 `weights[0]` 或假定有中心 bin。
- 对非有限或非严格递增的 `f_l/f_c/f_r`，binary-search 候选优化不应擅自定义新语义。若需要覆盖当前未显式拒绝的异常参数，可退回原来的全 B 扫描构造；不要顺便新增 throw、clamp 或修改 CLI 合约。
- `lower_bound/upper_bound` 的单调性依赖正常有限参数；生产实现需保证这一前置条件，而不是对 NaN 做未定义假设。
- 前缀版本也应复用同一份 `[first,last)` 和中心切分元数据，避免 sparse 与 prefix 使用不同边界。

直接压缩把热点权重存储从 `O(MB)` 降为 `O(T+M)`，构造不再写所有区间外零；若每行用 binary search，典型构造为 `O(M log B + T)` 量级（端点和空滤波器会增加少量候选检查）。其 setup 和 cache 收益均**待实测**，但相较双前缀和，它不改变每帧点积的求和顺序，正确性风险明显更低。

## 4. 双前缀和的风险、边界和适用条件

### 4.1 `O(B+M)` 不保证比 `O(T)` 快

粗略地，只有 `T` 相对 `B+M` 足够大时才可能摊薄新增前缀 pass；真实阈值依赖常数，必须实测。需要记录每个 case 的 `B、M、T、非空滤波器数`，不能只看 `n_fft`。

原 sparse 每帧约 `T` 次乘加和 `2T` 个元素加载请求；prefix 则增加：

- 每帧完整扫描 `B` 个 power；
- 每 bin 一次 `k*p_k` 乘法、两次前缀加法和两次前缀数组写入；
- 每线程约 `16*(B+1)` 字节的 double scratch；
- 每滤波器至多 8 个前缀端点 double 加载及若干系数运算，端点访问不再是原来的连续点积；
- 前缀递推存在循环依赖，简单实现不像原内层点积那样直接向量化和使用 FMA。

因此以下情况尤其可能不划算（全部**待实测**）：

- narrow band 或大量空滤波器使 `T << B`，prefix 仍逐帧扫描所有 B；
- `M` 很大而每个滤波器只有很少 taps，固定分段/端点开销占优；
- 正常完整频带三角滤波器常有相当多重叠，`T` 未必远大于 `B`；此时原连续点积可能已经便宜且 cache/向量化友好；
- 帧数较少时，新系数和 scratch 的 setup 不能忽略；
- 多线程下新增 `P/Q` 写入和读取增加 cache 压力，单线程收益不能外推到 16 workers。

反过来，若实际 `T` 很大、支撑宽且帧数多，prefix 才更有试验价值；仍需看端到端而非只看 Mel 插桩。

### 4.2 前缀相减抵消与非负性

数学上：

```text
P[r]-P[l] = sum_{k=l}^{r-1} p_k >= 0
Q[r]-Q[l] = sum_{k=l}^{r-1} k*p_k >= 0
```

但浮点先累加大前缀再相减会丢失低位。高 k、长前缀、宽动态范围 power、峰值集中在区间末端时，`P[l]` 与 `P[r]` 或对应 Q 可能非常接近；Q 还携带大 k 因子。结果可能是：

- 小正值的相对误差显著放大；
- 本应为零或小正数的区间和变成小负数；
- 正负系数分段组合后进一步抵消；
- 最终 Mel 出现小负数，而原逐项非负点积不会以这种方式产生负值。

双前缀和改变了求和结合顺序，**不应要求 bitwise 等于 sparse**。也不要把 `max(0,acc)` 当作默认“修复”：它会掩盖错误，并改变所有负结果而不解决正结果的抵消。

安全试验建议：

1. 先以 double `P/Q` 做独立 A/B，并用高动态范围输入专门检查误差，不只跑随机 PCM。
2. debug/验证版若出现负值或非有限值，用同一 compressed weights 重算直接点积，作为安全 fallback；这样至少非负性可恢复原语义，但 fallback 频率及成本**待实测**。
3. 正数仍可能严重抵消，所以“结果非负”不是充分正确性检查。
4. 若误差超容差，再独立比较长 double 前缀、分块前缀或干脆保留 sparse；这些都会改变成本，不能把 long double 当作无代价方案。
5. 最终是否接受应以 double 中间结果诊断、float32 写出后的官方容差及端到端稳定性共同决定。

### 4.3 必须显式处理的边界

- **空滤波器**：`l == r`，直接输出 `0.0`，不计算分段、不访问无效权重。`smoke.py` 的小 FFT 已会触发 librosa empty-filter 警告，说明这不是理论角落。
- **单侧/单 bin 滤波器**：`c == l` 或 `c == r` 时只算一段；中心 bin 不得重复计入。即使只剩一个 tap，也要由同一仿射段求值并与直接权重对照。
- **中心无 bin**：`lower_bound` 切分仍可工作，不能假定存在权重为 `enorm` 的离散峰值。
- **DC/Nyquist 截断**：`l=0` 或 `r=B` 都合法，前缀数组必须有 `B+1` 项，允许访问 `P[B]/Q[B]`。
- **`n_fft=1`**：`B=1`；当前窗口 `w[0]=0`，hot path 得 `power[0]=0`，所有输出应为零。前缀长度应为 2，不能调用不适用的通用 FFT 路径，也不能采用 scipy 的单点 Hann 特例。
- **`n_fft=2/8` 等低分辨率**：可能大量空或单 bin 滤波器，是边界和性能退化共同测试点。

## 5. 独立 A/B 实验矩阵

建议主 Agent 将构造和投影分开集成、分开测，不一次性替换整块：

| 变体 | 滤波器构造 | 每帧投影 | 隔离的问题 |
|---|---|---|---|
| S0 | 当前稠密 `M x B` + first/last | 当前 sparse dot | 固定 A2 ESTIMATE 对照 |
| S1 | 直接 compressed `(first,last,weights)` | 同一 sparse dot、同一 k 顺序 | 只测压缩构造/布局，无语义重排 |
| P0 | 当前稠密构造 + first/last，另预计算前缀系数 | double `P/Q` | 尽量隔离每帧 prefix 投影，保留相同构造基线 |
| P1 | 直接 compressed 几何/系数 | double `P/Q` | 完整轻量构造 + prefix 组合 |
| P1-safe | 同 P1，保留 compressed weights | prefix；负值/非有限时 direct fallback | 只在正确性需要时测，记录 fallback 次数 |

不建议一开始同时比较多种 scratch 布局、long double、分块前缀和线程改动；若 P0/P1 没有正确且稳定的收益，就停止扩展。

每个变体至少覆盖：

1. 官方四组与 CLI defaults，先单线程定位，再按现有生产线程策略复核；线程数变化只用于诊断，不混作算法收益。
2. 为算法形状增加的本地功能用例：宽支撑/`T` 大、narrow-band/`T` 小、低空滤波器比例、高空滤波器比例、单 bin 和双侧长滤波器。
3. `n_fft = 1,2,8`，以及中心恰有/无 bin、左右端点恰落在 bin、DC/Nyquist 截断。
4. 记录 `B、M、T、T/(B+M)`、非空/单侧/空滤波器数、压缩权重数、prefix fallback 次数。
5. 性能由主 Agent 在获批环境测：process wall 必须包含构造、线程和 I/O；另看 setup/Mel 插桩只用于归因。单线程与多线程结果分别报告，不用单次最小值推断普遍结论。全部性能结论**待实测**。

## 6. 正确性检查清单

### 6.1 compressed-only（S1）应先做强检查

- 将每个 compressed filter 展开为 `[B]`，与当前稠密 builder 逐 bin 对比；正常参数下权重及 `first/last` 应一致，区间外严格为零。
- 特别检查候选端点、空滤波器的 `(B,B)`、单 bin 支撑和最后一个 bin，防止 off-by-one。
- 因 S1 内层仍按相同 k 顺序点积，输出应高度接近甚至保持一致；若出现差异，先查布局/边界，而不是用放宽容差掩盖。

### 6.2 prefix（P0/P1）必须接受“非 bitwise”并做数值压力测试

- 对每个滤波器同时计算 direct double dot 与 prefix double 结果，报告 max abs、max rel（近零只用 abs）、负数次数和非有限次数。
- 输入覆盖零、随机 PCM、满幅正负、冲激、DC、Nyquist，以及人工构造的高动态范围 power：例如前缀前面有巨大能量、目标支撑内只有很小能量，用于主动暴露相减抵消。
- 中心切分与 `[l,r)` 用小型逐 bin 穷举参考核对，确认不漏 bin、不重复中心、不产生错误空段。
- 跑 `scripts/smoke.py` 的官方四组、defaults、`n_fft=1/2/8`；参考仍是 `librosa.feature.melspectrogram(... center=False, htk=True)` 和题目单点窗口特例。
- 最终 float32 输出按现有 `np.allclose(rtol=1e-3, atol=1e-4)` 和 shape 检查通过；另外保留 double 级诊断，因为最终容差通过不代表所有路径都没有抵消。
- 若采用 fallback，必须构造能触发它的测试，并确认 fallback 后与该滤波器 direct dot 一致。

## 7. 给主 Agent 的明确建议

1. **优先实施 S1：直接 compressed 构造 + 原 sparse dot。** 它与 FFT backend 解耦，能消除稠密 `O(MB)` 构造和存储，同时保留 HTK/Slaney、精确零裁剪及每帧求和顺序。实现关键是“候选区间包含端点、仍执行原权重式、再按 `==0.0` 裁剪”，并保持公共稠密接口。setup/总时间收益**待实测**，但这是两项中风险更低、最值得先落地的候选。
2. **双前缀和只作为独立、可撤回的 P0/P1 实验，不建议未经数据直接成为默认。** `O(B+M)` 不保证优于 `T` taps；新增两遍前缀写入、分段端点加载、递推依赖及抵消都可能抵消理论优势。历史 profile 也显示 Mel 并非当前唯一或主要瓶颈，整体收益更需端到端验证。
3. **先测/记录 T 再决定 prefix 是否继续。** 若官方案例普遍 `T` 不够大、空滤波器多或 narrow-band 退化明显，应直接保留 S1。只有当 P0/P1 在全部正确性和压力测试通过、且多案例端到端稳定获益时才考虑集成；不得以单个 Mel timer 或单 case 结果推广。
4. **不要为了 prefix 改 FFT。** 保持 `power[B]` 为唯一边界、现有 R2C A2 ESTIMATE 不变，也不引入 plan_many、Wisdom、Batch 或 Codelet 工作。
