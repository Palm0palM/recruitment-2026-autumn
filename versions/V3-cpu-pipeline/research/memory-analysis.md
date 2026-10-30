# V3 CPU-Pipeline 子任务 C：内存与线程静态分析

## 范围与证据

- 固定基线为当前 worktree 的 FFTW A2 ESTIMATE；本文分析 `src/main.cpp`、`src/melspec.cpp`、`include/fftw_support.h`、`scripts/fftw_profile.py`，并以本 worktree 的 `versions/V3-cpu-pipeline/BASELINE.md` 与 `README.md` 作为集群 profile 摘要。
- `src/main.cpp` 是受保护文件。本文不提出修改它、CLI、输出布局或 double 计算语义的方案。
- 原指定的 `versions/V2-fftw/PROFILE.md` 在当前固定 BASE 的 worktree 中不存在；本文不引用其内容，也不读取其他 worktree。
- 本文是静态分析，没有运行 shell、构建、功能测试、Benchmark、SSH、Slurm 或 Git，也没有产生新的测量数据或性能收益结论。

记：

- `N`：PCM 样本数；`F = n_frames`；`M = n_mels`；`B = n_fft/2+1`；`O = M*F`。
- `T`：实际选择的 worker 数。
- double 输出为 `8O` 字节，最终 float32 文件为 `4O` 字节。

## 1. 成本拆分

### 1.1 `read_pcm`：读入与转换

`src/main.cpp:28-44` 的现有 `read_s` 是一个聚合值，包含：

1. 打开文件、两次 seek 和获取长度；
2. `std::vector<int16_t> buf(n_samples)`：约 `2N` 字节分配及 value-initialization，随后被文件内容覆盖；
3. `f.read`：实际文件读取及 `buf` 数据页写入；
4. `y.resize(N)`：约 `8N` 字节分配、double 零初始化和首次触页；
5. 逐样本读取 int16、转换为 double、除以 32768 并写入 `y`。

因此不能把 `read_s` 全部称为磁盘 I/O。转换期间同时持有约 `2N + 8N` 字节的样本数据；`buf` 在函数返回后释放，不进入 compute 的常驻内存。

### 1.2 `compute_melspectrogram` 准备阶段

当前 `compute_setup_s` 从 `src/melspec.cpp:124` 开始，到 `:172` 结束，聚合了多个成本完全不同的步骤：

1. 周期 Hann 窗分配及 `n_fft` 次 `cos` 计算（`:127-130`）；
2. `build_mel_filterbank`：FFT 频率、`n_mels+2` 个 mel/hz 点，以及 `M*B` 个 double 权重的分配、零初始化和计算（`:69-105`、`:132`）；
3. `first/last` 两个 `M` 元素数组及对每行权重首尾零值的扫描（`:134-138`）；
4. `output.resize(O)` 的分配、零初始化/首次触页（`:149`）；
5. 查询并发度、计算 `T`，以及线程对象的准备；
6. 每线程 FFTW 输入/输出 scratch 的 `fftw_malloc`（`:156-169`）；
7. R2C plan 创建（已有嵌套的 `r2c_plan_s`）；
8. `power_buffers` 的 `T` 个独立 `B` 元素 double 数组的分配和零初始化（`:170-171`）。

主要存储量可近似写为：

- window：`8*n_fft`；
- mel 权重：`8*M*B`，外加 `M` 个 inner vector 的对象/分配开销；
- first/last：合计约 `8M` 字节；
- output：`8O`；
- 每线程 R2C 输入：`8*n_fft`，FFT 输出：`16*B`，power：`8*B`；
- plan 内部存储不能从这段源码推导。

**现有 `compute_setup_s` 不能证明 output resize 是准备阶段的主导项。** V3 `BASELINE.md` 记录的 hepnode2 集群 A2 诊断作业 59 中，default 的 read 为 `0.180045 s`、setup 为 `0.210458 s`、compute 全部为 `0.565350 s`、write 为 `0.222535 s`；setup 是 compute 的子集，不能再与 compute 相加。V3 `README.md` 同时指出 worker FFT 约占 worker 阶段的 68–75%，而 default 的读入/准备/写出约占诊断 wall 的 61%，两种口径不可混用。这些摘要说明准备和 I/O 值得拆分，但仍不能据此分配 mel、resize、page fault 或 scratch 各占多少，也不能把诊断 wall 当作正式成绩。

### 1.3 worker 阶段

每个 worker 处理连续帧区间：

```text
begin = floor(F*id/T)
end   = floor(F*(id+1)/T)
```

分区覆盖所有帧且各区间大小差不超过 1。`worker(0)` 在调用 `compute_melspectrogram` 的主线程上执行，其余 `T-1` 个为 `std::thread`。

R2C 热路径每帧的现有四类计时为：

| 阶段 | 实际工作 | 主要内存行为 |
|---|---|---|
| `pack_s` | 读取重叠的 PCM 帧、乘 Hann 窗、写输入 scratch | 每帧读 `n_fft` 个 y/window 元素，写 `8*n_fft` scratch |
| `fft_s` | 共享 plan 的 new-array R2C execute | 读写本线程独立输入/输出 scratch |
| `power_s` | 对 `B` 个复数 bin 求模平方 | 读 FFT 输出，写 power |
| `mel_s` | 仅遍历每个滤波器的非零支撑并累加 | 读共享 mel/power；向 output 的 `M` 个行写第 `t` 列 |

`mel_s` 已包含 output store；filterbank 构建不在 worker 的 `mel_s` 内。output 的逻辑布局是 `[M,F]`，所以固定 `t`、遍历 `m` 时写地址 stride 为 `F*sizeof(double)`，并非一次连续写出整帧。各 worker 的 `t` 区间连续，但在每个 mel 行上，两个相邻分区边界可能共享 cache line；这属于有限的边界 false sharing，不足以据此改变受约束的输出布局。

现有脚本的 `thread_wall_sums` 是**所有线程的线程秒之和**，不是进程 wall time，也不是各阶段占总 wall 的百分比。它还遗漏线程启动/退出、调度偏斜、负载不均和未落入四段计时的代码。若创建线程失败，当前实现会保留已启动线程，并由主线程顺序补做未启动分区；正式并发度应以实际 `launched` 和日志为准，而非只看请求的 `T`。

### 1.4 `write_output`：转换与写出

`src/main.cpp:46-59` 的 `write_s` 聚合了：

1. `std::vector<float> out_f32(O)`：约 `4O` 字节分配、零初始化/首次触页；
2. 顺序读取全部 `8O` 字节的 double output，并逐元素转换、写入全部 float 缓冲；
3. `f.write` 写出 `4O` 字节以及流状态检查；
4. 函数返回时的缓冲释放和文件流析构。

因此 `write_s` 也不能全部归为存储设备写入。compute 返回后，window、mel、scratch 和 plan 已销毁；转换/写出阶段主要同时保留 `y`、double output 和 float 输出副本。main 对所有 double 输出元素的读取发生在第 2 步，这一事实直接影响下一节的合法性判断。

## 2. `output.resize`：初始化、首次触页与合法替代

### 2.1 当前行为

main 中 `output` 初始为空，`resize(O)` 必须令 vector 的 `size()==O` 并建立 `O` 个可合法访问的 double 元素。对当前 C++17 默认 allocator，新增 double 是 value-initialized，即零初始化；不能把它理解成仅取得一段未初始化容量。

实现通常会把零初始化优化成批量清零，但语言语义并没有“这些元素马上会被 worker 覆盖，所以可以省略构造”的例外。物理成本分两类：

- allocator 取得虚拟地址/管理堆元数据；
- 清零写入触发物理页分配、缺页、清零及内存写入，即首次触页成本。

新 mmap 页、已复用堆页和不同 OS 内存策略下，缺页与缓存表现会不同，不能只用 `resize` 墙钟时间推断某一项硬件成本。

正常路径中 worker 确实会对每个 output 元素写一次，随后 main 又读取全部元素。但“之后全部覆盖”并不能 retroactively 改变 `std::vector<double>::resize` 的语义。

### 2.2 为什么不能用 UB 绕过

以下做法都不合法，不能作为优化：

- 仅 `reserve(O)`，然后通过 `data()[i]` 写 `[0,O)`：容量不是大小，尚未存在的 vector 元素不能这样访问，main 看到的 size 也不正确。
- 私改 size、伪造 vector 内部指针、用类型转换接管 raw allocation：破坏 vector 所有权和对象表示，且不可移植。
- 依赖“malloc 返回的内存上已经有 double 对象”或在没有正确建立元素生命周期时读写：对象生命周期/vector 不变量不成立。
- 让 main 读取未初始化 double：C++17 下读取 indeterminate double 不是可接受的值传播路径，可能导致 UB 和错误输出。
- 给默认 `std::vector<double>` 偷换一个跳过 `construct` 的 allocator：`std::vector<double>&` 已固定为默认 allocator 类型；不同 allocator 的 vector 是不同类型，不能绑定到该接口。

需要精确区分：**未初始化的 raw double 存储并非一创建就自动产生 UB**；若能用另一种合法 owning-storage 类型建立对象，并保证每个元素在任何读取前都被赋值，可以设计出合法的未初始化缓冲方案。但它不能直接满足当前 `std::vector<double>&` 接口，也不能让受保护的 main 继续以当前方式接收它。它不能成为本文建议的实施路线。

### 2.3 合法替代是否真的减少成本

| 方案 | 合法性/接口 | 成本判断 |
|---|---|---|
| 保持当前一次性 `resize(O)` | 合法，已满足所有约束 | 只有一次分配和一次必要的元素初始化；没有 per-frame 增长问题 |
| `reserve(O)` 后再 `resize(O)` | 合法 | 对空 vector 不减少清零或首次触页，通常只是多余调用 |
| `assign(O, 0.0)`、`vector<double>(O)` | 合法 | 同样构造并初始化 O 个 double，没有原理上的节省 |
| resize 后再 `memset`/预触页 | 合法但冗余 | 增加一遍内存写；预触页本身不能替代 resize 已完成的初始化 |
| 并行清零或并行预触页 | 可合法实现 | 不减少总写入量，增加线程/同步成本；只能在 NUMA A/B 中验证，不能宣称必然获益 |
| 静态缓冲或跨调用复用容量 | 某些设计可合法，但改变生命周期/重入性 | 当前 CLI 每次进程只调用一次 compute，没有复用收益；不应引入 |
| 先建 raw storage，再复制/移动到 vector | 可合法 | 仍需构造 O 个 vector 元素，反而增加复制/存储；不能跳过成本 |
| 改用自定义 vector、raw owner 或 float 输出 | 不满足当前接口、main 或 double 语义约束 | 不实施 |

**结论：在保持 `std::vector<double>&`、受保护 main 和全部元素均被 main 读取的条件下，不应也不能声称可合法跳过 `resize` 初始化。** 当前已经是一次性分配，继续围绕 output 清零做“跳过初始化”的优化没有合法空间。合理工作是用独立插桩确认它是否真有显著成本，再评估准备阶段其他合法项目或 NUMA 放置；即使 resize 很贵，也不等于存在符合约束的消除办法。

## 3. Worker 缓冲、对齐、first-touch 与 NUMA

### 3.1 当前分配和对齐

- 所有输入/FFT 缓冲都在启动线程前分配，保留“worker 内不因分配失败而抛异常”的重要性质。
- `Buffer<T>` 使用 `fftw_malloc/fftw_free`，为 FFTW SIMD 和 new-array execute 提供所需对齐。所有 worker 使用相同大小、相同 in/out-of-place 形式和相同分配方式，符合共享 plan 对数组布局/对齐一致性的要求。
- 不应把 FFTW 输入/输出改成普通 `new`、vector data 或各自不同的对齐方式后仍假定共享 plan 合法。
- power 不使用 FFTW，普通 vector 的 double 对齐在功能上足够；没有证据支持仅为它增加自定义对齐或 padding。
- 各线程的 input、fft、power 都是独立数据缓冲，不存在多个 worker 同时写同一 scratch 的逻辑竞争。

### 3.2 首次触页和可能的 NUMA 位置

在常见 Linux first-touch 策略下，可以形成以下**待验证假设**，而不是既定事实：

| 数据 | 分配/初始化者 | 首次实质数据写入者 | NUMA 含义 |
|---|---|---|---|
| `y` | main 读入时 resize/转换 | main | 多半靠近 main 所在节点；worker 只读共享 |
| window、mel、first/last | main 准备 | main | 多半靠近 main；所有 worker 共享读取 |
| double output | main 在 resize 中清零 | main | 物理页可能在 worker 启动前已放到 main 节点 |
| FFTW input/output | main 调用 `fftw_malloc`，ESTIMATE 下不依赖 planner 填数据 | 各 worker 首帧 pack/FFT | 数据页有机会落在实际运行 worker 的本地节点 |
| power | main 构造并零初始化所有 inner vector | main | 虽由 worker 后续重写，但首次触页已发生在 main |

allocator 复用已驻留页、系统 NUMA balancing、内存策略、线程迁移以及非 Linux 平台都会改变结果。仅看分配发生在主线程，不能等同于物理页一定在主线程节点；反之，`fftw_malloc` 也不等于已经本地化物理页。

当前 output 的串行清零与 worker 后续写入存在 NUMA 目标冲突：worker 希望各自本地写，main 的最终 float 转换又需要顺序读全部 output。把 output 额外并行预触页只会增加成本，且无法保证操作系统之后不迁移页。window/mel/y 的 per-worker 复制会增加准备时间和内存，也不应在没有测量的情况下实施。

### 3.3 可评估但不能预设收益的方向

1. 先保留现有预分配和 FFTW 对齐，不碰共享 plan 约束。
2. 用诊断构建区分“虚拟分配”“power 零初始化”和“worker 首次使用”，并结合线程绑核 A/B 判断 NUMA 是否真有影响。
3. power 与 output 不同：正常 R2C 每帧会先写满所有 `power[0..B)`，单点 FFT 分支也先写 `power[0]`。理论上可用合法拥有未初始化 double 数组的 owner（例如正确的 `new double[B]` owner）在启动线程前分配，再由 worker 完整写入后读取，从而避免 power 的零初始化并让其首次触页发生在 worker。这个方案必须逐分支证明写后读、保持异常安全并单独验证；它只适用于 scratch，**不能外推为 output vector 可跳过初始化**。预计收益也未建立，优先级低于插桩。
4. 不建议为 output 改布局/加 padding、复制整套共享 mel/window、加入正式预触页 pass，或无测量地更换 huge page/NUMA 内存策略。

## 4. 8/12/16 线程及可选绑核 A/B 设计

### 4.1 当前并发度不是固定 16

当前实际选择为：

```text
T = min(16,
        max(1, hardware_concurrency()),
        max(1, F/256))
```

因此：

- 短输入会主动减少线程；小用例中 8/12/16 三组可能全部落到同一个 T，不能把它们当成有效线程数对照。
- `hardware_concurrency()` 是实现相关的系统提示，不保证反映 Slurm cpuset 中当前进程可用的 CPU 数；可能报告更大的在线 CPU 数。
- Slurm 的 cpuset/cgroup 会限制实际可运行 CPU，但不会自动把 `std::thread` 数量改成配额，也不保证每个 worker 固定在一个核心上。
- 每个 FFTW plan 是串行执行后端，只有外层 `std::thread` 并行。

`OMP_NUM_THREADS`、`OMP_PROC_BIND`、`OMP_PLACES` 等是 OpenMP runtime 的控制变量。当前没有 OpenMP worker runtime，**它们不会直接控制或绑定 `std::thread`**；也不能靠这些变量完成本实验的线程数/亲和性切换。Slurm 自身环境变量同样不能代替代码中的实际 affinity 检查和设置。

### 4.2 对照矩阵

建议由主 Agent 在正式评测阶段做以下 2×3 对照：

| 组 | worker 上限 | affinity |
|---|---:|---|
| T8-off / T8-on | 8 | 继承 cpuset / 显式绑核 |
| T12-off / T12-on | 12 | 继承 cpuset / 显式绑核 |
| T16-off / T16-on | 16 | 继承 cpuset / 显式绑核 |

所有组必须保持现有连续帧划分、ESTIMATE、每线程独立 scratch、相同输入和输出语义。有效比较的前置条件：

1. 作业实际 affinity mask 至少有 16 个可用 CPU，且记录完整 CPU ID 列表；不能仅凭申请了 16 CPU 就认定成立。
2. 每个用例记录最终 T、成功 launched 数和分区；若 `F/256` 或 hardware concurrency 使实际 T 低于目标，该用例不进入对应 8/12/16 结论。
3. off/on 使用相同 T 和相同 CPU 集合；on 组任何 affinity 调用失败都要标为无效运行，不能静默降级后继续比较。
4. 输出逐用例验证；线程创建、join、compute、read/write 都在既定总时间口径内。
5. 正式测试由主 Agent 独立运行，交错/轮换六组顺序，保留逐次 wall time、min/median 和环境信息；不能用插桩二进制的单次 wall time代替正式结果。

8/12/16 的优劣不能预设：增加线程会分摊帧计算，也会增加启动、共享只读数据流量、output 写入和内存带宽压力；绑核可能减少迁移和改善 first-touch 稳定性，也可能因 CPU 选择不佳或限制调度而变慢。

### 4.3 可选绑核实现约束

若主 Agent 后续实现 Linux 诊断/实验版本，建议遵循：

1. 启动前用 `sched_getaffinity` 取得当前线程/进程被允许的原始 mask，枚举实际允许的 CPU ID；不能假定是 `0..15`，也不能假定 ID 连续。
2. off 组完全继承原 mask。on 组按同一已记录列表将 worker id 映射到不同允许 CPU；若开启 SMT，需记录逻辑 CPU/物理核关系，不能把两个 SMT sibling 误称为两个独立物理核。
3. 可在每个 worker lambda 入口、任何数据写入之前，用其 native handle 对应的 `pthread_setaffinity_np` 或当前线程 affinity API 自绑定；主线程的 worker 0 也要显式处理。
4. 主线程若在 compute 中被绑到 worker 0 CPU，应在 join 后、返回 main 前恢复原 mask，避免把 `write_output` 的亲和性变化混入“compute 绑核”对照。恢复逻辑需异常安全。
5. 不建议主线程在启动子线程前先缩成单 CPU mask 而忽略继承问题；若采用启动 gate，应把 gate 的同步成本计入并保持一致。
6. 记录绑定返回值及绑定后的实际 mask。numeric-order 映射只是可复现实验策略，不自动等价于 NUMA 最优映射；拓扑感知选择应是另一个有记录的策略，不可偷偷混入 off/on 对照。
7. 不引入 FFTW 内部线程，也不把 OpenMP 环境变量当成 std::thread 的替代实现。

## 5. 独立插桩设计与正式计时隔离

### 5.1 已有能力与缺口

`MELSPEC_PROFILE` 当前提供 plan、聚合 setup 和每线程 pack/FFT/power/mel；`scripts/fftw_profile.py` 在 tmp 中生成 main 副本，额外得到 read/compute/write，并验证输出。它的文件头已明确“独立 profile executable，绝不作为 score 计时”。

需要补充的是**非重叠的准备阶段分解和线程生命周期信息**，而不是继续增加热循环内部计时点。

### 5.2 可新增的独立诊断项

| 类别 | 建议记录 | 解释限制 |
|---|---|---|
| output resize | resize 前后 wall、`O`、`8O` 字节；可选主线程 minor/major fault、RSS 差值 | wall 不能单独区分分配、清零、缺页；RSS/fault 也不是精确流量 |
| mel filter | `build_mel_filterbank` 总时；频率/hz 点、权重构造、first/last 扫描可再分段 | 不在每个 m/k 内计时，避免扰动和海量日志 |
| scratch | input/fft 分配、power 构造、plan、各自分配字节数；线程入口/首帧时间 | 分配 wall 不等于物理触页；首帧时间包含正常计算 |
| worker | 现有四段线程秒之外，记录 begin/end、worker 起止时间、总 wall、实际 launched | 汇总线程秒不能当进程 wall；起止还包含调度和偏斜 |
| 线程创建 | launch 循环、worker(0)、join 的 wall；T、hardware concurrency、cpuset、绑核结果 | 日志和 affinity 查询本身有成本，只能用于诊断 |
| read/write 细分 | 在 tmp 生成的 main 副本中细分缓冲构造、文件 read、y resize/转换，以及 float 构造、转换、write | 不修改受保护 main；细分版只能用于诊断 |

如确需页故障归因，可在诊断版线程边界使用 Linux per-thread `getrusage` 等只读统计，比较 worker 前后的 minor faults/context switches；不得在 perf 或硬件计数不可用时伪造 cache/NUMA 数据。也不要为“测量 first-touch”而在正式路径新增完整预写 pass。

### 5.3 不得混入正式计时

- 新计时必须只存在于独立诊断构建，默认编译时宏展开为空；正式二进制不得执行 chrono、fprintf、affinity 查询、fault/RSS 采样或逐线程统计。
- 诊断版的 `diagnostic_process_wall_s`、read/compute/write 和 worker 数据只用于定位，不能与无插桩正式成绩混排，更不能据其计算加速比。
- 正式计时的输出语义、CLI 和源码保护不变；诊断输出仍需校验，但 PASS 不等于性能有效。
- 本次子任务未执行任何命令或测试，故没有命令行、原始测量或 PASS/FAIL 结果可报告。

## 6. 建议优先级与明确不应实施项

### 可交给主 Agent 后续评估

1. 先把 setup 拆成 mel、output resize、FFTW scratch、power、plan、线程生命周期，避免把约 0.21 s 的历史聚合值误归因。
2. 在满足实际 cpuset 和 T 的条件下做 8/12/16 × off/on 六组对照，保留当前帧划分并记录真实 affinity。
3. 仅在测量显示有意义时，考虑 power scratch 的合法免零初始化/worker first-touch；它与 output 的约束不同，必须独立证明和验证。

### 不应实施

- 不用 `reserve`、伪造 size、raw pointer、allocator  hacks 或 indeterminate read 绕过 `output.resize`；不伪称“worker 会覆盖”就足以合法跳过初始化。
- 不修改受保护 main，不改变 `std::vector<double>&`、double 输出语义、row-major 布局或 CLI。
- 不把 reserve+resize、重复清零、并行预触页、静态复用描述成已证明的 output 优化。
- 不破坏 FFTW 对齐/new-array 约束，不无依据地给 power 加 padding、复制共享 mel/window 或引入额外正式内存遍历。
- 不假定 hardware concurrency 等于 Slurm cpuset，不假定 CPU ID 连续，不假定 16 线程或绑核必然最快。
- 不用 `OMP_*` 直接控制 std::thread，不把诊断插桩、绑核日志或页故障采样混入正式计时。
