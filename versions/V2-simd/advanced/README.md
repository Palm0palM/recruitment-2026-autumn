# V2 SIMD 进阶实验：集群基准、AVX-512 与跨帧

共同前置版本为 `dc1af4bdd5c98f9a79a466bf061646399dd399cb`。原始 B1～B6 记录见上一级 README；本文件追加后续实验，不改写历史数据。

## 1. 冻结与测量口径

- 冻结源码 SHA-256：`78c968fedcf8d30207af7f19c36dc739275df85a8e733bbb5178535bb60ed552`
- 本地 GCC 16.2.1；集群 GCC 15.2.0。
- 构建：`-std=c++17 -O3 -march=native -pthread`，无 fast-math，保持 double。
- 集群：hepnode3，Xeon Gold 6338，1 task × 16 CPU，`--hint=nomultithread`，cpuset `0-15`，不传 `--mem`。
- 每个作业内先构建，再轮转全部程序执行顺序；每个程序每组 6 次，保存 min、median 和逐次 wall time。
- 输出在节点本地 scratch，四组均与官方 expected 全量比较。
- FFTW 对照：A2 原生 R2C ESTIMATE，算法锚点 `b4552f3962751b5a02d2aa56798907c5a1c29afc`，FFTW 3.3.11 静态库在当前计算节点原生构建。
- 基线作业 74、AVX-512 作业 79、Batch=4 作业 84、阶段 profile 作业 91。所有集群任务串行提交，未运行 V0。

## 2. 同节点三方基线（作业 74）

| 用例 | V1 median (s) | 冻结 SIMD median (s) | FFTW median (s) | SIMD vs V1 | FFTW vs SIMD |
|---|---:|---:|---:|---:|---:|
| default | 1.551891 | 1.223099 | 0.994768 | 1.268× | 1.230× |
| small_mels | 1.210489 | 0.880885 | 0.659749 | 1.374× | 1.335× |
| large_fft | 2.664824 | 1.959554 | 1.368281 | 1.360× | 1.432× |
| narrow_band | 2.599971 | 1.909333 | 1.305817 | 1.362× | 1.462× |

四组全部 PASS。当前自研版本已明显优于 V1，但 FFTW 仍快约 23%～46%，大 FFT 的差距更大。原始逐次数据见 `raw/cluster/baseline-74.*`；旧 hepnode2 或其他时间的数字不参与倍率计算。

## 3. AVX-512 单帧 8-lane：正确但退化（作业 79）

实验源码归档为 `raw/melspec-avx512.cpp`，未进入最终源码。新增内容仅为：

1. 通用 radix-4 的 8-lane double/ZMM FMA，保留 AVX2 尾部；
2. 谱恢复的 8-lane 反向加载与模平方；
3. 保留 AVX2 的 DFT4、len=8 专用路径，未把早期 Kernel 错误地条件互斥掉；
4. 使用 unaligned load，不假定 `std::vector` 64B 对齐，未额外展开以控制寄存器压力。

本地只进行 `-march=icelake-server` 交叉编译和 objdump，未执行 ZMM 二进制。集群反汇编确认 ZMM packed FMA 与全宽 `vpermpd`，不是未启用指令。j=0 未单独剥离，避免破坏连续向量分组。

| 用例 | AVX2 median (s) | AVX-512 median (s) | AVX-512 / AVX2 时间比 |
|---|---:|---:|---:|
| default | 1.228595 | 1.322818 | 1.077× |
| small_mels | 0.880850 | 0.978067 | 1.110× |
| large_fft | 1.951212 | 2.175912 | 1.115× |
| narrow_band | 1.912004 | 2.127356 | 1.113× |

四组正确性均 PASS，但 AVX-512 慢约 8%～12%，因此**不保留**。仅增加向量宽度并未改善目标机性能；可能的频率和内存数据流成本需要通过实际 wall time 判断，本实验不声称已用 PMU 证明降频原因。原始结果和 ZMM 证据见 `raw/cluster/avx512-79.*`、`raw/avx512-*`。

## 4. AVX2 Batch=4 跨帧（提交 `770942f`）

### 数据布局与数学

每个频率索引用四个 double lane 表示四个连续时间帧：

```text
re[i] = [frame(t+0).re[i], frame(t+1).re[i], frame(t+2).re[i], frame(t+3).re[i]]
im[i] = [frame(t+0).im[i], frame(t+1).im[i], frame(t+2).im[i], frame(t+3).im[i]]
```

位反转与四帧加窗/偶奇打包一次完成，FFT 全程不做跨 lane 操作。每个 radix-4 j 的同一组 twiddle 广播到四帧；initial radix-2 和 quarter=1 的 DFT4 直接规则执行，不再需要单帧首级转置和 len=8 的栈暂存。

谱恢复直接读取 `re[half-k]`，lane 顺序不变，消除单帧反向加载后的 permute。Mel 直接对四帧功率谱做 FMA 并连续写四个输出，不需要 AoSoA→SoA 转换。线程按全局 batch 分区，最后 0～3 帧走单帧路径；不物化全量功率谱。

### 集群结果（作业 84）

| 用例 | 冻结 SIMD median (s) | Batch=4 median (s) | FFTW median (s) | Batch vs 冻结 | Batch vs FFTW |
|---|---:|---:|---:|---:|---:|
| default | 1.219966 | **0.937963** | 0.996682 | 1.301× | **1.063×** |
| small_mels | 0.884385 | **0.595915** | 0.657029 | 1.484× | **1.103×** |
| large_fft | 1.959421 | **1.300800** | 1.363878 | 1.506× | **1.048×** |
| narrow_band | 1.913292 | **1.279153** | 1.310996 | 1.496× | **1.025×** |

四组全部 PASS。Batch=4 不仅缩小了差距，而且四组均超过同作业 FFTW。最终代码与二进制 SHA、完整逐次时间见 `raw/cluster/batch4-84.*`。

本地随机/边界回归包括官方四组、CLI 默认值、n_fft=1/2/8 和新增 tail_2/tail_3；公开 `compute_rfft` 的 1～16384 二次幂测试也全部通过。独立 Kernel microbenchmark 与端到端测试均保存于 `raw/batch4-micro-*`、`raw/local-ab.*`。

### 阶段成本（集群作业 91）

诊断副本按线程累计时间，以下为**所有线程累计 ns/batch**，不是进程 wall time。每批包含四帧，正式 benchmark 不带插桩。

| 阶段 | default ns/batch | default ns/frame | large_fft ns/batch | large_fft ns/frame |
|---|---:|---:|---:|---:|
| 位反转/四帧输入打包 | 28,753 | 7,188 | 58,069 | 14,517 |
| Batch FFT | 47,881 | 11,970 | 114,890 | 28,722 |
| 实数谱恢复/模平方 | 12,411 | 3,103 | 25,205 | 6,301 |
| Mel 与四帧写出 | 5,436 | 1,359 | 10,972 | 2,743 |

收益来自整个数据流，而不是只来自 FFT 指令：打包不再需要四路独立散写，FFT 小阶段不需要转置，恢复不需要反向 permute，Mel 权重也可跨四帧复用。两组诊断输出均 PASS，原始每线程数据见 `raw/cluster/batch4-profile-91.out`。

## 5. 当前选择

当前最佳候选为提交 `770942f` 的 AVX2 Batch=4。单帧 AVX-512 路径作为失败实验归档，未混入该候选。后续 Batch=2/8 和算法重构必须继续与此冻结版本及 FFTW 同节点对照，不能将不同作业的倍率连乘。
