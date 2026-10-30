# 答辩材料：FFTW 路线的优化历程与结果

本轮最终候选：**R2C + float FFT + float 模平方 + PATIENT Wisdom**，同作业五轮中位数相对 A2
几何平均 **1.313×**；官方四用例 **4/4 PASS**（Job 209）。默认构建仍是全 double A2。

## 1. 问题与基线

任务：把 Hz→Mel（HTK/Slaney）、Hann 加窗、R2C 与稀疏 Mel 矩阵乘做成多线程 CLI，输出
`[mel][frame]` float32。原始优化起点 V1 已用 FFTW，但主要瓶颈仍是每帧 FFT 与滤波器累加。

第一轮（Job 58，同作业、同输入/编译器/线程）三次取最小：

| 版本 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| BASE/V1 | 1.539891 | 1.209326 | 2.655534 | 2.597472 |
| A1 half-C2C ESTIMATE | 1.104407 | 0.758823 | 1.596253 | 1.544488 |
| A1 half-C2C MEASURE | 1.591448 | 1.262902 | 2.260757 | 2.217947 |
| **A2 native R2C ESTIMATE** | **1.003307** | **0.662364** | **1.365753** | **1.310244** |
| A2 native R2C MEASURE | 1.372112 | 1.036879 | 1.903017 | 1.844392 |

A2 ESTIMATE 相对 V1：**1.535× / 1.826× / 1.944× / 1.982×**。
官方计时（三次取最小）：0.985 / 0.650 / 1.356 / 1.300 s，4/4 PASS。

> 结论 1：把半长 C2C + 频谱恢复换成原生 R2C，直接消除恢复阶段；冷 MEASURE 的规划成本
> （0.45–0.82 s）超过其核加速，单输入 CLI 不划算。

## 2. 第二轮：四个方向与组合

### 2.1 单精度 FFT（保留 double Mel）

只让 FFT 输入/输出走 `fftwf`，Hann 与 Mel 权重/累加仍 double；给出 float 模平方与 double
模平方两种路径。单次 Job 93 中位数：

| 配置 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| A2 double | 1.011655 | 0.660833 | 1.363302 | 1.320362 |
| float + double power | 0.906400 | 0.562626 | 1.213276 | 1.149965 |
| **float + float power** | **0.893212** | 0.574579 | 1.193986 | 1.135100 |

误差来源可分：输入量化、FFT 舍入、模平方舍入、Mel 累加。功率扰动满足
`|ΔP| ≤ 2|X||ε| + |ε|²`，Mel 权重非负，故 `|ΔS| ≤ Σ W|ΔP|`。低能量相消必须靠绝对容差，
所以除官方用例外补了 54 组对抗用例（满幅/近 Nyquist/静音/脉冲/强+弱多音，N=256…65536）。
最差归一化误差约 0.035（失败阈值 1），但这只是反例搜索结果，不是全输入证明。

### 2.2 Wisdom：把“搜索”移出计时

离线用 MEASURE/PATIENT 生成 plan，运行时只 `WISDOM_ONLY|ESTIMATE` 导入。envelope 绑定 FFTW
版本/编译标识/库 SHA/精度/CPUID-XCR0/尺寸/布局/对齐；任一项不符即明确 fallback 到 ESTIMATE。
CPU 指纹最初误含 APIC ID（随逻辑核变化），review 时修正为屏蔽 EBX[31:24]。

规划耗时（float，hepnode2）：8192 时 MEASURE 0.379 s / PATIENT 2.295 s；16384 时 0.531 / 3.628 s。
这些成本**不计入**每次 CLI 计时，但若部署无法预生成，收益不成立。

### 2.3 Batch（失败）

测试 B=1/4/8/16/32 与 padded-row、interleaved 两种布局。Job 126 原始 B=4 相对 A2 为
**0.960/0.950/0.895/0.891×**；Job 151 padded B=4 为 **0.996/1.002/0.939/0.936×**，
interleaved 更差（0.705× 几何平均）。原因是工作集、跨步 Mel 读取与 plan 分解变化抵消了调用开销。

### 2.4 ISA（失败方向）

用相同标量编译选项分别构建 AVX2-only 与 AVX-512-enabled 库。Job 126：AVX-512 相对 AVX2
仅 **1.021–1.054×**，相对现有 native 库无实质提升；AVX2-only 反而慢 2–5%。现有库本就在用
AVX-512 codelet，“关掉 AVX-512”不是优化。

## 3. 最终组合（Job 151 + 209）

同作业五轮中位数，括号为相对 A2：

| 配置 | default | small_mels | large_fft | narrow_band | 几何平均 |
|---|---:|---:|---:|---:|---:|
| A2 double ESTIMATE | 1.0047 | 0.6670 | 1.3697 | 1.3116 | 1.000× |
| float / float power | 0.8942 (1.124×) | 0.5600 (1.191×) | 1.1959 (1.145×) | 1.1393 (1.151×) | 1.153× |
| float + MEASURE wisdom | 0.8223 (1.222×) | 0.4916 (1.357×) | 1.0791 (1.269×) | 1.0298 (1.274×) | 1.280× |
| **float + PATIENT wisdom** | **0.8261 (1.216×)** | **0.4913 (1.358×)** | **1.0237 (1.338×)** | **0.9764 (1.343×)** | **1.313×** |

官方 4/4（Job 209，单次运行）：0.828 / 0.492 / 1.020 / 0.976 s。
含 Wisdom 加载的组合相对无 Wisdom float 为 **1.139×**（直接实测，不相乘）。
峰值内存约 **769 / 778 MiB**（default / large_fft），与 A2 基本持平。

## 4. 目标版本阶段 Profile

> 见下方“Profile 结果”一节；旧 PROFILE.md 的 69–75% 来自 double A2，不用于本候选。

## 5. 关键技术点（答辩可讲）

1. **一个 plan、16 个外层 worker、私有对齐 buffer**：new-array execute 共享只读 plan，
   规划与析构串行化；不开 FFTW 内部线程，避免 16×N 过度订阅。
2. **稀疏 Mel**：只遍历三角滤波器的非零支撑，按 bin 递增顺序 double 累加，保持语义。
3. **精度分层**：FFT 用 float 换取吞吐，重活（窗、权重、累加、输出）保持 double，误差可控。
4. **离线规划 + 在线导入**：把不可预测的 planner 搜索变成可复现实验前步骤；指纹失配安全回退。
5. **测量纪律**：同作业交错轮换、完整输出校验、同 A2 基线、保留所有轮次与失败记录；
   不把不同作业的最小值或独立收益相乘。

## 6. 失败实验与教训

| 尝试 | 结果 | 原因/处理 |
|---|---|---|
| A1 MEASURE | 比 ESTIMATE 慢 | 冷规划成本 > 执行收益 |
| Batch B=4…32 | 全部回退 | 工作集/访存/分解变化 |
| padded/interleaved 布局 | 未超过 A2 | 同上 |
| AVX2-only 库 | 慢 2–5% | 丢失现有 AVX-512 codelet |
| 全局 `-mno-avx512f` | 编译失败 | 覆盖 FFTW codelet 专属 flags；改为相同标量基线 |
| 脚本安装路径错误 | Job 104 失败 | 修正为 builder 实际 prefix |
| wait4 ru_maxrss | 数据无效 | 继承父进程 RSS；改采样 /proc VmHWM |
| 本地两轮筛选 | 计时噪声大 | 只保留正确性，不作加速结论 |
| 六配置官方验证 | Job 151 超时 | 成绩已完成；短作业 209 补两项官方验证 |

## 7. 局限与风险（必须主动说明）

- **精度**：float 是显式近似选项，默认仍 double；不保证任意幅度公共 API 输入满足 double 精度。
- **Wisdom 可移植性**：绑定 CPU/库/精度/尺寸；换环境必须重新生成，否则回退 ESTIMATE。
- **部署前提**：PATIENT 规划必须事先完成；若每次进程都现场规划，1.313× 不成立。
- **收益范围**：仅在本题四个用例与同节点测得；batch/ISA 收益不可外推。
- **许可证**：FFTW GPL-2.0-or-later，分发需履行源码义务。

## 8. 复现入口

- 冻结清单：`delivery/FREEZE.json`；构建部署：`delivery/FROZEN.md`
- 原理：`experiments/DESIGN.md`；全部数据：`experiments/RESULTS.md`
- 部署校验：`scripts/fftw_verify_deployment.py`
- 目标 Profile：`scripts/fftw-profile-final.slurm`
- A/B（对 SIMD Batch8 与 A2）：`scripts/fftw-ab.slurm`
