# V3 CPU-Pipeline 最终结果

## 推荐候选

算法提交：`d94e45c3a1d1f56dc53f93679e4cab0714b45076`，从固定 FFTW A2 `b4552f39...` 继承。
默认参数：`FFT_MODE=r2c FFTW_PLAN=estimate FRAME_BLOCK=1 MEL_PROJECTION=sparse CPU_THREADS=16`。
正式候选二进制 SHA-256：`518d97704ea18152ecc5b49d66717db8bec356099948d973c0168cd3a68544ec`。
`src/main.cpp`、`validate.py`、`data` 未修改；普通 CLI 和 float32 小端行主序输出保持。

## 集群正式结果

hepnode2，Xeon Gold 6338，1 task × 16 CPU，cpuset `0-15`，nomultithread，不写 mem。
作业 155：完整官方 `validate.py --runs 3`，每组取最小，**4/4 PASS**。

| 用例 | 最终候选（s） | max_abs_err | 同作业内 librosa / 候选 |
|---|---:|---:|---:|
| default | **0.957** | 0.03125 | 16.726× |
| small_mels | **0.648** | 0.01171875 | 23.840× |
| large_fft | **1.334** | 0.125 | 24.063× |
| narrow_band | **1.296** | 0.0625 | 24.828× |

同作业交错对照的 median（固定 A2 / 最终候选）及比值：

| 用例 | 固定 A2 median（s） | 最终 median（s） | A2 / 最终 |
|---|---:|---:|---:|
| default | 0.998759 | 0.985352 | 1.014× |
| small_mels | 0.664821 | 0.663430 | 1.002× |
| large_fft | 1.373683 | 1.355710 | 1.013× |
| narrow_band | 1.319273 | 1.301939 | 1.013× |

两张表来自不同测量环节，不交叉相除。官方数字是三次最小；对照表保留逐次数据与 median。
相对 A2 是约 1% 的整体改善，default 较明显；不宣称所有用例都有同等收益。

## 非 FFT 与端到端 Profile（作业 156，诊断构建）

诊断四组输出全部 PASS；插桩不作为正式成绩。default 的 worker 阶段：
pack 12.83%、FFT execute 70.18%、模平方 6.68%、Mel 10.31%。
V2 A2 对应为 12.39% / 69.27% / 6.56% / 11.78%。
default FFT 线程秒从 3.8067 到 3.7723，large_fft 从 7.8771 到 7.8778，
**FFT 执行成本基本未变**，优化不依赖改变 Plan、FFTW 后端或精度。

| 用例 | read（s） | compute setup（s） | compute 全部（s） | 转换/写出（s） | 诊断 wall（s） | 峰值 RSS（KiB） |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.178865 | 0.212227 | 0.555269 | 0.219934 | 0.985478 | 776344 |
| small_mels | 0.183412 | 0.052774 | 0.401118 | 0.073026 | 0.678508 | 343492 |
| large_fft | 0.183333 | 0.205102 | 0.912153 | 0.223208 | 1.353727 | 772792 |
| narrow_band | 0.177183 | 0.201881 | 0.858548 | 0.232108 | 1.297432 | 772356 |

setup 细分：

| 用例 | 压缩 Mel 构造（s） | output.resize（s） | scratch+plan（s） |
|---|---:|---:|---:|
| default | 0.000113 | 0.202678 | 0.009259 |
| small_mels | 0.000075 | 0.049972 | 0.002545 |
| large_fft | 0.000150 | 0.201584 | 0.003054 |
| narrow_band | 0.000114 | 0.198449 | 0.003005 |

压缩 Mel 构造已降到约 75–150 μs；output 初始化仍占 setup 大部分，但固定 vector/main 接口下不能合法跳过。
scratch 很小，不值得引入未初始化 owner 或额外 NUMA 复杂性。输出/输入 double 存储仍在，峰值内存未发生数量级变化。

## 独立采用与回退

- **采用 S1 压缩滤波器**：公开稠密接口不变，39 组输出与 A2 完全相同；构造和热点权重存储下降。
- **采用 B=1**：B=2/4/8/16 在集群整体回退，仅 small_mels 的 B=2 有微小局部优势，不据此选择全局参数。
- **拒绝双前缀矩**：default 局部变快但其他三组回退；高动态压力测试出现 168 个负值和严重抵消。
- **采用 16 workers**：集群四组均为 16 < 12 < 8；没有把 OMP 变量误当作 std::thread 绑核。
- **不做 output 初始化 hack、额外预触页或 NUMA 复制**：setup 证据不支持合法且有收益的改动。

B=1/2/4/16 均通过 ASan/UBSan、smoke 和 39 组对照；额外 Kimi 只读审查未发现分块越界、
矩索引、默认计时宏副作用或线程失败回退问题。既有受保护 main 的奇数字节 PCM 风险不在本路线修改。

## 复现与证据

- 构建/集群步骤：[REPRODUCE.md](REPRODUCE.md)
- 全部实验与失败数据：[EXPERIMENTS.md](EXPERIMENTS.md)
- 基线：[BASELINE.md](BASELINE.md)
- 集群原始：`raw/cluster/cpu-155.out`、`official-155.txt`、`compare-155.json`、`cpu-156.out`、`accounting.txt`
- 帧块与 P 失败：`raw/frame-block-*`、`raw/p1-*`、`moment-pressure.txt`

后续 CPU 优化若继续，应优先研究受保护接口之外的合法数据流，而不是重新尝试已被否决的
大帧块、双前缀、FFTW 后端变化或 output 初始化规避。任何新改动仍需同节点 A2/本候选对照。
