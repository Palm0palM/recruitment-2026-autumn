# MPI 强扩展与瓶颈记录

作业 265，hepnode[2-3]，2 节点 × 16 CPU；固定 FFTW A2 ESTIMATE，四个官方用例，
每次配置 3 次并轮换顺序。外部 wall 包含 launcher、MPI 初始化、读、分发、worker、
float/布局打包、Gatherv、root 写出和 Finalize。**108 次运行全部与 official expected PASS。**

## 1. 端到端结果（最小 wall，秒）

| 用例 | A2 单进程 | MPI 1 Rank<br>`srun` scatter | MPI 2 Rank<br>`srun` scatter | MPI 2 Rank<br>`mpirun` scatter | MPI 2 Rank<br>`srun` shared | 最佳 MPI / A2 |
|---|---:|---:|---:|---:|---:|---:|
| default | **0.989** | 2.506 | **2.183** | 2.315 | 2.197 | **0.45×** |
| small_mels | **0.662** | 1.703 | **1.478** | 1.672 | 1.497 | **0.45×** |
| large_fft | **1.351** | 2.904 | **2.309** | 2.488 | 2.338 | **0.59×** |
| narrow_band | **1.295** | 2.834 | **2.301** | 2.480 | 2.303 | **0.56×** |

这里的“最佳 MPI / A2”是 A2 时间除以最佳 MPI 时间，均小于 1，表示 MPI **更慢**。
A2 在相同作业中的 0.989/0.662/1.351/1.295 秒与既有正式基线一致，说明对照口径可比。
MPI 1→2 Rank 确实减少了 worker 时间，端到端仅改善约 13–26%；
但相对同节点 A2，全部四组都慢 1.7–2.2 倍。强扩展效率最高也只有约 **0.30**，不具实际收益。

## 2. 为什么多节点无效：固定成本已经超过可分摊计算

以每例最佳 `srun` + scatter 的 rank 0 阶段数据（秒）：

| 用例 | 外部总 | launcher<br>差额 | MPI init | read | distribute | compute | cast+pack | Gather | root write | finalize | **去掉 launcher/pack/Gather 的下界** | 下界/A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 2.183 | 0.388 | 0.354 | 0.183 | 0.072 | 0.280 | 0.085 | 0.563 | 0.096 | 0.152 | **1.138** | 1.15× |
| small_mels | 1.478 | 0.372 | 0.363 | 0.183 | 0.072 | 0.112 | 0.017 | 0.109 | 0.027 | 0.129 | **1.004** | 1.52× |
| large_fft | 2.309 | 0.390 | 0.347 | 0.183 | 0.076 | 0.472 | 0.125 | 0.509 | 0.083 | 0.116 | **1.271** | 0.94× |
| narrow_band | 2.301 | 0.373 | 0.362 | 0.183 | 0.074 | 0.468 | 0.118 | 0.518 | 0.083 | 0.118 | **1.282** | 0.99× |

“下界”连 launcher、float 打包、真实 Gather 全部强制设为零，只保留 MPI 初始化、
root 读取、任务分发、1/2 计算、root 写出、同步和 Finalize。即使如此：

- default 和 small_mels **仍慢于 A2 15%/52%**，不存在任何优化空间能翻盘。
- large_fft 只有约 **6%** 的理论余量，narrow_band 只有约 **1%**；任何非零通信、launcher
  或打包都会立即变成负收益。因此不是“Gather 再优化一点就行”。

实测 Gather 在 197 MB 输出上约 0.51–0.56 秒；局部 double→float 打包再需约 0.09–0.13 秒，
两者都远大于 p2 计算节省。`srun` 已是最低开销启动器，`mpirun` 多约 0.2–0.3 秒；
`shared` 对部分用例略好于 scatter，但差异远小于固定开销，且会放大共享文件系统争用。

## 3. 4 Rank 与正式提交判断

当前集群没有同时空闲的 4 个节点；其他用户作业触发 `RealMemory=1` 的一节点一作业限制。
因此本报告**没有伪造 4 Rank 实测**。2 Rank 的固定成本和实测 scaling 已足以否定
default/small_mels；对 large_fft/narrow_band，上述“零通信下界”最多也只有 1–6% 余量，
而真实 4 Rank 只会增加 Gather 节点和 launcher/初始化成本，不可能达到通用收益目标。

更根本的限制：`validate.py` 直接执行程序，原始 `src/main.cpp` 没有 Rank 管理；
当前独立 driver 必须依赖研究 wrapper/launcher，正式评测合法性未知。即使假设未来性能成立，
这条路线也**不能作为不改受保护入口的正式提交候选**。

**最终判断：MPI 多节点路线在本集群和本题约 1 秒的单节点规模下失败。**
保留原型和负结果用于 PPT/后续说明，不合并进 master，不宣称正式加速。

## 4. 数据与复核

- `raw/mpi-bench-265.json`：108 次运行、每 Rank 阶段计时、命令和最大误差。
- `raw/mpi-experiment-265.out/.err`：作业原始输出、源码/二进制哈希和节点信息。
- `raw/mpi-official-264.txt`：未修改官方 `validate.py` + 研究 wrapper 的 2 Rank 四组 PASS。
- `raw/slurm-accounting-264.txt`、`-265.txt`：Slurm 状态和资源。

测量在不同节点/作业之间波动约 1–3%，但 MPI 的差距是 70–120%，远超测量噪声。
任何重跑必须保留同作业 A2 对照，不能挑选单次 MPI 最小值与旧 A2 比较。
