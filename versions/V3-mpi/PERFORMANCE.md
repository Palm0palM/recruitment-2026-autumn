# MPI 强扩展与瓶颈记录

本页只接受主 Agent 的受控实测；模型公式不是测量结果。正式数据到位后补表。

## 测量口径

- 固定 FFTW A2 ESTIMATE；同作业构建 A2 与 MPI driver，`-O3 -march=native`、double。
- 每节点 1 Rank、16 CPU、`--hint=nomultithread`，不写 `--mem`；1/2/4 Rank。
- 输入路径分别测 bcast、scatter、shared；输出始终 root 单点写出到节点本地 scratch。
- 每组轮换执行配置，三次取最小并保留 median；任何一次输出 FAIL 都停止。
- A2 直接启动；MPI 用 `mpirun` 包住完整 launcher。外部 wall 包含初始化、分发、计算、
  double→float/布局打包、Gatherv、写出、同步和 Finalize。
- 强扩展 `S(p)=T_A2_min/T_MPI_min(p)`，效率 `E(p)=S(p)/p`。不用跨作业/跨节点数字混算。
- 分阶段读取每 Rank 日志，取最慢 Rank；内部 `process_s` 不含 mpirun 启动前成本，不能代替外部 wall。

## 数据量与待验证假设

PCM 约 49.27 MB，完整 double 输入约 197.08 MB；大输出 float32 约 196.95–197.01 MB。
Bcast 输入量最大；Scatter 仅增加每对相邻 Rank 的 `n_fft-hop` halo；Shared 依赖共享文件系统。
输出 Gather 在 double→float 后执行，但必须先打包 `[local_frame][mel]`，其内存/布局成本单列。

## 结果表（待实测）

| 用例 | 配置 | 最小 wall（s） | median（s） | vs 同作业 A2 | 效率 | 正确性 |
|---|---|---:|---:|---:|---:|---|
| 待测 | A2 / 1 / 2 / 4 Rank × 三种输入 | | | | | |

## 阶段瓶颈与最终判断（待实测）

需要分别回答：

1. MPI 1 Rank 相对普通 A2 的 launcher/init 固定成本。
2. 2/4 Rank 的计算分摊能否覆盖输入分发、输出 Gather 和 root 写出。
3. 哪一种输入路径端到端最好，是否仅在个别用例成立。
4. 打包和 Gather 是否抵消多节点 FFT 收益。
5. 即便原型更快，独立 launcher 是否具备正式评测兼容性。

未获得完整四组输出与同作业对照前，不作“MPI 值得/不值得”的最终结论。
