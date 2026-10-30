# MPI 原型正确性记录

作业 258，hepnode[2-3]，2 节点分配；1/2 Rank 每节点 1 Rank + 16 PE 映射，
4 Rank 每节点 2 Rank、各 8 核并绑定；小矩阵不作为性能成绩。原始日志：
`raw/mpi-experiment-258.out`，Slurm 记账：`raw/slurm-accounting-258.txt`。

## 测试矩阵

| 用例 | 参数/边界 | 1 Rank ×3 输入 | 2 Rank ×3 输入 | 4 Rank ×3 输入 |
|---|---|---|---|---|
| random-default | n_fft8192/hop128/M256 | PASS | PASS | PASS |
| random-large | n_fft16384/hop128/M64 | PASS | PASS | PASS |
| one-frame | N=n_fft，F=1，空 Rank | PASS | PASS | PASS |
| two-frames | 非整除、F=2 | PASS | PASS | PASS |
| hop-greater-than-fft | n_fft256/hop512 | PASS | PASS | PASS |
| fft-one | n_fft=1，题意零窗 | PASS | PASS | PASS |
| fft-eight | n_fft=8 | PASS | PASS | PASS |
| short-N0/N1/N2047 | N<n_fft | 2 Rank 全部失败，符合预期 | | |

三种输入路径：`bcast`、`scatter`、`shared`。**63 个合法运行**均与独立 A2 串行输出
**逐字节完全一致**，不是只用容差近似通过；三个短输入均返回非零并拒绝，无死锁。

4 Rank 的运行日志确认 Rank 0/1 在 hepnode2、Rank 2/3 在 hepnode3；每 Rank cpuset 8。
`one-frame` 与 `two-frames` 证明空 Rank 不读 PCM、不调用 DSP、仍参与集合通信；
`hop>n_fft` 证明帧间空洞不被非法填补。

## 原型实现验证点

- 全局帧连续均分与局部 PCM 切片 `(count-1)*hop+n_fft` 正确。
- 输出在 Gather 前转换并打包为 `[local_frame][mel]`；root 用 resized vector datatype
  按 `recvcounts=local_frames`、`displs=begin_frame` 汇成完整 `[n_mels,n_frames]`。
- root 是唯一写出者，输出 float32 小端、行主序，无文件头。
- 任何 Rank 失败经 `MPI_Abort` 使整体失败，不写出部分成功结果。

## 重要限制

1. 这是独立 `scripts/mpi_melspec.cpp` driver 的结果，不是受保护 `src/main.cpp` 正式入口。
2. 它通过研究 wrapper 调用未修改的 `validate.py` 也不能自动证明赛事允许额外 launcher；
   正式兼容性仍标记未知，不能用本结果宣称最终提交合法。
3. `shared` 小矩阵出现同时读共享文件的长尾等待（功能仍正确），正式性能必须单独测量，
   不能拿本次 1 秒级的初始化/共享 I/O 数字判断扩展性。
4. 4 Rank 正确性运行采用每节点 2 Rank ×8 核的小矩阵映射，**不是**强扩展的正式
   每节点 1 Rank ×16 核口径。
