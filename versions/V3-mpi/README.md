# V3 MPI 多节点路线

状态：研发与受控验证中。固定算法 BASE：`b4552f3962751b5a02d2aa56798907c5a1c29afc`。
本路线不改 FFTW A2 内核，只研究跨 Rank 的帧分配、输入分发和输出汇集。

## 已确认基线

| 用例 | V1（hepnode3，秒） | FFTW A2 ESTIMATE（hepnode2，秒） |
|---|---:|---:|
| default | 1.537 | 0.985 |
| small_mels | 1.196 | 0.650 |
| large_fft | 2.634 | 1.356 |
| narrow_band | 2.578 | 1.300 |

后续强扩展必须重测同作业 A2，不直接拿上述不同节点/harness 的最小值计算 MPI 加速比。

## 当前结论边界

- OpenMPI 4.1.6 已在计算节点通过 singleton、`srun --mpi=pmix`、`mpirun` 的单节点与双节点 hello 验证。
- `validate.py` 直接启动程序，原始 main 没有 Rank 管理；普通多 Rank 不是其原生执行方式。
- 独立 driver + 研究 wrapper 仅用于实验，**当前不能声称是正式提交入口**。
- 三个 Kimi 子任务完成了数据划分、输出汇总和性能模型；主 Agent 负责集成和全部集群测试。

详见 [ENVIRONMENT.md](ENVIRONMENT.md)、[DESIGN.md](DESIGN.md) 和后续 [PERFORMANCE.md](PERFORMANCE.md)。
原始环境日志、正确性与强扩展结果归档在 `raw/`；研发 Subagent 的设计材料在 `research/`。
