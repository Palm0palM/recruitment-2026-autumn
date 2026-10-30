# V3 MPI 多节点路线

状态：**已完成受控验证；结论为实验原型失败，不能作为正式提交。**
固定算法 BASE：`b4552f3962751b5a02d2aa56798907c5a1c29afc`。
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
- 2 节点小矩阵已完成：1/2/4 Rank × bcast/scatter/shared，63 个合法输出逐字节等于 A2，
  3 个短输入正确拒绝；详见 [CORRECTNESS.md](CORRECTNESS.md)。

## 最终结果

- 未修改官方 `validate.py` 的研究 wrapper 在 2 Rank 下四组 **4/4 PASS**。
- 2 节点最强组合（`srun` + scatter）仍为 A2 的 **0.45–0.59 倍**，即慢 1.7–2.2 倍。
- 即使把 launcher、float 打包和 Gather 全部设为零，default/small_mels 仍慢 15%/52%；
  large_fft/narrow_band 也只有 1–6% 理论余量。详细阶段数据见 [PERFORMANCE.md](PERFORMANCE.md)。
- 当前无 4 节点空闲资源，未伪造 4 Rank 实测；2 Rank 固定成本和零通信下界已否定通用收益。
- 独立 driver 需要 launcher，正式评测兼容性未知，因此不能作为最终提交方案。

详见 [ENVIRONMENT.md](ENVIRONMENT.md)、[DESIGN.md](DESIGN.md) 和后续 [PERFORMANCE.md](PERFORMANCE.md)。
原始环境日志、正确性与强扩展结果归档在 `raw/`；研发 Subagent 的设计材料在 `research/`。
