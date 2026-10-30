# 同节点交错 A/B：冻结候选 / 旧版 A2 / SIMD Batch8

脚本：`scripts/fftw-ab.slurm`。本 worktree **只提供公平测量框架**，不构建、不修改 SIMD 实验。

## 前置条件

- 冻结候选与 A2 由本 worktree 从当前提交构建；PATIENT Wisdom 使用
  `experiments/raw/cluster/final-wisdom/patient/`。
- SIMD Batch8 可执行文件由该实验负责人提供绝对路径；本脚本只读取并执行：

```sh
export SIMD_BATCH8_EXE=/home/zsx/<simd-worktree>/tmp/bin/<batch8-binary>
sbatch --wait -w hepnode2 --export=ALL scripts/fftw-ab.slurm
```

- 必须等节点上其他 Agent 的作业结束再提交。脚本启动时会检查 `squeue -w <node> -t RUNNING`，
  若发现其他作业立即以退出码 3 拒绝执行，不会与别人并行跑 Benchmark。

## 测量口径

- 单节点、1 task、16 CPU、`--hint=nomultithread`、不申请 `--mem`。
- 三个候选按轮换顺序交错执行，5 轮取中位数；每轮都做完整输出比对（原容差）。
- 含 Wisdom 的候选必须报告 `status=loaded`，fallback 直接判失败，不把回退当收益。
- 同时输出每用例中位数与相对 A2 的几何平均。

## 输出

```
tmp/logs/ab-<job>.json          结构化每次耗时与最大误差
tmp/logs/ab-table-<job>.txt     逐用例表
tmp/logs/ab-<job>.out/.err      作业日志与哈希
```

## 解释边界

- 三个候选代码不同，必须同时看正确性与耗时；不得只挑最快的一次运行。
- 不把此前不同作业的独立加速比相乘，也不把 SIMD 的本地结果直接与本集群结果拼接。
- 若 SIMD 候选在任一用例完整输出校验失败，本轮 A/B 结论为该候选未通过验收，而不是速度比较。
