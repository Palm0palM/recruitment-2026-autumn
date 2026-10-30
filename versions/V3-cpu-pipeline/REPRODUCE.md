# V3 CPU-Pipeline 复现步骤

## 固定来源与默认候选

- 共同算法 BASE：`b4552f3962751b5a02d2aa56798907c5a1c29afc`（FFTW A2 R2C + ESTIMATE）。
- 实验分支：`v3/cpu-pipeline`，独立 worktree；没有合并 master 或修改其他 worktree。
- 默认候选参数：`FRAME_BLOCK=1`、`MEL_PROJECTION=sparse`、`CPU_THREADS=16`。
- 最终算法提交：`d94e45c3a1d1f56dc53f93679e4cab0714b45076`；官方结果见 [RESULTS.md](RESULTS.md)。
- 保留 FFTW 原生 R2C、单变换 Plan、double、外层 worker 分区和受保护 CLI。
- 不修改 `src/main.cpp`、`validate.py`、`data`，不研究 Wisdom/Batch FFT/Codelet。

独立提交：
- S1 压缩滤波器：`50e75cc`
- B 帧块框架：`50519dc`（集群选择 B=1）
- P 双前缀失败实验：`a8a6a09`（不采用）
- setup 细分：`9a49a8c`
- 线程扫描：`d94e45c`

## 本地构建与正确性

```bash
source tmp/session-env.sh
make -B -j4 FFT_MODE=r2c FFTW_PLAN=estimate FRAME_BLOCK=1 \
  MEL_PROJECTION=sparse CPU_THREADS=16 EXEC=tmp/bin/candidate
$PY scripts/smoke.py tmp/bin/candidate
```

`fftw_check.py` 可对照固定 A2 二进制，覆盖 39 组随机/边界/满幅/冲激/Nyquist 输入与短输入拒绝：

```bash
$PY scripts/fftw_check.py --reference tmp/bin/a2-base tmp/bin/candidate
```

本地全量对照（每次输出都检查 expected；不是正式集群成绩）：

```bash
$PY scripts/fftw_bench.py \
  --programs a2-base=tmp/bin/a2-base candidate=tmp/bin/candidate \
  --runs 5 --out tmp/results/candidate.json
```

阶段诊断独立编译，不用于正式计时：

```bash
$PY scripts/fftw_profile.py --mode r2c --planner estimate \
  --frame-block 1 --mel-projection sparse
```

双前缀失败实验可用 `MEL_PROJECTION=moments` 独立复现；`scripts/moment_check.cpp` 展示抵消。
默认 sparse 不会执行前缀逻辑，不要通过 clamp 隐藏负值。

## 集群依赖与作业

远端使用独立目录 `/home/zsx/melspec-v3-cpu-pipeline`。部署本分支 bundle/clone 后：
```bash
mkdir -p tmp/logs
sbatch -w hepnode2 --export=ALL,MODE=build scripts/cpu_pipeline_cluster.slurm
# 等 build 完成后串行执行
sbatch -w hepnode2 --export=ALL,MODE=bench,OFFICIAL=b1 scripts/cpu_pipeline_cluster.slurm
sbatch -w hepnode2 --export=ALL,MODE=threads,OFFICIAL=t16 scripts/cpu_pipeline_cluster.slurm
sbatch -w hepnode2 --export=ALL,MODE=profile,PROFILE_BLOCK=1,PROFILE_MEL=sparse scripts/cpu_pipeline_cluster.slurm
```

可用 `--dependency=afterok:<jobid>` 保证不并行占用节点。作业一 task、16 CPU、nomultithread、不写 mem。
`build_fftw.sh` 在本实验 tmp 内无 sudo 构建 FFTW 3.3.11 静态 double 库，集群不依赖预装数学库。
所有大输出在节点本地 `/tmp/melspec-v3-cpu-$JOBID-*`，作业结束自动清理。
每次正式比较都从固定 b4552f3 重编 A2，记录 commit、源码/库/程序 SHA-256、节点和 cpuset。
完整官方 `validate.py --runs 3` 必须 4/4 PASS，不能只凭 max_abs_err 判定。

## 文档和原始证据

- `BASELINE.md`：固定 V1/A2 与本地基线。
- `EXPERIMENTS.md`：逐项采用/回退决策、数学与性能结果。
- `research/`：Kimi 三项独立研究和审查材料。
- `raw/`：本地逐次输出、正确性与压力测试。
- `raw/cluster/`：集群原始日志、完整官方输出和比较 JSON。

不要提交 tmp 下的 venv、PCM/expected、FFTW 库、二进制、perf.data 或临时大输出。
