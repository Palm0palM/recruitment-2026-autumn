# 本地 → 集群工作流

题目原文见根目录 `README.md`；优化结论和原始数据见 `versions/`。
不要修改 `src/main.cpp`、`validate.py` 或 `data`。临时文件放 `tmp/`，需要展示的测量日志再归档进对应版本。

## 本地：编译与小规模验证

```bash
source tmp/session-env.sh
make -j4 CUDA_ARCH=120           # 本机 CUDA 功能构建；L40 集群使用默认 89
# make -j4 CUDA=0                # 或构建 V1 CPU 对照
$PY scripts/smoke.py ./melspec   # 固定种子随机 PCM，官方四组、默认 CLI、小 FFT
$PY scripts/bench.py --slice 0.5 --runs 1  # 真实 PCM 切片；不是正式成绩
```

- venv：`tmp/.venv`；目标文件：`tmp/build`；程序：`./melspec`。
- V3 CUDA 默认使用 `CUDA=1`、FP32、batch512、fused、初始化重叠；`CUDA=0` 构建 V1 CPU 对照。
- 集群通过私有 `CUDA_HOME` 提供 CUDA/cuFFT，不依赖系统 toolkit 或 FFTW。
- `data` 是题目提供的集群绝对路径链接，本地不改它；副本从 `$MELSPEC_DATA` 或 `tmp/data` 读取。
- `setup.sh` 可自动发现旧的本地副本 `../autumn26/cluster_melspec/data`。
- 本地只有约 15 GiB 内存，不跑官方全量 librosa；默认只做 smoke test。

## 同步、计算节点编译与 benchmark

远端目录可自选。V1 使用独立的 `~/recruitment-2026-autumn-v1`，避免覆盖远端旧工作区。

```bash
rsync -av --exclude=.git/ --exclude=tmp/ --exclude=__pycache__/ \
  --exclude=AGENTS.md --exclude=melspec --exclude='*.o' \
  ./ hepnode:recruitment-2026-autumn-v1/
ssh hepnode 'cd ~/recruitment-2026-autumn-v1 && mkdir -p tmp/logs && sbatch scripts/bench.slurm'
```

作业默认在 `hepnode[1-3]` 上申请 1 节点、1 task、16 核，**不写 `--mem`**，
加 `--hint=nomultithread`。编译在作业内执行 `make -B`，避免复用本地二进制或其他 CPU 的目标文件。
默认使用集群已有的 `~/venv/bin/python`，没有 spack 或系统 FFTW/BLAS 可直接调用。

```bash
# 默认：官方 validate.py，四组全量，各 3 次取最小，含同作业内 librosa 对照
sbatch scripts/bench.slurm
# 快速检查：单组全量，不额外测 librosa；参数仍来自官方 TEST_CASES
CASE=small_mels RUNS=1 sbatch scripts/bench.slurm
```

- 官方四组：`sr=96000, hop=128`；FFT 为 8192/16384，Mel 数为 256/64。
- 正确性：`rtol=1e-3, atol=1e-4`。包装脚本将官方 FAIL 转为作业失败。
- 计时：整体进程 wall time，含 PCM 读取、输出转换和写盘。官方输出文件使用节点本地 scratch。
- `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` 只约束 Python 对照库；**不控制 V1 的 `std::thread`**。
  V1 继承 Slurm cpuset，自己最多启动 16 个线程。
- 提交前必须创建 `tmp/logs`（Slurm 在脚本执行前打开日志）。`slurm.job` 只是同一流程的兼容入口。

## 集群 profile

```bash
ssh hepnode 'cd ~/recruitment-2026-autumn-v1 && sbatch scripts/profile.slurm'
rsync -av hepnode:recruitment-2026-autumn-v1/tmp/logs/ tmp/logs/v1-cluster/
```

集群 `perf_event_paranoid=4`，不能使用 perf 硬件计数或火焰图。
`profile.py` 在 `tmp/profile/` 生成诊断源码副本（正式源码不变），分线程累计：
加窗打包、半长 FFT、功率谱恢复、稀疏 Mel 投影，并计时 read/compute/write。
同时记录 `/usr/bin/time -v` 和诊断输出的全量正确性检查。

**各线程阶段耗时之和不是进程 wall time**；插桩会引入开销，不作为性能成绩。
源码结构变化后插桩定位会报错，需同步维护 `profile.py`，不能静默沿用。

## 本地 perf 与火焰图

```bash
git clone --depth 1 https://github.com/brendangregg/FlameGraph.git tmp/tools/FlameGraph
source scripts/env.sh
$PY scripts/perf_local.py --cases default large_fft --out tmp/perf-v1-new
```

输出目录必须尚不存在，以免覆盖上次证据。脚本以 `-O3 -march=native -pthread -g` 单独构建，
不关闭内联、不修改源码或正式二进制；全量输入，保留 V1 的 16 个线程。
`perf stat -r 3` 后用 `task-clock:u / 499 Hz / DWARF 16 KiB` 抓栈，包含子线程，
通过 `perf script --inline` 和 Brendan Gregg FlameGraph 生成 SVG。每组输出会对全量 expected 校验。

- WSL 实测 PMU 的 cycles/instructions/cache 事件不支持，不能报告 IPC/cache miss；
  `task-clock:u` 火焰图表示**用户态 CPU 样本分布**，不是 wall time 或 off-CPU 等待。
- DWARF 抓栈使用 `-m 128`（512 KiB ring/CPU），避免超过普通用户的锁页额度；
  原始 `perf.data`、栈文本、输出二进制只留 `tmp/`。
- 图中横轴不是时间线；全图合并了各线程，点击可放大，Search 可查 `FFTPlan::apply`。
- 当前 V1 的本地报告与展示用 SVG 见 `versions/V1-legacy-port/local-perf/`。

## 环境信息

集群完整复核见 `versions/_cluster/recheck-2026-10-30.md`。
本地历史环境（2026-10-30）：WSL，i7-14650HX，15 GiB 内存，AVX2/FMA、无 AVX-512，
GCC 16.2.1，Python 3.12.13，numpy 2.5.4 / scipy 1.18.1 / librosa 1.0.0。
本地虽有 FFTW 和 CUDA，但 V1 不使用；本地时间不能与集群 V0/V1 混算加速比。
