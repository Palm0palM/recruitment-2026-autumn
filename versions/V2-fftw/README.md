# V2 Agent A — FFTW 路线（实验记录）

> **第二轮 Kimi 专项实验**：已集成可选 float FFT 与安全 Wisdom 导入，默认仍保留 double A2。
> 集群五次重复对照中，float + PATIENT Wisdom 的四组中位数相对 A2 为
> **1.216× / 1.358× / 1.338× / 1.343×**，几何平均 **1.313×**。
> 这不是本地结果；完整数据、官方验收状态与失败记录见 [experiments/RESULTS.md](experiments/RESULTS.md)。
> PATIENT 最佳组合已在作业 **209** 完成官方 **4/4 PASS**，补测耗时 **3:35**；本轮已收尾。
> 目标版本针对性 Profile（作业 251，30 秒）与部署级 Wisdom 加载/回退校验已完成，见 [PROFILE.md](PROFILE.md) 第 0 节。
> 冻结清单与交付流程见 [delivery/](delivery/)。
> 原理见 [DESIGN.md](experiments/DESIGN.md)，构建及 Wisdom 准备见 [REPRODUCE.md](experiments/REPRODUCE.md)。
> 下文保留第一轮 all-double A2 的历史交付记录，不把两轮数据混算。

- Worktree：`melspec-v2-fftw`；分支 `v2/fftw`。
- 严格共同 BASE：`15f90558b4ab83ad0e6500e84aa807a942564229`。
- 不修改 `src/main.cpp`、`validate.py`、`data`；double、周期 Hann、HTK/Slaney、稀疏 Mel 和 V1 线程划分不变。
- AI：OpenCode / GPT-6.1 Sol；实现、测试、测量与报告均在本分支。

## 交付状态：A2 ESTIMATE 已通过集群正式验证

- 正式候选代码 commit：`b4552f3962751b5a02d2aa56798907c5a1c29afc`。
- A1 首次实现 commit：`a7a0752`；A2 首次实现 commit：`6c9f87a`；均独立保留，没有合并 master。
- 候选参数：`FFT_MODE=r2c FFTW_PLAN=estimate`，保持 double 和外层 16 个 worker。
- 远端独立目录：`/home/zsx/melspec-v2-fftw-agent-a`，没有覆盖 V1 或其他实验。
- 作业 56 构建依赖（3:13），58 benchmark + 官方验证（7:06），59 单独 profile（0:10），
  全部 `COMPLETED / 0:0`，依次执行；节点 hepnode2，cpuset `0-15`，Xeon Gold 6338。
- **完整官方 validate.py：四组 4/4 PASS**；同作业内五版本交错对照的所有 60 次完整输出也均 PASS。

### 官方计时（每组 3 次取最小）

| 用例 | A2 ESTIMATE 全进程（s） | vs 同次官方 librosa | max_abs_err | 状态 |
|---|---:|---:|---:|---|
| default | **0.985** | 15.472× | 0.03125 | PASS |
| small_mels | **0.650** | 22.859× | 0.01171875 | PASS |
| large_fft | **1.356** | 22.783× | 0.125 | PASS |
| narrow_band | **1.300** | 23.900× | 0.0625 | PASS |

### 相对 V1 的公平对照（同作业、同输入/编译器/线程/输出位置，三次取最小）

| 用例 | 新编译 BASE/V1 | A1 EST | A1 MEASURE | A2 EST | A2 MEASURE | A2 EST vs V1 |
|---|---:|---:|---:|---:|---:|---:|
| default | 1.539891 | 1.104407 | 1.591448 | **1.003307** | 1.372112 | **1.535×** |
| small_mels | 1.209326 | 0.758823 | 1.262902 | **0.662364** | 1.036879 | **1.826×** |
| large_fft | 2.655534 | 1.596253 | 2.260757 | **1.365753** | 1.903017 | **1.944×** |
| narrow_band | 2.597472 | 1.544488 | 2.217947 | **1.310244** | 1.844392 | **1.982×** |

这张表来自 `fftw_bench.py`，不是上张官方表；两种 harness 的进程计时有约 1–2% 差异，
不将不同 harness 的最小值混算，不与旧 hepnode3/V1 或本地秒数混算。
原始每次 wall time、median、误差见 `raw/cluster/compare-58.json` 与 `fftw-58.out`。
A2 EST 的 median 同样优于 V1/A1；不是仅靠某一个异常最小值获胜。

### 构建身份

- 编译器：GCC/G++ 15.2.0，应用 `-std=c++17 -O3 -march=native -pthread`，无 fast-math。
- FFTW 3.3.11：double static、SSE2/AVX/AVX2/AVX-512 codelets，threads/omp/mpi 均关闭；
  启用某类 codelet 不代表每个 plan 都选择该指令集。
- 库 SHA-256：`c487eddeb264013f9f051af9fb15ce66d60fc6d15d32ff7eb31501f593554ef5`。
- 候选可执行文件 SHA-256：`f1ab166c877e3921ed1a1d330a69ca542e243cd69eaef9184443662e7eb88e64`。
- 算法文件 SHA-256：`b43d32ec26380a411a995ada96ca65fedfee0e920ee0a0d2ccc79aab4ded53b5`；
  头文件/Makefile/其他二进制哈希也在原始日志中。
- 未借用其他环境的 BLAS/FFTW .so；静态 FFTW 链接行、构建配置与源码校验均已记录。

新瓶颈和 V3 建议见 [PROFILE.md](PROFILE.md)。下面保留设计与本地实验过程。

## A1：半长 C2C

只将手写 FFT 换成 `fftw_plan_dft_1d(n_fft/2)`；保留奇偶打包、共轭恢复与稀疏 Mel。
每次 `compute_melspectrogram` 只创建一个 plan，所有 worker 用 new-array execute 共享只读 plan。
每线程独立 `fftw_malloc` 缓冲，满足 FFTW 的相同布局/对齐约束；只有外层最多 16 线程。
不链接 FFTW threads/omp 后端。规划/销毁串行并加 mutex，worker 只执行 FFT，不调用 planner。
所有输入在规划完成后填充，以免 `FFTW_MEASURE` 破坏输入。保留单点 FFT 的题意边界。

```bash
source tmp/session-env.sh
make EXEC=tmp/bin/a1-estimate FFT_MODE=c2c FFTW_PLAN=estimate -j4
make EXEC=tmp/bin/a1-measure FFT_MODE=c2c FFTW_PLAN=measure -j4
$PY scripts/smoke.py tmp/bin/a1-estimate
$PY scripts/smoke.py tmp/bin/a1-measure
```

初次有效阶段：本地两种 planner 各 8 项 smoke 全 PASS（随机 PCM 的官方四组、CLI 默认值、
`n_fft=1/2/8`）；原始输出见 `raw/a1-*-smoke.txt`。此时尚未测全量性能，不据此宣称收益。

本地已安装 FFTW `3.3.11-1`（pkg-config 3.3.11）。集群不可假定预装库，
通过 `scripts/build_fftw.sh` 无 sudo 自建静态 double 库，依赖均放本实验的 `tmp/`。
编译程序时传 `FFTW_PREFIX=/absolute/path/to/tmp/opt/fftw-3.3.11`，避免运行时 .so 搜索路径依赖。
tarball SHA-256：`5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1`。
默认 `-O3 -march=native`，SSE2/AVX/AVX2，在支持的目标机上启用 AVX-512；不启用 FFTW 内部线程。
源代码/库不提交 Git；FFTW 为 GPL-2.0-or-later，分发链接后的可执行文件时须遵循 GPL、
提供相应完整源码/构建材料并保留许可和版权说明；本实验不重新授权原题代码，也不擅自发布二进制。

## A2：原生 R2C

保留相同的窗口、Mel 与线程分区；每线程独立对齐的 n_fft 个 double 输入和 n_fft/2+1 个
复数输出，使用一个 out-of-place R2C plan 的 new-array execute。直接模平方，移除手工共轭恢复。
V1 的 n_fft=1 周期窗为零的特殊处理保留，避免 scipy 单点 Hann 特例改变语义。
`compute_rfft` 公共辅助接口同样支持 R2C（不修改头文件或 CLI）。

```bash
make EXEC=tmp/bin/a2-estimate FFT_MODE=r2c FFTW_PLAN=estimate -j4
make EXEC=tmp/bin/a2-measure FFT_MODE=r2c FFTW_PLAN=measure -j4
$PY scripts/smoke.py tmp/bin/a2-estimate
$PY scripts/smoke.py tmp/bin/a2-measure
```

A1 独立代码提交 `a7a0752`；A2 增量提交后仍可通过 `FFT_MODE=c2c` 独立构建 A1。
两条路线与两种规划方式的对象目录分开，防止复用不匹配的目标文件。
额外修正 V1 的非法短输入边界：先检查 N>=n_fft 与 hop>0，再算 n_frames，
避免 C++ 向零取整使 N=n_fft-1 被误判成一帧；不改变任何合法输入的数学语义。
本地 A2 两种 planner 各 8 项 smoke 全 PASS。进一步将 A1/A2 × 两种 planner 与独立编译的 BASE/V1
比较：各 39 组随机/零值/满幅正负/冲激/Nyquist/精确帧边界合法输入均 PASS，另各 3 组短输入安全拒绝。
最大输出差为 `3.0517578125e-05`，容差仍为题目 `rtol=1e-3, atol=1e-4`。
ESTIMATE/MEASURE 对照、正式集群验证均已完成；失败实验同样保留。

## 首轮本地全量对照（有效阶段）

同 BASE/V1，自行编译，统一 `-std=c++17 -O3 -march=native -pthread`；A1/A2 另链接本地系统 FFTW。
全部五种二进制、四组、各三次输出均通过官方 expected，保留每轮数据，以下为进程 wall time 最小值。

| 用例 | V1 | A1 ESTIMATE | A1 MEASURE | A2 ESTIMATE | A2 MEASURE | A2 EST vs V1 |
|---|---:|---:|---:|---:|---:|---:|
| default | 1.219841 | 0.803781 | 1.013315 | **0.738040** | 0.930709 | 1.653× |
| small_mels | 0.994249 | 0.574242 | 0.786273 | **0.474060** | 0.704391 | 2.097× |
| large_fft | 3.127328 | 1.906353 | 1.987124 | **1.618784** | 1.972875 | 1.932× |
| narrow_band | 3.130435 | 1.813475 | 1.885710 | **1.527347** | 1.727116 | 2.050× |

暂选 **A2 ESTIMATE** 进行正式验证：MEASURE 在本题单次 CLI 的总时间口径下反而更慢，不能只看纯 FFT 执行。
本地 WSL 的频率/宿主机负载会扰动测量，结论必须回集群同作业内复核，不混算跨平台加速比。
原始逐次数据与 min/median 见 `raw/local-compare.txt`、`raw/local-compare.json`。

## 独立插桩构建

正式版的 `FFTW_*` 计时宏全部为空；定义 `MELSPEC_PROFILE` 才记录 planner、compute 准备及
每线程 pack / FFT execute / 恢复或模平方 / Mel 时间。宏不控制算法选择，不作为正式成绩。
`scripts/fftw_profile.py` 在 tmp 生成只添加调用计时的 main 副本，不修改受保护 main；另行编译并全量核对输出。
集群以 `MODE=profile` 单独提交作业，绝不在 perf 不可用的节点伪造硬件计数。
本地诊断实测：ESTIMATE 的 planner 约 0.0004–0.0013 s，MEASURE 约 0.267–0.371 s；
部分执行能变快，但本题每个 CLI 只处理一次输入，规划开销实际计入成绩，不使用预导入 wisdom。
保留 local phase 日志，不以单次插桩 wall time 代替上表 min。
添加计时宏后，关闭插桩的 A2 ESTIMATE 二进制 SHA-256 前后相同：
`97594fa7f89161aae8e41bb086dee0736ec2839cf83f59556947188c0f57720c`，8 项 smoke 复核 PASS。

失败尝试：WSL 未安装 GNU `/usr/bin/time`，初次本地阶段 profile 因此失败。工具已改为自动检测，
缺失时仍计时 read/compute/write 和进程 wall，但明确不报告 RSS；集群有 GNU time 才采集相应统计。
原始失败日志见 `raw/failed-profile-gnu-time-missing.txt`，不为此修改系统或共享 Python 环境。

## 评测流程

`scripts/fftw_bench.py` 在同一环境以轮换顺序运行 V1、A1/A2 与两种 planner，保留每次 wall time、
min/median、对同次 V1 的加速比，每次输出都与官方全量 expected 校验。输入/输出 I/O、规划、
线程创建均计入进程时间；不导入/计时 librosa，不把本地结果当作集群成绩。

`scripts/fftw-cluster.slurm` 为本实验独立作业：1 task × 16 CPU、nomultithread、不写 mem。
MODE=build 仅在计算节点构建静态 FFTW；默认先做五种二进制的全量三次对比，再对 A2 ESTIMATE
运行完整官方 `validate.py --runs 3`。中间输出在节点本地 scratch，源码、库和日志留独立 home 目录。
依赖构建日志、源码/库/程序哈希和 cpuset 均输出；只串行提交本实验作业，不跑 V0。

资料：FFTW 官方 [下载](https://www.fftw.org/download.html)、
[线程安全](https://www.fftw.org/fftw3_doc/Thread-safety.html)、
[new-array execute](https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html)、
[规划标志](https://www.fftw.org/fftw3_doc/Planner-Flags.html)。
