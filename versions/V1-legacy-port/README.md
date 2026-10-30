# V1 — 迁移旧招新赛最终解答

## 范围与代码来源

- 起点：笔试仓库 `fae5cac`；本次未自动 commit/tag，源码状态以 SHA-256 为准。
- 来源：`../autumn26/melspec/src/melspec.cpp`，与旧档案 `versions/melspec_v3_multithread.cpp` 完全一致。
- V1 的 `src/melspec.cpp` **逐字节原样迁移**，SHA-256：
  `3b00a2d777a20de397c3a2e407b60f6a21258940470851cbeb82c4c21c64f957`。
- 相对 V0 的算法文件 diff：`src/melspec.cpp | 160 行，110 insertions(+), 50 deletions(-)`。
- `src/main.cpp`、`include/melspec.h`、`validate.py`、`data` 未改动；两题接口相同。
- 旧版文档报告的 82.30 分仅是旧赛事历史结果，不是笔试 V1 成绩。
- 使用 AI：OpenCode / GPT-6.1 Sol，承担仓库梳理、迁移、评测脚本修复、诊断与记录。

## 题目理解

读取小端 int16 PCM，归一化为 double；无中心填充、周期性 Hann；实数 FFT 的模平方；
HTK Mel 三角滤波器 + Slaney 面积归一化；输出 `[n_mels, n_frames]` 行主序 float32。
允许数学等价优化，禁止数据特征特化和音频处理库；新版允许科学计算库、多线程/GPU。
V1 不引入任何外部数学库。

README 的 CLI 默认参数仍是 2048/512/128/44100，**实际 benchmark 以不可修改的 validate.py 为准**：
`sr=96000, hop=128, n_fft=8192/16384, n_mels=256/64`，输入 24,634,880 个样本。
这意味着约 192k 帧，高重叠率会放大 FFT 的计算量。

## 迁移的算法

1. 缓存位反转索引与旋转因子，复用每线程 FFT/功率谱缓冲区。
2. 偶/奇样本打包成 `n_fft/2` 个复数，半长 FFT 后恢复正频率功率谱。
3. 每帧算完功率谱立即投影；仅遍历各 Mel 三角的非零连续区间，不保存全量功率谱。
4. 按连续帧段分配给最多 16 个 `std::thread`；短输入减少线程，保留创建失败的回退。
5. 数值类型保持 double，最终由不可改动的 main 转换为 float32。

相较 V0，主要消除了 6.3/12.6 GiB 功率谱和稠密 Mel 乘的跨行访问。
V1 并未复用重叠帧间的 FFT，也未实现 SIMD FFT/float32 化，这些留待 profile 后判断。

## 仓库清理与工作流修复

- 不回滚已有提交、不丢弃历史记录。旧目标文件、缓存及散落 scratch 移入 `tmp/archive/pre-v1/`。
- 构建中间文件迁至 `tmp/build`，补充头文件依赖；明确 `-pthread`，移除未使用的 `-fopenmp`。
- 统一 Slurm 入口和 `tmp/logs`；修复 `CASE/RUNS` 粘连，改为 1 task × 16 物理核，禁止 `--mem`。
- 修复辅助 bench 的 `data/data/expected` 路径和未使用的 `MELSPEC_DATA`；正确性失败返回非零。
- 更新脚本文档，移除“spack 可直接使用”、perf/火焰图可采集等过期说明。
- 原远端 `~/recruitment-2026-autumn` 有未提交工作，保留不动；本版同步到独立目录
  `~/recruitment-2026-autumn-v1`，不传本地二进制/对象文件，不使用 rsync `--delete`。
- 正式 benchmark 与诊断构建分离。插桩副本只生成在 `tmp/profile/`，题目源码不变。

## 本地验证

```bash
make -j4
tmp/.venv/bin/python -X pycache_prefix=tmp/pycache scripts/smoke.py
tmp/.venv/bin/python -X pycache_prefix=tmp/pycache scripts/bench.py --slice 0.5 --runs 1
```

随机 PCM 回归：官方四组、无选项 CLI 默认值、`n_fft=1/2/8` 共 8 项 PASS。
真实 PCM 前 0.5 秒四组 PASS，但前段能量很小，**随机 PCM 回归不可省略**。
切片单次计时含首次 librosa 导入/JIT，不能作为 speedup 结论。原始结果见 `raw/local-smoke.txt`。
另将诊断构建与正式构建在同一份随机 PCM 的官方四组上逐字节比较：输出全部一致，阶段计时存在。
随机回归的完整 stdout/stderr 另存为 `raw/local-smoke-random.txt`。

## 集群 benchmark / profile

### 环境与口径

- benchmark 作业 **45**，profile 作业 **47**，均在 **hepnode3**，`COMPLETED / 0:0`。
- 1 节点、1 task、16 CPU，`--hint=nomultithread`，实际 cpuset `0-15`、`nproc=16`。
- CPU：Intel Xeon Gold 6338 @ 2.00GHz；作业内 `free -g` 显示 **503 GiB** 内存，
  与早期报告泛称的“每节点约 1 TB”不同，本版以实际作业日志为准。
- GCC 15.2.0，`-std=c++17 -O3 -march=native -pthread`；计算节点 `make -B` 重新编译。
- Python 3.14.4、numpy 2.5.4、librosa 1.0.0。Python 对照线程上限 16；无外部 C++ 数学库。
- 官方 `validate.py --runs 3`：整体进程 wall time，含输入读取、float32 转换、输出写盘；
  输出文件在节点本地 `/tmp/melspec-bench-45-*`。输入是题目只读 BeeGFS 文件。
- benchmark 总作业时间 6:38，含四组各 3 次 librosa 对照与编译；不是 V1 单次计算耗时。
- 完整构建命令、源码与二进制哈希、官方表格见 [`raw/bench-45.out`](raw/bench-45.out)。
  编译仅有原始 `main.cpp` 两个未使用形参警告，见 `raw/bench-45.err`。

### 全量正式结果

| 用例 | n_fft | n_mels | 最小 wall time（3 次，s） | 同作业内 librosa / V1 | max_abs_err | 正确性 |
|---|---:|---:|---:|---:|---:|---|
| default | 8192 | 256 | **1.537** | **12.416×** | 3.12e-2 | PASS |
| small_mels | 8192 | 64 | **1.196** | **15.241×** | 1.17e-2 | PASS |
| large_fft | 16384 | 256 | **2.634** | **15.537×** | 1.25e-1 | PASS |
| narrow_band | 16384 | 256 | **2.578** | **15.695×** | 6.25e-2 | PASS |

四组完整输出均满足 `rtol=1e-3, atol=1e-4`；不能只看最大绝对误差与 `atol` 比较。
这些是本版建立的**集群优化基线**。V0 没有集群数据，不计算 V0→V1 跨平台加速比；
旧赛事评分公式也不当作本次笔试的评分承诺。

### Profile：热点已从 Mel 转移至 FFT

集群禁止 perf；作业 47 使用独立的诊断构建，各组全量输出再次全部 PASS。
完整线程日志、`time -v` 与阶段汇总见 [`raw/profile-47.out`](raw/profile-47.out)。

下表比例为**16 个线程各阶段 wall time 累计值占比**，不是整个进程的占比；
包含插桩扰动，只用于定位热点，不能替代上表正式耗时。

| 用例 | 加窗/打包 | 半长复数 FFT | 实数谱恢复/功率谱 | 稀疏 Mel |
|---|---:|---:|---:|---:|
| default | 4.57% | **75.97%** | 15.01% | 4.45% |
| small_mels | 4.42% | **75.62%** | 14.98% | 4.97% |
| large_fft | 3.92% | **78.51%** | 13.35% | 4.22% |
| narrow_band | 4.05% | **80.64%** | 13.73% | 1.59% |

诊断构建的进程阶段 wall time（单次，不能当作 benchmark）：

| 用例 | 读取/归一化（s） | compute 全部（s） | 转换/写出（s） | 峰值 RSS（KiB） |
|---|---:|---:|---:|---:|
| default | 0.176799 | 1.079710 | 0.217623 | 772664 |
| small_mels | 0.173367 | 0.924900 | 0.070355 | 339780 |
| large_fft | 0.179832 | 2.113690 | 0.245265 | 773524 |
| narrow_band | 0.176758 | 2.060420 | 0.251281 | 772704 |

结论：

1. **FFT + 功率谱恢复占线程内时间约 91–94%**。稀疏 Mel 已不是主要瓶颈，
   narrow_band 的 Mel 工作更少，但整体仍与 large_fft 接近，与此吻合。
2. 大输出组峰值 RSS 约 **755 MiB**，small_mels 约 **332 MiB**，无 swap/major page fault。
   不再物化大功率谱，基本解决了 V0 的中间矩阵内存问题。
3. default 的 compute 1.080 s 与最慢 worker 0.872 s 差约 **0.207 s**，
   其中包括计划/滤波器准备、输出与 scratch 分配初始化、线程启停等，不能都算到 FFT。
   `main.cpp` 固定的输入归一化、double 输出、float32 转换也留下不可忽视的串行/访存成本。
4. 不将单次诊断构建略快于正式最小值解释为优化收益：插桩改变代码布局，且两次作业存在运行扰动。

## 下一版设计建议（尚未实施）

**V2 主线：保持逐帧稀疏 Mel 和 16 线程分区不变，优化 FFT 与实数谱恢复。**

- 先保持 double，减少变量：将 `std::complex` radix-2 蝶形改成更利于向量化的 SoA 布局，
  尝试 radix-4/混合基数与 AVX-512；实数恢复/模平方一起向量化。
- 本地无 AVX-512，保留通用/AVX2 路径；集群才是 SIMD 收益与正确性最终验证平台。
- 可另做自带构建的 FFTW/pocketfft 对照，评估自写 FFT 的差距；不依赖集群不存在的预装库。
- float32 FFT 留作独立精度/性能实验，不能因本次 double 通过就假定 float32 必过。
- 验收：本地随机/边界 smoke → 同节点、相同参数的四组全量、3 次取最小 → 再 profile。
  记录相对 **V1 同平台基线** 的增量收益；若节点/环境不同需重测 V1 对照。
- 本轮只完成 V1 和设计，不提前混入 V2 代码。

## 集群复现命令

```bash
rsync -av --exclude=.git/ --exclude=tmp/ --exclude=__pycache__/ \
  --exclude=AGENTS.md --exclude=melspec --exclude='*.o' \
  ./ hepnode:recruitment-2026-autumn-v1/
ssh hepnode 'cd ~/recruitment-2026-autumn-v1 && mkdir -p tmp/logs && sbatch scripts/bench.slurm'
# 得到 benchmark job id 后，顺序运行诊断（本次为 45 → 47）：
ssh hepnode 'cd ~/recruitment-2026-autumn-v1 && sbatch --dependency=afterok:45 --kill-on-invalid-dep=yes scripts/profile.slurm'
rsync -av hepnode:recruitment-2026-autumn-v1/tmp/logs/ tmp/logs/v1-cluster/
```

当前通用入口说明见 `scripts/README.md`；原远端工作区、原 Git 历史均未覆盖或重写。
