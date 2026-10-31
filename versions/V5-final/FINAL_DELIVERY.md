# V5 Final — 最终交付

完成 F1–F4；L40、R9700 两个平台均完成正确性、冻结 A/B、独立回归与启动边界验证。

## 1. 交付建议

**同一套源码，按硬件分别构建。优先保证可交付性与正确性，不加载两套 GPU Runtime。**

| 部署 | 建议构建 | 启动/分派策略 |
|---|---|---|
| 无 GPU、最低 CPU 启动成本 | CPU-only | 无 CUDA/HIP DSO；完整 SIMD B16/B8/FP64 回退 |
| L40 / NVIDIA | CPU+CUDA | 默认 auto，按重新标定的成本模型选择；保留强制 cpu/cuda |
| R9700 / 输入规模多变 | CPU+HIP | 默认 auto，HIP 队列仅在选中 HIP 后预热；CPU 不启动预热线程 |
| R9700 / 大型固定任务、追求最低 GPU 整程耗时 | AMD dedicated | 保留冻结 AMD 的 main 前预热；默认 auto=HIP，不提供 CPU 分派 |

**AMD 两种构建不能混写成绩。** 懒初始化 HIP 比 dedicated 多约 180–200 ms，
这是在不改受保护 main 的前提下，避免 CPU 路径无条件预热 GPU 的明确代价。
专用构建保留了原 AMD 最优流水线；它不影响 CPU+CUDA 稳定交付。

## 2. 后端来源与冻结边界

| 部件 | 来源 | 采用内容 / 不改变的部分 |
|---|---|---|
| 架构基础 | origin/v4/hybrid `af56878` | 唯一公开 Dispatcher，私有完整后端入口 |
| CPU 数学核心 | v2/simd `e448c989` | FP32 B16 AVX-512、B8 AVX2/FMA、FP64 回退；FFT 核心不重写 |
| CPU 升级 | exp/cpu-pipeline-simd `e14647f`，源码 `cbf214d` | 仅 Slab Reuse、合法 THP 提示、空 batch 引用安全修复 |
| NVIDIA | v3/cuda `3e111946` | FP32/B512/Fused/Context Overlap/Async Output/Resize Thread/Staging16 |
| AMD | v3/amd `eff3ed1` | gfx1201、FP32 hipFFT 512/256、Fused、Async H2D、Direct Double、THP、Resize Thread |

- 未 merge/cherry-pick 历史分支，未引入 MPI，未重新研究 FFT。
- CUDA 只调整初始化错误分类/分派适配器：去掉串行 cudaGetDeviceCount 预探测，
  保留原有重叠初始化。未移植 cuda-sprint 的后台设备分配或 GPU 输入转换。
- HIP 只调整预热触发时机与线程错误传播；数值 kernel、FFT、传输/输出流程保留。
- `src/main.cpp`、`validate.py`、`data`、`include/melspec.h` 均未修改。
- 自动源码审计：`scripts/final_source_audit.py` 验证 CPU 与 cbf214d 数值 token 一致，
  CUDA 去掉明确适配器后与 3e111946 字节一致，HIP kernel/helper 字节一致，计算体仅预热/诊断适配差异。

| 平台 | 硬件与运行环境 | 实际绑核 |
|---|---|---|
| NVIDIA | hepnode2，Xeon Gold 6338，NUMA=4、SMT=1，L40 sm89，driver 595.91.07，CUDA 13.4.92 | 0–15，NUMA0 |
| AMD | hepnode0，Xeon Gold 6338，NUMA=2、SMT=2，R9700 gfx1201，私有 ROCm 7.1.1 / LLVM21 | 0–15,64–79（16 核含 SMT siblings） |

两平台均为 16 个 CPU worker、Linux THP `[madvise]`。同一平台各候选使用相同分配与绑核；
两个节点的 NUMA/SMT 不同，因此不跨节点复用 CPU 系数或计算硬件加速比。

## 3. 构建方法

构建/日志/缓存放 `tmp/`。共享 Python、PCM、expected、工具链只读；不运行 setup.sh。
CPU 的两项优化默认开启；`CPU_PACK_REUSE=0` / `CPU_HUGEPAGE=0` 可用于消融，
但关闭后必须重新标定 auto，不应沿用当前性能系数。

```bash
mkdir -p tmp/bin tmp/logs

# CPU-only；-march=native 自动选择 B16 / B8 / FP64
make CUDA=0 HIP=0 -B -j8 EXEC=tmp/bin/melspec-cpu

# 通用 x86-64 回退（需可移植 CPU 二进制时）
make CUDA=0 HIP=0 -B -j8 EXEC=tmp/bin/melspec-generic \
  CXXFLAGS='-std=c++17 -O3 -march=x86-64 -pthread'

# L40：私有 CUDA 13.4.92，sm_89
export CUDA_HOME=/home/zsx/melspec-v3-cuda-agent/toolkit/cuda-13.4
export CUDA_CACHE_PATH="$PWD/tmp/cuda-cache"
make CUDA=1 HIP=0 -B -j8 CUDA_HOME="$CUDA_HOME" CUDA_ARCH=89 \
  EXEC=tmp/bin/melspec-cuda

# R9700：既有只读 ROCm 7.1.1 工具链，不安装/覆盖其他实验环境
source /home/zsx/melspec-v3-amd/tmp/amd-gpu/env.sh
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/warm.db"
export AMD_COMGR_CACHE_DIR="$PWD/tmp/comgr-cache"
mkdir -p tmp/rocfft-cache tmp/comgr-cache
make CUDA=0 HIP=1 -B -j8 EXEC=tmp/bin/melspec-hip
make CUDA=0 HIP=1 HIP_ONLY=1 -B -j8 EXEC=tmp/bin/melspec-amd
```

HIP auto 只在显式 `ROCFFT_RTC_CACHE_PATH` 指向**非空普通文件**时使用暖缓存 GPU 模型；
缺失/未准备缓存时留在 CPU，不为探测缓存而试跑 GPU。显式 HIP/专用版不受此保护限制。
首次部署可用强制 HIP 正常计算目标配置以生成缓存，例如：

```bash
for fft in 8192 16384 32768 65536; do
  MELSPEC_BACKEND=hip tmp/bin/melspec-hip data/audio.pcm tmp/cache-bootstrap.bin \
    --sr 96000 --n_fft "$fft" --hop_length 128 --n_mels 256 --f_min 0 --f_max 22050
done
```

这是显式离线缓存准备，不是 Dispatcher 内的选择试跑。文件存在性仅是必要条件，
不证明所有 FFT/batch/版本的计划已缓存；新配置仍需准备自己的相应计划。

CUDA+HIP 同时启用会明确报构建错误。CPU+HIP 禁止开启全局 eager prewarm；
只有 HIP_ONLY=1 可启用。GPU 构建不是无 GPU 库的通用二进制：缺少 Runtime DSO
时应部署 CPU-only，而不是指望动态加载器也能执行 CPU fallback。

## 4. 分派与错误语义

| MELSPEC_BACKEND | CPU-only | CPU+CUDA | CPU+HIP | AMD dedicated |
|---|---|---|---|---|
| 未设 / auto | CPU | 模型选择 CPU/CUDA | 模型选择 CPU/HIP | HIP（保留全局预热） |
| cpu | CPU | CPU | CPU | 明确报错，改用 CPU+HIP/CPU-only |
| cuda | 明确报错 | CUDA | 明确报错 | 明确报错 |
| hip | 明确报错 | 明确报错 | HIP | HIP |

一次调用只运行一个完整计算后端，不同时在 CPU/GPU 计算同一份任务，不做试跑。
选择为 GPU、但在初始化时确认无设备/驱动不足时，auto 可在任何数值 GPU kernel 前回退 CPU。
只有专用 `BackendUnavailable` 类型可回退；内存分配、FFT、Kernel、传输及其他初始化错误
向上传播。强制 GPU 不捕获该异常，明确报错。CPU 决策不调用 GPU API。
`n_fft=1` 的冻结零输出特例在 auto 中留在 CPU；强制 GPU 仍检查设备存在。

```bash
MELSPEC_BACKEND=cpu  tmp/bin/melspec-cuda data/audio.pcm tmp/out.bin [原 CLI 参数]
MELSPEC_BACKEND=cuda tmp/bin/melspec-cuda data/audio.pcm tmp/out.bin [原 CLI 参数]
MELSPEC_BACKEND=hip  tmp/bin/melspec-hip  data/audio.pcm tmp/out.bin [原 CLI 参数]
MELSPEC_DISPATCH_LOG=1 tmp/bin/melspec-cuda data/audio.pcm tmp/out.bin [原 CLI 参数]
```

模型特征（只看规模与部署画像，不看用例名/音频内容）：

```
W = frames*n_fft*log2(n_fft)/1e8
I = input_pcm_bytes/1e6
O = frames*n_mels*4/1e6
C = W*max(0, n_fft/16384 - 1)
Tbackend = bias + a*W + b*I + c*O + d*C
```

C 是通过 32K/64K 实测引入的工作集压力代理，不宣称测得了 cache miss。
平台系数分别在 `include/melspec_model_cuda.h`、`include/melspec_model_hip.h`；
训练/留一验证：CUDA 43/44、43/44；HIP 44/44、43/44。独立 holdout 不参与拟合。
详见 [F2_CUDA.md](F2_CUDA.md)、[F3_HIP.md](F3_HIP.md)。

## 5. 同环境 A/B 与最佳成绩

### F2 独立稳定检查点

代码/模型 `8e1659a`，独立证据 `45b4093`（后续附加官方日志）。
Job 425，L40 节点同作业 6 轮交错中位数：

| 用例 | 冻结 B16 | 升级 CPU | 冻结 CUDA | 整合 CUDA |
|---|---:|---:|---:|---:|
| default | .8314 | .6676 | .7990 | .8013 |
| small_mels | .4847 | .4230 | .5391 | .5450 |
| large_fft | 1.0322 | .8508 | .8020 | .8125 |
| narrow_band | 1.0054 | .8079 | .8041 | .8157 |

CPU 减时 12.7%–19.7%；CUDA 适配器差距 .3%–1.4%。Job 435 独立 12 点全部 PASS，
12/12 选择与当场最快后端一致：auto 5.162 s、始终 CPU 5.288 s、始终 CUDA 6.708 s。
官方 validate.py --runs 3：三模式各 4/4 PASS，auto 最小值 .655/.423/.789/.791 s。

### F3 AMD 构建对照

Job 436，同作业 6 轮中位数：

| 用例 | 升级 CPU | 懒初始化 HIP | 专用预热 HIP | 冻结 AMD |
|---|---:|---:|---:|---:|
| default | .6959 | .8804 | .6947 | .6953 |
| small_mels | .4415 | .6435 | .4633 | .4584 |
| large_fft | .8942 | .9345 | .7521 | .7507 |
| narrow_band | .8514 | .9253 | .7469 | .7437 |

AMD 专用构建保留原成绩；不能用这列替代 CPU+HIP 的性能。
所有数字均完整进程 wall，包含 PCM 读入、初始化、FFT/Mel、转换与写盘，非 Kernel 时间。
不同节点之间不算硬件胜负或加速比。

### F4 最终复核

作业 443（L40）→444（AMD），串行依赖。代码版本 `c00b3eb`。

下表未注明 CPU-only 的 CPU 列均为统一二进制强制 CPU，包含对应 GPU DSO 装载成本。
官方四组 6 轮的 min/median（不是跨作业挑选最小值），便于交付复核：

| 构建/模式（作业） | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| CPU-only（425） | .6553/.6615 | .4157/.4206 | .8316/.8459 | .8011/.8112 |
| L40 auto（443 额外 A/B） | .6632/.6687 | .4214/.4250 | .7816/.8047 | .7897/.8052 |
| AMD dedicated（444 holdout） | .6863/.6972 | .4495/.4615 | .7426/.7603 | .7404/.7492 |

每格 `min / median`，秒。跨节点不从此表计算 GPU 胜负；严格收益只取下述同作业内 A/B。

#### L40：Job 443（最终源码、独立 14 点、6 轮）

| 指标 | 结果 |
|---|---|
| 正确性 | 14/14 点，每轮每个候选都校验 PASS；官方三模式各 4/4 PASS |
| 与当场最快后端一致 | 14/14 |
| 始终 CPU / 始终 CUDA / auto | 6.4055 / 7.3426 / **5.7393 s** |
| 逐点最优参考值 | 5.7166 s，auto 计时差距 **+0.40%** |
| CPU-only / unified 强制 CPU / auto 小任务启动中位 | 7.68 / 12.47 / 11.22 ms |
| CPU / 小任务 auto 的 GPU 设备访问行数 | **0 / 0**（强制 CUDA 为 46；完整 strace 已保留） |
| 真实隐藏 GPU | auto 输出与 CPU 逐字节一致；强制 CUDA exit=1，明确不可用错误 |

同作业内另做官方五候选交错 A/B，中位数（最小值/所有原始值见 raw/f4-cuda）：

| 用例 | 冻结 B16 | 最终 CPU | 冻结 CUDA | 最终 CUDA | 最终 auto | 选择 |
|---|---:|---:|---:|---:|---:|---|
| default | .8358 | .6652 | .8039 | .8058 | .6687 | CPU |
| small_mels | .4873 | .4216 | .5453 | .5426 | .4250 | CPU |
| large_fft | 1.0376 | .8506 | .8020 | .8124 | .8047 | CUDA |
| narrow_band | 1.0044 | .8102 | .7986 | .8192 | .8052 | CUDA |

这个额外 A/B 中 narrow_band 的强制 CPU 比强制 CUDA 快 1.12%，故逐点命中为 3/4；
独立 holdout 的同一点反过来是 CUDA 快。保留两组数据，不为它添加用例特例。
auto 偶尔快于分别计时的 oracle 是进程间噪声，不是更快的第三种算法。
总体 GPU 适配器中位累计约 +1.02%，CPU 相对冻结版约减时 18.4%。

CPU 和 CUDA 插桩 profile 的四组输出分别通过校验。CUDA default 的诊断为
y_convert 51.8 ms、gather 6.9 ms、FFT 7.4 ms、Mel 10.3 ms、异步输出区间 59.0 ms、
compute_total 273.3 ms；剩余时间主要在受保护 main 的 I/O/转换及进程生命周期。
这些 profile 含同步/插桩，不作正式成绩，不把线程和/重叠区间直接相加。

#### AMD：Job 444

auto、强制 CPU、强制 HIP、AMD dedicated 均通过原 `validate.py` **4/4 PASS**。
独立 14 点、6 轮、四个候选每次输出均检查，通过 **14/14**。

| 指标 | 结果 |
|---|---|
| 始终 CPU / 始终懒初始化 HIP / auto / dedicated | 6.1565 / 8.6871 / **6.1078** / 7.2999 s |
| 与当场最快 CPU/HIP 一致 | **13/14**，唯一误判代价 **2.52%** |
| CPU-only / unified 强制 CPU / auto 小任务启动中位 | 5.43 / 18.10 / 17.03 ms |
| CPU / 小任务 auto 的 GPU 设备访问行数 | **0 / 0**（强制 HIP 为 4；完整 strace 已保留） |
| 真实隐藏 GPU | auto 与 CPU 逐字节一致；强制 HIP exit=1，明确不可用错误 |

这组 holdout 全部点的强制 CPU 恰好更快；auto 比 CPU 累计低 .79% 是独立计时噪声，
不宣称算法收益。唯一误判在 16K FFT、230003 帧、256 mels：CPU 1.0023 s、HIP 1.0275 s。
另有已验证的 GPU 优势区：训练集中 64K FFT、32768/65536 帧分别 CPU .7136/1.3030 s、
懒 HIP .5168/.6982 s；真实启动回归也确认该 GPU 优势点确实选择 HIP。

官方四组最终中位数（同一独立回归，非跨作业最佳值）：

| 用例 | CPU | 懒 HIP | auto | dedicated |
|---|---:|---:|---:|---:|
| default | .7042 | .8791 | .6954 | .6972 |
| small_mels | .4466 | .6449 | .4431 | .4615 |
| large_fft | .8898 | .9293 | .8797 | .7603 |
| narrow_band | .8536 | .9260 | .8459 | .7492 |

不能把 dedicated 的大型任务收益算给混合 auto；也不能把多数小任务组成的累计耗时
当作生产负载分布。两种构建均交付，按实际部署目标选择。

**冷缓存验证（独立新数据库）**：8192 FFT 首次进程 2.5855 s，暖后 .6985 s；
16384 FFT 首次 2.0446 s，暖后 .7456 s。暖缓存模型不能直接用于冷 JIT。
因此最终修订 `4227f51` 加入上述缓存准备保护。Job 445 对最终修订复核：
官方四组/四候选每轮通过；未配置缓存时 auto 与 CPU 逐字节一致且不调用 GPU；
暖缓存 GPU 优势点与真实无设备回退也通过。

Job 445 另修正冻结 profiler 的 async_h2d/staging_rows 参数标签顺序，仅影响诊断，
不影响 PROFILE=0 的计算；原始错误标签日志保留。修正后 default 的诊断：
懒 HIP `prewarm_wait=171.45 ms, compute_total=387.86 ms`，
dedicated `prewarm_wait=11.46 ms, compute_total=215.33 ms`，两者输出均校验通过。
这些是插桩诊断，非正式成绩。

收尾 HIP-only 保护未进入 CPU-only / CPU+CUDA：两种 Dispatcher 的预处理结果与
F4 `c00b3eb` 完全一致，CUDA.cu SHA 不变（见 raw/final-compatibility.json）。

## 6. 正确性与安全证据

- CPU B16/B8：68 组合成、独立 librosa；generic/B8/B16 严格 ASan/UBSan 24 组
  arbitrary-double / reused-vector 接口测试均通过（F1/F2）。
- CUDA：本地 RTX 5060 sm120 强制后端 68 组合成通过；L40 44 点每次输出均校验，
  独立 12 点与官方三模式全部通过。
- HIP：R9700 强制后端 68 组通过；44 点与官方 A/B 每轮均校验通过。
- 允许原官方逐元素 `rtol=1e-3, atol=1e-4`，未放宽；组合 CPU 的近零 FP64 尾帧
  不承诺 bit-exact，沿用源实验已解释的舍入限制。
- Dispatcher 合约单测：CPU/auto 小任务零 GPU API；仅不可用初始化可回退；
  注入真实计算类型 runtime_error 不回退；非法参数在分派算术前拒绝。
- 实际 CUDA 无设备测试已通过，auto 输出与 CPU 逐字节一致，强制 GPU 非零退出；
  AMD 目标节点同样通过。F4 保存了运行时验证与完整设备访问 strace。

## 7. 复现与证据目录

```bash
# 提取 Git 冻结控制到 tmp，不使用其他 Worktree 的构建产物
bash scripts/final_prepare.sh
# 如果使用 --single-branch clone 导致源提交不可用，先获取来源再重试：
# git fetch origin exp/cpu-pipeline-simd v3/cuda v3/amd v4/hybrid

# 仅在各自计算节点上编译/测试（需先建立 tmp/logs，并提供共享只读 Python/data）
sbatch -w hepnode2 scripts/final_cuda.slurm                 # MODE=train，训练/A-B
MODE=final sbatch -w hepnode2 scripts/final_cuda.slurm     # 独立回归/官方/profile
sbatch scripts/final_amd.slurm                            # hepnode0，无 GPU GRES
MODE=validate sbatch scripts/final_amd.slurm              # 先完成 AMD train
# 最终缓存保护、不可用路径、修正诊断的快速回归：
sbatch scripts/final_hip_guard.slurm

# 从保留的训练数据重建模型；不会读取 holdout
python scripts/final_model.py versions/V5-final/raw/f2/f2-training-425.json \
  --gpu cuda --header include/melspec_model_cuda.h --out tmp/cuda-model.json
python scripts/final_model.py versions/V5-final/raw/f3/f3-training-436.json \
  --gpu hip --header include/melspec_model_hip.h --out tmp/hip-model.json

python scripts/final_source_audit.py --out tmp/source-audit.json
python scripts/final_summary.py RAW.json --gpu cuda        # HIP 时换 --gpu hip
```

| 路径 | 内容 |
|---|---|
| raw/f1/ | 选择性补丁、回退/接口/sanitizer |
| raw/diagnostic-zfs/ | 首轮 home/ZFS、未绑定诊断，不作标定依据 |
| raw/f2-preprobe/ | 旧串行探测 A/B，与移除探测的适配器差异 |
| raw/f2/、raw/f2-holdout/ | L40 训练、冻结对照、独立回归、官方与 CPU profile |
| raw/f3/ | AMD 训练、独立合成正确性、懒初始化/专用/冻结对照 |
| raw/f4-local/ | 本地最终运行时启动/不可用回归 |
| raw/f4-cuda/ | 最终 L40 holdout、冻结 A/B、官方、启动 strace 与 CPU/CUDA profile |
| raw/f4-amd/ | 最终 AMD holdout、官方、冷暖缓存、启动 strace、懒/专用 profile |
| raw/f4-guard/ | 最终缓存保护、真实不可用、官方回归、修正后的 profile |
| raw/provenance/ | 冻结 AMD 原有运行时警告等原始证据，不用于新成绩加速比 |

JSON 保留逐次时间、最小值、中位数、执行顺序、逐次校验、后端决策；meta 保留主机/绑核/
环境/候选哈希。Slurm 输出记录源码 SHA、编译器、驱动、GPU 和 THP 状态。
运行时/工具链/PCM/expected/数据库/二进制不提交 Git。

## 8. 部署限制

1. 模型针对单进程一次性调用、16 个 CPU worker、目标绑核与 Linux THP 标定。
   换 CPU/GPU/驱动/编译选项、THP、存储或常驻服务复用 context/plan 后必须重测。
   直接嵌入 API、复用输入/输出容量时也不同于受保护 CLI 的完整进程口径。
2. ROCm 的持久 rocFFT JIT 缓存是低延迟前提；新环境/新 plan 先生成自己的缓存，
   不把冷缓存首次 JIT 结果当作稳定成绩，也不跨 ROCm/LLVM/gfx 复用数据库。
3. CPU+GPU 二进制会加载相应 Runtime DSO，虽不一定建立设备队列，仍有启动成本；
   极小任务或无 GPU 运行库环境使用 CPU-only。
4. 工作集特征是测量模型，非逐点 oracle；交叉点附近会受预测残差、CPU 频率与内存状态影响。
   不按官方用例名修正选择。训练集统计与最终 holdout 统计严格分开。
5. 大任务显存不足等真实计算错误明确失败，不静默改为 CPU 重算。未验证范围不能承诺最优。
6. ROCm 7.1.1 短进程退出时仍有 `SharedSignalPool, 77 Signals leaked` 警告。
   原冻结 AMD Job 395 的 profile/日志已有完全相同警告，已保留对照，未宣称修复。
   本轮不引入未经 A/B 的库/流生命周期重写；数值验证、退出码与 CPU 路径均通过。
   常驻服务/长期资源生命周期不属于本次已验证交付范围。

## 9. 最终状态与源码检查点

| 阶段 | 代码检查点 | 实机证据 |
|---|---|---|
| F1 | acfe1c6 | CPU B8/generic 本地，B16 Jobs 419/425 |
| F2 独立稳定交付 | 8e1659a、45b4093 | Jobs 425/435 |
| F3 HIP 适配与标定 | 04962ce、c00b3eb | Job 436 |
| F4 两平台终验 | c00b3eb | Jobs 443/444 |
| HIP 缓存保护/诊断修正 | 4227f51 | Job 445；CPU/CUDA 预处理单元不变 |

正式计分只用未插桩二进制；生成 profile 的程序另行构建。模型 header 已从保留训练数据
重新生成并逐字节比对一致。受保护文件与冻结数学源码审计通过。源工作树保持只读，
工作分支为 `v5/final`；最终源码、脚本、文档与全部文本/JSON 证据均留在该分支。
