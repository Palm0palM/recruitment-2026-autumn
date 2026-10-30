# 第三轮集群结果与推荐实现

日期：2026-10-30。远端专属目录：`~/recruitment-2026-autumn-v2-overlap-next`。
成功 Benchmark 作业 **199**（hepnode2），Profile 作业 **200**（hepnode2）。
默认 `src/melspec.cpp` 仍为 V1。

## 1. 环境与测量

- Xeon Gold 6338，1 task × 16 物理核、nomultithread、未设置 `--mem`。
- GCC 15.2.0；官方 `validate.py --runs 3`，四组全量、三次进程 wall-time 最小值。
- 私有 FFTW 3.3.10 源码构建，启用 AVX2/AVX-512，静态链接 `libfftw3.a`；
  不使用其他 Agent 的库或工作目录。
- 候选：V1、冻结 half+delta、P0、FFTW、FFTW+P0。
- 作业 199 总时长 25:23，含五份官方 librosa 对照，不是单次程序时间。
- 作业 182 因 half-delta 二进制路径写错而失败，未产生候选端到端结论；
  修正后由 199 完成，失败日志保留。

## 2. 官方全量结果（秒）

| 用例 | V1 | half+delta | P0 | FFTW | FFTW+P0 |
|---|---:|---:|---:|---:|---:|
| default | 1.541 | 1.380 | 1.296 | 1.177 | **1.089** |
| small_mels | 1.199 | 1.052 | 0.964 | 0.843 | **0.766** |
| large_fft | 2.644 | 2.134 | 1.959 | 1.729 | **1.537** |
| narrow_band | 2.591 | 2.080 | 1.779 | 1.690 | **1.357** |

五个版本均为 **4/4 PASS**。FFTW+P0 是四组最快候选，源码 SHA-256：
`761109fca0a3662be76aaf3341adfffd907b54f937d2489ced19150028f4e761`。

### 同节点加速比

| 用例 | V1 / half+delta | V1 / P0 | V1 / FFTW | **V1 / FFTW+P0** | half+delta / FFTW+P0 |
|---|---:|---:|---:|---:|---:|
| default | 1.117× | 1.189× | 1.309× | **1.415×** | 1.267× |
| small_mels | 1.140× | 1.244× | 1.422× | **1.565×** | 1.373× |
| large_fft | 1.239× | 1.350× | 1.529× | **1.720×** | 1.388× |
| narrow_band | 1.246× | 1.456× | 1.533× | **1.909×** | 1.533× |

FFTW+P0 同作业 librosa speedup 分别为 **12.287× / 17.178× / 18.365× / 20.947×**。
相对冻结 half+delta，组合候选又获得约 1.27–1.53× 改善。

## 3. 交错逐次测量

除官方最短时间外，还在同作业内对五个候选做三轮全量轮转，保存每次 wall/CPU time。
下表为逐次 wall-time 比值中位数：

| 比值 | default | large_fft | narrow_band |
|---|---:|---:|---:|
| FFTW+P0 / V1 | 0.746 | 0.618 | 0.568 |
| FFTW+P0 / half+delta | 0.823 | 0.749 | 0.696 |
| FFTW / half+delta | 0.870 | 0.834 | 0.830 |
| P0 / half+delta | 0.946 | 1.000 | 0.871 |

组合候选在交错测量中同样最快；P0 在 large_fft 的独立收益不稳定，
但组合版仍在该组优于单独 FFTW。原始逐次值见 `raw/cluster/ab-*-199.json`。

## 4. 128 点微基准

Xeon 计算节点，2000 次迭代：

| 组数 | 原 radix-2 ns/FFT | FFTW batched ns/FFT | FFTW ns/batch |
|---|---:|---:|---:|
| 33 | 819.095 | 449.303 | 14827.011 |
| 65 | 820.289 | 454.739 | 29558.042 |

FFTW 每次约为原 radix-2 的 0.55，支持替换内核的端到端方向。
自研 codelet 只在本地完成端到端筛选，未占用集群时间；其整程序不如 FFTW，
微基准接近 FFTW 不能作为推荐依据。

## 5. P0 集群 Profile

作业 200 使用独立单线程 chrono 插桩，全部输出 PASS：

| 用例 | half+delta Hann 秒 | P0 Hann 秒 | Hann 比值 | 插桩总时间比值 |
|---|---:|---:|---:|---:|
| default | 2.917 | 1.615 | 0.554 | 0.893 |
| narrow_band | 5.824 | 1.136 | **0.195** | 0.796 |

Hann/norm 时间分别下降约 45%/80%，与支持区间裁剪一致；
没有把总插桩时间当作未插桩 benchmark。

## 6. 推荐与边界

**当前 Overlap 推荐实现：`melspec_overlap_fftw_pruned_prototype.cpp`。**
理由是四组全部最快、全部 PASS，且相对冻结 half+delta 和 V1 均有同节点实测收益。
代价是引入 FFTW 第三方依赖、每线程批量 Plan/数组及一次性初始化成本；
自研 codelet 不推荐，P0-only 可作为无外部库备选。

本工作没有获得同作业的直接 FFTW R2C 整程序候选数据，因此**不宣称已超越直接 FFTW R2C**。
下一步若需判断与 FFTW 路线的绝对差距，应在同节点加入该正式候选做交错对照；
在此之前，不继续为追求数字而扩大 polyphase/AVX-512 等实验范围。
