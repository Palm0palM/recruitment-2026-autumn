# V3 CPU-Pipeline / SIMD B16 研究结果

**结论：找到稳定超过3%的非FFT改进。** 可移植slab单项在四组高重叠用例中减少约4.5–7.9%
完整进程耗时；Linux THP可用时，组合两轮减少约14.5–20.8%。推荐作为显式可选CPU补丁，
不自动修改稳定版本或未来Hybrid。两种优化默认关闭。

## 控制与隔离

从冻结SIMD FP32 B16 `e448c989c921762884582804efe23d6607678ff9` 建立
`exp/cpu-pipeline-simd`，目录 `/home/palm/projects/heptagon/melspec-cpu-pipeline-simd`。
原 `v3/cpu-pipeline` 保持 `b0ddf2bb7d2cfc19b2f0eb114cf0c8e387331589`，不改master、v2/simd或Hybrid。
正式源码快照 `cbf214d`，不修改 src/main.cpp、validate.py、data、include/melspec.h。
无新依赖、无fast-math、无FFT内核/精度/Plan/CLI改动。

所有正式比较在hepnode3、Xeon Gold6338、GCC15.2、16物理CPU、cpuset0–15（同一NUMA节点），
单作业内交错AB/BA各4次，每配置8次，使用完整程序进程wall的median。
每次作业重新编译冻结控制；时间不是worker局部耗时，更不是跨作业最小值比。

## 已证实的独立改进

### 1. 跨批原始音频slab复用（可移植算法改动）

独立提交 `3011dbf`，选项 `MELSPEC_PACK_REUSE=1`。按hop余数组织原始double音频行，
跨最多128帧复用，同一行连续16个double即对应16帧的同一window索引。
加窗仍为double乘法后转float；没有每帧加窗后再转置，没有改变FFT或Mel求和顺序。
只有高重叠尺寸启用，其他ISA/精度/非高重叠走原路径；不针对官方参数或音频内容。

| 用例 | 作业342 减时 | 作业348 减时 | 修复后364 | 最终380 |
|---|---:|---:|---:|---:|
| default | 5.00% | 5.54% | 4.49% | 4.95% |
| small_mels | 7.62% | 7.89% | 7.64% | 7.00% |
| large_fft | 6.60% | 6.04% | 6.02% | 5.87% |
| narrow_band | 6.42% | 5.83% | 6.51% | 6.11% |

四轮共128对逐次时间，127对候选更快；保留第一轮narrow的一次慢样本，不筛选异常以提高成绩。
pack（含slab准备）线程和/16由约 `.083/.082/.164/.164` 降至 `.042/.041/.094/.092` 秒。
其余FFT/recover/Mel基本不变。改善主要来自打包SIMD化/数据布局，非省略Hann乘法。
完整数学与边界证明见 [REUSE.md](REUSE.md)。

### 2. output合法初始化的页面成本（可选Linux调优）

独立提交 `ce56b4c`，选项 `MELSPEC_OUTPUT_HUGEPAGE=1`。
标准vector.reserve后，仅对属于本分配的完整页请求MADV_HUGEPAGE，然后执行原resize。
每个double仍正常构造和初始化，不使用未初始化值、伪造vector或越过allocator元数据。
不支持/提示失败时安全退回原行为，不改变全机THP/NUMA策略。

| 用例 | 作业354 减时 | 修复后364 | 最终380 |
|---|---:|---:|---:|
| default | 16.12% | 15.72% | 15.95% |
| small_mels | 6.60% | 6.73% | 6.03% |
| large_fft | 12.98% | 12.90% | 12.62% |
| narrow_band | 12.90% | 13.13% | 13.49% |

96对逐次比较全部更快。大输出resize约 `.20 → .080s`，small约 `.050 → .021s`；
大输出进程minor faults约 `210k → 115k`，AnonHugePages约382976KiB。
RSS仍同量级，没有减少必要输出大小或8字节/元素的初始化语义。
这依赖目标Linux的THP配置和内存状态，**不能宣传为跨平台同幅度收益**。
见 [HUGEPAGE.md](HUGEPAGE.md)。

## 安全性与验收

- 修复冻结版已有的n_fft=1空batch引用UB，独立提交 `cbf214d`。不影响四组大FFT计时。
- 作业364修复后的slab、HP各自68组独立合成输入与冻结B16 bit-exact，另与librosa比较PASS；
  严格ASan/UBSan的24组非空vector/预留capacity/非PCM double接口测试PASS。
- 早期sanitizer默认recover的警告原样保留，不能把退出0当作通过，见 [SAFETY.md](SAFETY.md)。
- 本机AVX2 B8的68组bit-exact回退、generic x86-64的24组接口、严格本地sanitizer均通过；
  AVX-512仅交叉编译，本机不冒充执行B16。
- 作业380：最终修复后的slab、HP、组合各自68组合成与24组严格sanitizer PASS，
  三个配置分别完成未修改validate.py全量四组4/4 PASS。组合另经作业396 runs=3再次4/4 PASS。

## 组合注意事项

单项改善不能相加。组合曾在DC合成案例中触发额外bit-exact断言；定位300个不同元素全部在
64..78的FP64尾帧，最大差约 `9.96e-29`，B16完整批次完全相同。组合68组按原官方容差
并对独立librosa全部PASS，但不承诺逐位一致。详细诊断及原始失败见 [INTEGRATION.md](INTEGRATION.md)。
组合严格sanitizer和官方完整验证已通过，两轮同节点正式对照如下（秒）：

| 用例 | 作业380 冻结→组合median | 作业396 冻结→组合median | 两轮减时 |
|---|---|---|---:|
| default | .839665 → .665051 | .835033 → .661286 | 20.80–20.81% |
| small_mels | .489775 → .418123 | .488436 → .417370 | 14.55–14.63% |
| large_fft | 1.041353 → .845978 | 1.038698 → .848111 | 18.35–18.76% |
| narrow_band | 1.000576 → .805212 | 1.001427 → .804120 | 19.53–19.70% |

两轮共64对逐次比较，组合全部更快。作业380另做slab→组合增量对照，分别再减少
16.12% / 6.94% / 13.56% / 14.68%，不是两个单项百分比相加。
最终官方runs=3（作业396）时间：`.643 / .412 / .841 / .800s`，4/4 PASS，
max_abs `.0390625 / .015625 / .125 / .09375`。官方单程序最小值只记录，不用来算跨作业加速。

## 通用性与内存

独立随机PCM尺寸额外8次交错：N/H=4096/31快7.13%；8192/257为0.30%、
2048/509为-0.55%、非重叠1024/1027为-0.47%。后三组接近噪声或轻微回退，
因此不承诺所有输入更快，也没有据此硬编码官方参数。详见REUSE.md与synthetic-reuse-380.json。

诊断进程峰值RSS median（MiB；不是同时比较的性能成绩）：

| 用例 | 冻结335 | slab342 | HP354 | 组合380 |
|---|---:|---:|---:|---:|
| default | 763.0 | 763.1 | 763.4 | 763.3 |
| small_mels | 334.9 | 334.4 | 334.0 | 333.9 |
| large_fft | 771.5 | 771.4 | 771.5 | 771.5 |
| narrow_band | 771.4 | 771.5 | 771.6 | 771.2 |

slab自身仍增加有界scratch（这四组约3–4MiB/16worker，另少量offset表），
只是不改变由y/output/写出float缓冲等大对象支配的全进程峰值量级；不宣称零额外内存。
最终组合诊断：pack/16约`.042/.041/.095/.092s`，output init约`.081/.021/.080/.080s`，
read/write与FFT基本维持原数量级，符合两项独立假设。

## Profile与停止结论

完整read/output init/pack/FFT/recover/Mel/write分解与Amdahl上界见 [BASELINE.md](BASELINE.md)。
pack约10–17%进程时间，足以支撑实现；Mel含store约1–2%，即使完全消除也达不到3%。
16核同一NUMA节点，分区均衡；无依据重做线程数/跨NUMA迁移。
PMU被perf_event_paranoid=4限制，未测得cache miss或硬件周期，不虚构PMU归因。
不增加prefetch/NT store/PGO/LTO或FFT重写。

跨帧Hann三基底预调制有精确实数等价式，但每帧读取量至少3倍、算术更多、三路FFT更贵，
且引入新舍入/相位误差，数学/流量/复杂度审查后停止，未实现，见 [AGGRESSIVE.md](AGGRESSIVE.md)。

## 交付

- [源码补丁](patches/cpu-pipeline-simd.patch)：仅src/melspec.cpp，已对冻结源码做git apply --check。
- [复现](REPRODUCE.md)：独立编译选项、通用验证、严格sanitizer、串行Slurm与Profile。
- 原始证据raw/；派生汇总raw/analysis-summary.json注明每个数据来源。
- 两个优化默认关闭，按目标ISA/系统显式启用；不修改未来Hybrid，不Push/Merge。
- 最终二进制SHA256：`51db61cb844c3a4860c20f7e39130670e49ef332974932f092c3f0993f4645d0`；
  源码SHA256：`595289c17c0ab385a191707e4722d52dc79f7d9840fe9951e384efa502d34700`。
- Slurm完整状态raw/accounting.txt；364是严格bit-exact失败，348/354虽COMPLETED仍含旧UB警告，
  均完整保留；380/396是最后的通过记录，不把早期警告掩盖掉。
