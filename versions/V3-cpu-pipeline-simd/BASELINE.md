# 冻结 SIMD B16 全程序 Profile

作业 335，hepnode3 / Xeon Gold 6338 / GCC 15.2，cpuset 0–15，单 NUMA node0，
16 物理核。控制源码 SHA256 `95aa8e56fb5ca91d89de153bc699a2a2361bb48b174b5076478d258dcf282f3f`。
三次诊断每次全量 expected PASS；正式无插桩控制也四组 PASS。

三次 median，秒；pack/FFT/recover/Mel 是线程累计时间除以 16 的**近似并行贡献**，
不是严格的可删除 wall time；实际 workers 含启动、尾部不均衡、抢占等成本。

| 用例 | 诊断进程 wall | read | output init | 其余 setup | workers | pack /16 | FFT /16 | recover /16 | Mel+store /16 | write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | .834 | .175 | .202 | .013 | .179 | .083 | .065 | .017 | .012 | .231 |
| small_mels | .491 | .173 | .050 | .009 | .176 | .082 | .064 | .017 | .010 | .060 |
| large_fft | 1.043 | .175 | .202 | .028 | .370 | .164 | .147 | .033 | .023 | .233 |
| narrow_band | 1.005 | .173 | .198 | .027 | .335 | .164 | .147 | .012 | .010 | .231 |

尾帧 .0007–.004 秒；进程剩余包括析构/退出/外层计时，不能把所有行独立 median 强行相加。
read 包含 int16 缓冲初始化、读入、double 初始化与归一化；write 包含 float 初始化/转换/写入，
均属受保护 main，不修改。worker Mel 内包含最终连续 16 帧写入。

## Amdahl 筛选

- 完全消除 pack 的近似上限：约 9.9% / 16.8% / 15.7% / 16.3% 的 elapsed 减少。
  因此要有 >3% 减时，需要 pack 至少降低约 30% / 18% / 19% / 18%，尚有可信空间。
- 完全消除 Mel+store 仅约 1.4% / 2.1% / 2.2% / 1.0%。不单独研究 Mel SIMD 或 NT store。
- recover 完全消除约 2.0% / 3.4% / 3.2% / 1.2%；融合仍有真实计算，单独 >3% 不可信，停止。
- output 初始化占约 24% / 10% / 19% / 20%，必须保留每个 double 的合法初始化。
  默认大输出约 210k minor faults（全进程）；可研究页面分配成本，不做未初始化 vector。
- workers 最大/最小约在几个百分点内，分区均衡；cpuset 全在一个 NUMA node，
  不做无依据跨 NUMA 迁移/线程数 sweep。当前启动线程分区是连续帧，保持。
- 不研究 FFT kernel / plan；不把历史 FFTW 阶段比例用于这个基线。

## 实现前假设

1. **跨批复用原始音频 slab**：不是在每帧加窗后转置。将覆盖一组连续 batch 的原始 double
   音频按 hop 余数分行，行内连续 hop 样本代表相邻帧。每次 B16 pack 可从每行连续加载
   16 个 double，用 double 广播 Hann 乘法再转换到 float。原始重排跨多个 batch/重叠窗口复用。
   原算法 pack 读约 8FN 字节；slab 预处理 O(S + 每 slab 的重叠 N) 而非 O(FN)，
   SIMD pack 的逻辑读量仍是 8FN，但来自紧凑缓存，减少标量构造/转换指令。
   不声称减少必要的 FN 次 Hann 乘法。风险：额外预处理、scratch 占 L2、行间 stride/TLB。
2. **页面初始化成本**：先检查目标 THP 策略。若有依据，只对 output.reserve 后私有完整页面
   请求 MADV_HUGEPAGE，再执行原 resize；这不会跳过任何对象初始化。Linux 可选提示，失败或
   不支持必须安全回退。不能把页面提示当作通用算法改进，必须跨轮 measured。

证据：raw/pipeline-335.out、raw/profile-335.json；无插桩控制用时位于同一原始日志末尾。
