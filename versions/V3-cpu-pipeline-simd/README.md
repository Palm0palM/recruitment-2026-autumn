# 最后一轮 CPU-Pipeline / SIMD B16

冻结控制：`e448c989c921762884582804efe23d6607678ff9`，编译时必须显式
`-DMELSPEC_FP32_BATCH=16`；普通 make 仍是 FP64，不能误作 B16。
本分支从冻结点独立建立，不修改 v3/cpu-pipeline、v2/simd 或 Hybrid。

## 测量约束

- 本机无 AVX-512，只做交叉编译及回退检查。B16 在已授权的专属集群目录运行。
- 原 src/main.cpp、validate.py、data 不改；完整进程 A/B 使用原 main 和无插桩源码。
- Profile 只生成 tmp 内 melspec 诊断副本；原 main 使用 GCC 函数进出钩子测 read/write，
  不替换入口、不改变 I/O 行为。插桩成绩不是性能成绩。
- 同作业内交错 A/B，多轮且全量 expected 校验；候选还需独立合成输入覆盖。
- 优先 >3% 端到端空间；线程累计阶段时间不当作进程 wall time。
- 已知负结果：每帧连续加载后 8x8 转置、普通 frame block、双前缀矩等不重复。

## 结果与交付

- 找到两个独立有效方向：跨批raw slab复用、可选Linux输出THP提示。
- slab四轮端到端减少约4.5–7.9%；HP三轮约6.0–16.1%。均以冻结B16为控制，不是旧FFTW。
- 组合两轮约14.5–20.8%，68组合成、严格sanitizer及两轮官方四组通过。
- 两种优化默认关闭；不改稳定分支或未来Hybrid，未Push/Merge。

入口：[研究结果](RESULTS.md) · [完整基线Profile/上界](BASELINE.md) · [移植与复现](REPRODUCE.md)

独立研究：[slab](REUSE.md) · [THP](HUGEPAGE.md) · [激进等价方案的停止分析](AGGRESSIVE.md)

安全与组合：[空batch UB修复](SAFETY.md) · [组合尾帧舍入诊断](INTEGRATION.md)

[仅源码补丁](patches/cpu-pipeline-simd.patch)只修改src/melspec.cpp；所有原始日志见raw/。
