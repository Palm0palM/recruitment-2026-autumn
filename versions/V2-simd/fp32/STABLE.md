# FP32 Batch16 候选节点

- 标签：`v2-simd-fp32-b16`
- 算法提交：`e448c989c921762884582804efe23d6607678ff9`
- 默认提交关系：普通 `make` 按目标 ISA 自动选择；AVX-512F 目标生成 FP32 Batch16，
  AVX2/FMA 生成 Batch8，其他目标回退 FP64。`MELSPEC_FP32_BATCH=0` 可强制 strict FP64。
- 集群证据：hepnode3 作业 271，四组全量 PASS。
- 中位数：0.834009 / 0.485544 / 1.035395 / 0.999337 s。
- 对照：同作业 FP64 Batch8 0.865315 / 0.527069 / 1.138472 / 1.117619 s；
  FFTW float+PATIENT 0.855914 / 0.516943 / 1.064758 / 1.012433 s。
- 精度：相对 FP64 的 tolerance 利用率最大 0.0132，约 70 倍余量；高动态边界全部 PASS。
- 阶段成本：FP32 B16 的 FFT 每帧约 5.4/12.3 μs，明显低于 FP64 B8 的 12.0/28.7 μs。
- 回退：无 AVX-512 → FP32 Batch8（AVX2/FMA）；否则 FP64 Batch8/Batch4/单帧。

该 tag 是性能 fast path 候选，不表示所有部署默认应使用 FP32；严格 double 语义仍使用 `v2-simd-inplace-stable`。
