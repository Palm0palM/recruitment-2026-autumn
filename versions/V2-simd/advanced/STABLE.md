# 稳定节点：v2-simd-batch8-stable

- 标签：`v2-simd-batch8-stable`
- 固定提交：`09c99959688beef7a0a4c6d9c6fc42f110e6b843`
- 最佳算法提交：`70502f3`
- 源码 SHA-256：`229f2d0de9f85dec15b0597aa4b702cf80690c052cad457c4bf462199b80bf80`
- 集群证据：hepnode3 作业 95，16 物理核，四组全量 PASS。
- 中位数：default 0.891592 s、small_mels 0.543694 s、large_fft 1.207567 s、narrow_band 1.192772 s。
- 同作业 FFTW：0.995743 / 0.661361 / 1.362104 / 1.307900 s，Batch=8 四组均更快。
- 阶段 profile：作业 134。输入打包约占四个计算阶段的 26%～29%，大 FFT 谱恢复未随 Batch=8 改善。

后续实验必须保留此版本作为稳定对照，每个候选独立提交并重新验证。标签用于恢复完整代码、测试工具和报告，不表示后续实验已包含在该节点中。
