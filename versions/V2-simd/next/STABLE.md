# 当前最佳节点：v2-simd-inplace-stable

- 标签：`v2-simd-inplace-stable`
- 算法提交：`aa2405b26de170c1fae8ea693febf27cb5adf6ae`
- 组成：AVX-512 Batch=8、Mel 支持区间裁剪、原位功率谱；AVX2 Batch=4 回退。
- 集群证据：hepnode3 作业 214，16 物理核，四组全量 PASS。
- 中位数：default 0.864017 s、small_mels 0.523456 s、large_fft 1.133168 s、narrow_band 1.115651 s。
- 同作业 FFTW：0.992357 / 0.658082 / 1.357742 / 1.303727 s。
- 正确性：随机、零值、冲激、正弦、尾帧、CLI 全 Nyquist、1～16384 rfft、官方四组及 ASan 通过。
- 阶段证据：作业 208 的支持裁剪与作业 214 的原位融合均有独立收益。

后续默认以此节点为对照。`v2-simd-batch8-stable` 保留为不含支持裁剪和原位融合的早期基线。连续转置与 DIF 均未进入本节点。
