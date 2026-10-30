# V3 CPU-Pipeline 数据流与计算融合实验

稳定基线：算法提交 `d94e45c`，B=1 / sparse / 16 workers，官方 `0.957 / 0.648 / 1.334 / 1.296 s`。
本轮不重复普通 Frame Block、双前缀矩或线程数调整；每项先写性能假设，再独立测量。

## D1：output 初始化与首批 FFT 重叠

**假设**：output.resize 是必要的合法零初始化，但可在独立线程执行；worker 同时做第一批 pack/FFT/power，
在第一次写 output 前通过 shared_future 同步。若初始化与首批计算可重叠，default 最多可隐藏约 0.2 s。

实现边界：
- 不改变 `std::vector<double>&`，不跳过任何元素初始化，不读未初始化内存。
- 初始化完成前 worker 不访问 output；失败时 worker 不写 output，主线程 join 后重新抛出异常。
- 初始化线程资源不足时退回原串行路径；FFTW Plan 和输入/输出 scratch 不变。
- 默认 `OUTPUT_OVERLAP=0`，候选显式设 1；短期 17 线程和内存带宽竞争可能抵消收益。

本地正确性：smoke 8/8 PASS，39 组对照与稳定版逐元素完全相同，3 组短输入拒绝。
本地五次 median（stable → overlap，秒）：default 0.799→0.795、small_mels 0.532→0.532、
large_fft 1.746→1.769、narrow_band 1.666→1.617。结果不一致，本地不作采用决定，转同节点 A/B。
原始结果：`raw/output-overlap-*`。集群结果待补。
