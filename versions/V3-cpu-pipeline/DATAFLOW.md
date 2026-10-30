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

### D1 集群结果：不采用

作业 212（hepnode2）交错对照 median：default stable 0.988 vs overlap 0.994 s；
small_mels 0.660 vs 0.751 s；large_fft 1.360 vs 1.359 s；narrow_band 1.305 vs 1.314 s。
default/narrow 回退，large_fft 基本持平，small_mels 明显回退。初始化与 FFT 的内存竞争、
短期额外线程和同步成本抵消了理论重叠收益，因此保留 `OUTPUT_OVERLAP=0`，不再扩展该方向。

首次官方补跑因包装脚本将候选别名 `overlap` 误当成二进制文件名而失败；
实际文件名是 `output-overlap`。这不是算法 FAIL，原始错误保留，另用 `bench.slurm` 的 VARIANT 补跑完整验证。
第二次补跑又因通用 bench.slurm 未传 `FFTW_PREFIX`，在集群无 pkg-config/FFTW 时编译失败；
需在后续调用显式提供本实验静态库前缀。这些均属于验证流程问题，不算算法结果。

## D2：功率谱与 Mel 融合

**假设**：稀疏 Mel 的总 taps 不大于 n_bins。直接在每个支撑 bin 上由 FFT 复数输出计算模平方并累加，
可以删除完整 power 写入/读取，同时不增加每个 Mel 行内的 k 求和顺序变化。n_fft=1 保留原 power 特例。
`FUSE_POWER_MEL=1` 仅作用于 R2C hot path，C2C 和公开辅助函数不变。

本地 smoke 8/8、39 组对照均与稳定版逐元素相同，短输入拒绝。编译器诊断显示融合循环已使用
32-byte AVX2 vectors 和 unroll factor 4；因此没有额外手写 SIMD/FMA 的必要。
本地五次 median（stable → fusion，秒）：default 1.070→1.126、small_mels 0.791→0.822、
large_fft 1.739→1.686、narrow_band 1.761→1.673。大 FFT/窄带有趋势，default/小帧组回退，待集群判定。
原始结果：`raw/fusion-*`。
