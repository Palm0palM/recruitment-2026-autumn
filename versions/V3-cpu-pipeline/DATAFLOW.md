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

### D2 集群结果：采用融合

作业 244（hepnode2）交错 median（stable → fusion）：default 0.978→0.982 s（噪声内），
small_mels 0.661→0.649 s（约 1.7%）、large_fft 1.357→1.343 s（约 1.1%）、
narrow_band 1.308→1.274 s（约 2.7%）。官方 fusion 四组全 PASS：
`0.958 / 0.638 / 1.316 / 1.261 s`。三组稳定收益、default 无明显回退，且 39 组输出与稳定版完全相同，
因此将 `FUSE_POWER_MEL=1` 设为默认，并再做最终复测。原始结果见 `raw/cluster/*244*`。

最终作业 261 复测（stable → fusion median）：default 0.980→0.973 s，
small_mels 0.664→0.650 s，large_fft 1.349→1.350 s，narrow_band 1.298→1.267 s。
第二轮官方为 `0.961 / 0.641 / 1.320 / 1.256 s`，仍 4/4 PASS。
两轮共同结论是 narrow_band 约 2.4–2.7%、small_mels 约 1.7–2.2% 的端到端改善；
default 与 large_fft 接近测量噪声，不对它们宣称稳定收益。

作业 262 显示独立 power_s 已变为 0（工作融合进 mel_s）。
default 原 power+Mel 线程秒约 0.913，融合后约 0.831；narrow_band 原约 1.075，融合后约 0.572。
FFT 线程秒基本不变，证明收益来自消除中间 power pass，而不是改变 FFT backend。
最终融合版另通过 ASan/UBSan、8 项 smoke 和 39 组逐元素一致检查。

D1 的 output-overlap 在作业 248 也完成官方四组 PASS，但因性能无稳定收益仍不采用。

## D3：Mel 向量化审查（不新增实现）

S1 已把每行权重压成连续 storage；GCC 16.2 对原稀疏内层和融合内层都报告
`loop vectorized using 32 byte vectors and unroll factor 4`，尾循环也使用 16-byte vectors。
说明当前通用构建已经利用 AVX2，不存在“完全未向量化”的问题。
再引入手工 FMA、多个部分和或重新排序会改变数值路径，并增加维护成本；
在已有自动向量化且 Mel 只占少量端到端时间的情况下，不为局部指标增加独立 SIMD 实现。

## D4：输入加窗与数据局部性审查（不新增实现）

- Hann pack 已被编译器用 32-byte vectors 自动向量化；每帧仍必须执行窗口乘法并写 FFTW 对齐输入。
- hop=128 的重叠 PCM 在顺序帧间有自然缓存复用，额外跨帧缓存只会增加内存和管理工作。
- 不修改 FFTW 内核/Plan 就不能省略“加窗后的连续输入”这一步；不能把未加窗 PCM 直接交给原 R2C Plan。
- protected `read_pcm` 的 int16 缓冲、double 归一化和生命周期不可改，本轮不通过预加载/替换 main 绕过。

因此没有满足约束且有明确收益假设的新实现，停止该方向，不重复已被否决的普通 Frame Block。

## D5：可消除成本与固定成本复核

- 可测试并已闭环：压缩 Mel 构造、output 初始化重叠、power/Mel 融合。
- 受保护 main 固定：PCM 读取/归一化、double 输出读取、float32 分配/转换/写出。
- output 初始化可合法并行但集群无稳定收益；不能用 UB 跳过。
- 后续只接受端到端稳定改善，不再用单一 worker/Mel 指标替代成绩。
