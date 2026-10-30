# V3 CUDA Benchmark 记录

## 正式 FP32 候选（作业 194，hepnode2）

- 代码：`4fdbd5cec47e73fb88f1be3bc4903c3779a8c1ce`，G1–G3 单流 fused pipeline。
- 参数：FP32 R2C，batch=256，fused=1，pageable host staging，CUDA_PROFILE 关闭。
- 私有 CUDA 13.4/cuFFT，sm_89，L40；16 CPU，`--hint=nomultithread`，未写 `--mem`。
- 实际 cpuset：`0-3,16-19,32-35,48-51`；输出使用节点本地 scratch。
- 完整 `validate.py --runs 3`，每组三次取最小，**4/4 PASS**。
- 候选二进制 SHA-256：`5dbfb09740adb7f3f5b4375433e92914e0b314365d29ca4e6932424c44f42135`。

| 用例 | GPU FP32 全进程（秒） | 同次官方 librosa / GPU | max_abs_err | 状态 |
|---|---:|---:|---:|---|
| default | **1.183** | 12.121× | 0.09375 | PASS |
| small_mels | **0.660** | 20.958× | 0.0234375 | PASS |
| large_fft | **1.183** | 25.048× | 0.21875 | PASS |
| narrow_band | **1.173** | 25.118× | 0.15625 | PASS |

错误比小样本更大，但逐元素仍满足 `rtol=1e-3, atol=1e-4`；不以绝对误差单项误判。
这是完整进程口径，包含 main 读取/归一化、host/device 转换、plan、GPU pipeline、D2H 和 main 写盘。

## 与 A2 的关系（不把不同 harness/节点混算）

此前同版本 A2 在 hepnode2 的官方数字为 0.985 / 0.650 / 1.356 / 1.300 秒。
本 GPU 候选在 large_fft/narrow_band 上更快，small_mels 接近，default 较慢；GPU 不是全面胜出。
但 CPU 绑核/作业环境会影响结果，最终选择以同作业 A2/GPU 交错 sweep 为准，而不是只引用上述旧数字。

重复官方作业 195 已取消，没有重复计时。后续 fused/separate、batch 与 pinned 的 sweep
保留全部逐次 wall time 和官方 expected 校验，不只保留获胜的最小值。

## 初始化重叠正式候选（作业 215，hepnode2）

代码 `c4df954db14acae4266d09f1828c9bb6e479708d`，FP32、batch256、fused、pageable、
`CUDA_OVERLAP_INIT=1`。提前 `cudaFree(0)` 与 host 准备重叠，复审修正已包含。
官方四组再次全部 PASS：default **1.012 s**、small_mels **0.573 s**、large_fft **1.013 s**、
narrow_band **1.015 s**；相对同次官方 librosa 分别为 13.514× / 23.118× / 28.350× / 28.247×。
候选二进制 SHA-256：`0f4407a7b7130e643063396552a1acfe523f80a71fac32cc89fb570e062413b1`。

相比首次 194 的 1.183 / 0.660 / 1.183 / 1.173 秒，初始化重叠显著改善了所有用例，
特别是 default。原始官方输出与源码/二进制哈希见 `raw/cuda-215.out`、`raw/official-215.txt`。

## 同作业 A2 / CUDA sweep（作业 201、220）

两次 sweep 都在 hepnode2，A2 与 CUDA 以轮换顺序运行，每次完整输出都对 expected 校验。
220 的九种变体为 batch128/256/512、fused/separate、FP64、pinned 与初始化重叠，各三次。

以 220 的三次最小进程时间计算，代表性结果：

| 用例 | A2 min（s） | 未重叠 GPU min（s） | 初始化重叠 GPU min（s） | 重叠 vs A2 |
|---|---:|---:|---:|---:|
| default | 1.007871 | 1.144456（b256） | **0.989239（b512）** | **1.019×** |
| small_mels | 0.805170 | 0.656839（b256） | **0.563742（b512）** | **1.428×** |
| large_fft | 1.803441 | 1.170345（b256） | **1.030107（b256）** | **1.751×** |
| narrow_band | 1.683831 | 1.173513（separate/b256） | **1.033304（b256）** | **1.630×** |

batch512 的 default/small_mels 略优，batch256 的 large/narrow 略优，差距约 1% 量级，
不把微小区间夸大成普遍规律。综合候选先用 batch256 完成官方 215，再以 batch512 正式复核。
FP64 较慢；pinned 在部分 default/large 有小幅收益但在 small_mels 等场景不稳定，不设默认。
fused/separate 差异较小，保留 fused 以避免功率矩阵及其显存流量。

原始逐次结果：`raw/sweep-201.out`、`raw/sweep-201.json`、`raw/sweep-220.out`、`raw/sweep-220.json`。
这些数字来自 `cuda_bench.py`，与官方 validate 的 harness 有小幅口径差异，不与官方最小值混算。

## 最终 batch512 复核（作业 226）

同一算法 commit `c4df954`，FP32、batch512、fused、pageable、初始化重叠。
完整官方 `validate.py --runs 3`，四组全部 PASS：

| 用例 | GPU FP32（秒） | 同次官方 librosa / GPU | max_abs_err |
|---|---:|---:|---:|
| default | **1.016** | 13.877× | 0.09375 |
| small_mels | **0.578** | 23.895× | 0.0234375 |
| large_fft | **1.035** | 28.883× | 0.21875 |
| narrow_band | **1.040** | 28.435× | 0.15625 |

正式二进制 SHA-256：`bcb55ab0986ee41330e9501f667d1bc19c557045c659fe7e8d513d7061c35f30`。
最终默认构建只把上述参数设为默认，不改变算法；同节点重建后需与该哈希对照。

**结论：GPU 在本题真实规模下有竞争力，但不是所有用例都明显快于最佳 CPU。**
同作业 220 中，初始化重叠 batch512 对 A2 的 default/small_mels 约 1.019×/1.428×，
batch256 对 large/narrow 约 1.751×/1.630×；最终选择简单的 batch512 候选，
其大 FFT 与 batch256 的差异约 1%，以官方 226 的完整验证为准。
