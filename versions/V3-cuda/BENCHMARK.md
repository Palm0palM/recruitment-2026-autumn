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
将保留全部逐次 wall time 和官方 expected 校验，不只保留获胜的最小值。
