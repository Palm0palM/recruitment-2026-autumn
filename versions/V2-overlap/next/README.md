# V2-overlap 第三轮：P0/P1 与组合候选

> **推荐实现与集群结果见 [cluster-next.md](cluster-next.md)**。FFTW+P0 四组全部最快，
> 相对 V1 加速 1.415× / 1.565× / 1.720× / 1.909×，全部 PASS。

起点：`4f9f87c15bca240ec286d8eb90f877f53bc1f49d`。
默认 `src/melspec.cpp` 始终为 V1；以下实现都在独立实验文件中。

## 独立提交

1. `ac6ba83`：P0 按 Mel 支持裁剪 Hann/power + 128 点微基准。
2. `61b372c`：FFTW batched C2C 重组。
3. `7b1d911`：自研 radix-4/AVX2 codelet，含失败调试记录。

组合候选 `melspec_overlap_fftw_pruned_prototype.cpp` 将 P0 与 FFTW 组合，
与两个独立实验分开测量，不能将其收益全部归因于某一项。

## 本地完整 A/B

`ab_next.py` 对 V1、half+delta、P0、FFTW、codelet、FFTW+P0 做三轮全量轮转，
记录逐次 wall/CPU time，最后再次逐版本校验 expected。下表是逐次 wall-time 比值的中位数，
不是单次最快值；本地宿主机负载只用于筛选，最终以集群为准。

| 候选 / V1 | default | large_fft | narrow_band |
|---|---:|---:|---:|
| half+delta | 0.856 | 0.709 | 0.800 |
| P0 | **0.838** | 0.721 | **0.705** |
| FFTW | **0.617** | **0.484** | 0.582 |
| codelet | 0.702 | 0.727 | 0.700 |
| FFTW+P0 | 0.672 | 0.585 | **0.456** |

相对冻结 half+delta：P0 的比值中位数为 0.850/0.894/0.889；
FFTW 为 0.688/0.704/0.716；FFTW+P0 为 0.748/0.772/0.563。
P0 与 FFTW 独立有效；FFTW+P0 在 narrow_band 最强，但在 default/large 的组合收益不呈现简单相乘关系。
Codelet 不如 FFTW，不能用在微基准中的约 225 ns/FFT 替代整程序结论。

## P0 阶段验证

同一插桩方法下，Hann+norm 的绝对时间变化：

| 用例 | half+delta Hann 秒 | P0 Hann 秒 | 比值 |
|---|---:|---:|---:|
| default | 1.424 | 1.258 | 0.883 |
| large_fft | 2.680 | 1.758 | 0.656 |
| narrow_band | 3.620 | 0.635 | **0.175** |

narrow_band 的 Hann 成本下降约 82%，与只需 16% 频点相符。
default/large 还受到按索引循环、分支和访存的影响，并不达到理论 bin 比例。
插桩总时间受宿主负载影响，不与未插桩 A/B 混用；全部诊断输出通过 expected。

## 当前候选边界

- 自研 codelet 保留为研究记录，不作为当前推荐。
- FFTW 路线依赖外部 libfftw3；集群使用私有源码构建，不使用其他 Agent 的库。
- 集群作业 199/200 已完成官方三次最小值、三轮交错逐次值和 P0 Profile；
  最终推荐为 FFTW+P0，详见 `cluster-next.md`。
