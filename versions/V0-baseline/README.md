# V0 — Baseline（原始朴素实现）

- 代码状态：tag `v0-baseline`（commit `25f2871`），`src/melspec.cpp`、`src/main.cpp` 均为题目原始版本
- 关键实现：逐帧 `std::vector<std::complex<double>>` 手写 Cooley-Tukey、每帧重建旋转因子、
  `power_spec` 完整物化（`vector<vector<double>>`，形状 `(n_bins, n_frames)`）、
  稠密 Mel 矩阵乘（`mels[i][k] * power_spec[k][t]` 按列访问）、单线程

## 1. 结论（先看这里）

1. **瓶颈不在 FFT，而在 Mel 投影那一次稠密矩阵乘**：实测占 **94%** 的运行时间；
   `perf` 采样显示 `compute_melspectrogram` self 时间 97.4%，FFT 相关仅 ~2.5%。
   原因是 `power_spec[k][t]` 的列访问：固定 `t` 遍历 4097 个 `k`，每次跨 ~1.5 MB 行，cache/TLB 全 miss。
2. **内存不可行**：`power_spec` 在 default 用例下就是 **6.3 GiB**，large_fft/narrow_band 需要 **12.6 GiB**。
   16 GB 的本地机器跑 large_fft 只能靠 swap；librosa 全量更是需要 ~25 GB。
   → 结论：必须改为**分块流式**，不能物化功率谱。
3. **朴素实现比 librosa 慢 20 倍左右**（5 s 片段：17.5 s vs 0.83 s），
   `validate.py` 的 speedup 列会是 0.05x 量级。
4. 全流程 `double` 而输出只要求 `float32`，容差 `rtol=1e-3, atol=1e-4` → **float32 化是白送的带宽收益**。
5. `Makefile` 已带 `-O3 -march=native -fopenmp`，但代码中无任何 OpenMP 指令 → 24 核完全闲置。

## 2. 环境

| 项 | 本地（趋势参考） | 集群（正式成绩） |
|---|---|---|
| 机器 | i7-14650HX，12C/24T，15 GiB | 待网络恢复后记录（`nproc`/`free -g`/`lscpu`） |
| 编译器 | g++ 16.2.1，`-O3 -march=native -fopenmp` | 待记录 |
| 数据 | 集群数据副本（MD5 校验一致）：`audio.pcm` 49,269,760 B | `/cache/recuritment/melspec/data/` |

## 3. 结果

### 3.1 本地（趋势参考，非成绩）

| 用例 | 输入 | 耗时 | librosa | speedup | 备注 |
|---|---|---|---|---|---|
| default | 5 s 截断 | 17.55 s | 0.83 s | 0.05x | `raw/local-timings.txt` |
| default | 50 s 截断 | 待补 | 2.12 s | — | |
| default | 全量 256.6 s | 见 `raw/v0_default_local.txt` | 未测（OOM 风险） | — | |
| large_fft | 全量 | **未测**：需 12.6 GiB 功率谱，本地会严重 swap | — | — | 必须上集群 |

### 3.2 集群（正式成绩，待补）

集群 SSH 在 2026-10-30 上午处于维护状态（`Maintaining... / Connection closed`），
恢复后执行下面命令补齐，并把输出粘到 `raw/cluster/`：

```bash
ssh hepnode && cd <repo> && git pull && make -j "$(nproc)"
mkdir -p logs && sbatch scripts/bench.slurm
# 结果：logs/validate-<jobid>.txt（四组 PASS/FAIL + 时间 + librosa 加速比）
```

## 4. 瓶颈定位（证据）

**方法 A：固定输入，只改 `n_mels`，分离「Mel 投影」与「FFT 部分」**

| n_mels | 5 s 片段耗时 | 说明 |
|---|---|---|
| 8 | 1.06 s | 只剩 FFT + 分帧 + 功率谱写入 ≈ 1.0 s |
| 64 | 4.76 s | |
| 256 | 17.55 s | Mel 投影 ≈ 16.5 s ≈ **94%** |

**方法 B：`perf record -g`**

```
# Samples: 70K of event 'task-clock:uppp'    (5 s 片段, default 参数)
  99.96%  self  compute_melspectrogram(...)     ← 稠密 Mel 乘 + 逐帧循环
   2.52%  children  compute_rfft(...)           ← FFT 部分
```

**方法 C：内存**

```
default     : 4097 × 192397 × 8 B = 6.3 GiB   (power_spec, double)
large_fft   : 8193 × 192333 × 8 B = 12.6 GiB
输出文件    : 256 × 192397 × 4 B  = 197 MB
```

## 5. 正确性

5 s 片段与 librosa 逐元素对比：`max|Δ| = 1.46e-3`，`np.allclose(rtol=1e-3, atol=1e-4) = True`
（大数值处绝对误差超过 `atol`，靠相对误差项通过——说明**容差本身就允许相当的数值误差**，为 float32 化留下空间）。
全量四组的官方校验需在集群补齐。

## 6. 复现命令

```bash
# 本地（趋势参考）
./scripts/setup.sh
tmp/.venv/bin/python scripts/bench.py --case default --slice 5 --runs 3
tmp/.venv/bin/python scripts/bench.py --case default --runs 3     # 全量，约 15~25 min

# 集群（正式）
ssh hepnode && cd <repo> && make -j && mkdir -p logs && sbatch scripts/bench.slurm
```

## 7. 下一步（V1 的假设）

| 优先级 | 优化 | 预期依据 |
|---|---|---|
| 1 | 去掉功率谱物化，改分块流式（块内 `[t][k]` 行主序） | 6.3 GiB → 几 MB，直接消掉内存墙 |
| 2 | Mel 投影改稀疏（三角滤波器只覆盖少数 bin） | 稠密 256×4097 vs 稀疏约 4096 taps → ~256× 计算量差 |
| 3 | FFT 换 FFTW `r2c` 批处理 + 计划复用 | 去掉每帧重新分配/重算旋转因子/复数 FFT 的 2× 冗余 |
| 4 | float32 全流程 | 带宽减半，容差允许 |
| 5 | OpenMP 按帧并行 | 24 核当前完全闲置 |

> 先做 1+2（算法/访存），再做 3（换库），最后 4+5（精度与并行），每一步单独成版并记录增量收益。
