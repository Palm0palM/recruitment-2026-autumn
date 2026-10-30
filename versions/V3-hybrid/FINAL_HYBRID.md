# Hybrid 最终报告（V3-hybrid）

## 1. 目标与结论

目标：每次 `compute_melspectrogram()` 只选择**一个**冻结后端完整处理输入，
选择依据是输入规模与硬件环境，选的是**端到端更快**的那一个，而不是 Kernel 更快的那一个。

结论：做到了，而且收益与代价都能被数据解释。

- 小任务（`frames ≲ 64k`，取决于 `n_fft`/`n_mels`）固定选 CPU：CUDA 每进程
  固定成本约 0.24–0.25 s，小输入下 CPU 快 5–20 倍。
- 大任务（`n_fft=16384` 且 ≥ 65k 帧，或 `n_fft=8192` 且 ≥ 160k 帧）固定选 CUDA：
  吞吐优势压过固定成本，最高实测 1.51 倍。
- 38 点终验：auto 累计 8.331 s，始终 CPU 10.010 s，始终 CUDA 14.003 s，
  逐点最优 8.067 s；auto 只比不可实现的最优上界慢 3.27%，逐点选择与最优一致 37/38。
- 没有为了“让 CPU 参与”而人为分流：auto 在 38 点中选了 7 次 CUDA，
  其余 31 次 CPU，全部由同一个线性成本模型决定。
- 唯一一次判错是 `official-small_mels`（CPU 预测 0.505 s / CUDA 0.552 s）：
  该点正落在交叉点，且当场 CPU 比训练会话慢 34%，CUDA 反而快 20%。
  这是唯一超过 10% 的逐点代价，原因与量化见 `BENCHMARK.md` 3.6 节。
- 强制 CPU / 强制 CUDA / auto 三模式都通过官方 4/4；强制 CUDA 在设备不可用时
  明确报错；auto 只在设备不可用时回退 CPU，CUDA 真实计算错误一律向上抛。

## 2. 交付物

| 内容 | 位置 |
|---|---|
| 公开 Dispatcher + 成本模型 | `src/melspec.cpp` |
| CPU 后端（冻结 SIMD） | `src/melspec_cpu.cpp` |
| CUDA 后端（冻结 async 流水线） | `src/melspec_cuda.cu` |
| 私有后端接口 | `include/melspec_backend.h`、`include/melspec_cuda.h` |
| 架构与错误语义 | `versions/V3-hybrid/ARCHITECTURE.md` |
| 全部测量与原始数据 | `versions/V3-hybrid/BENCHMARK.md`、`raw/h0|h1|h2/` |
| 成本模型与交叉点 | `versions/V3-hybrid/DISPATCH_MODEL.md` |
| 测量/拟合/汇总脚本 | `scripts/hybrid_bench.py`、`fit_hybrid_model.py`、`summarize_hybrid.py` |

受保护文件 `src/main.cpp`、`validate.py`、`data/`、`include/melspec.h` 未修改；
CLI 参数、输入输出格式、精度容差均不变。

## 3. 后端切换机制（一句话版）

```text
MELSPEC_BACKEND=auto（默认）→ 线性成本模型比较 T_cpu 与 T_cuda → 选一个后端完整执行
MELSPEC_BACKEND=cpu         → 直接 CPU，不探测 CUDA
MELSPEC_BACKEND=cuda        → 先检查设备可用性，不可用则报错；计算错误不回退
```

模型输入只有 `frames`、`n_fft`、`n_mels`（由调用参数与输入长度算出），
不看用例名、不看文件内容、不做试跑、不加锁、不初始化 CUDA（除非已决定用 GPU）。
详见 `DISPATCH_MODEL.md`。

## 4. 正确性

| 检查 | 结果 |
|---|---|
| CPU-only 构建官方 4/4 | PASS（`raw/h2/h2-official-cpu-only-397.txt`） |
| 统一二进制 强制 CPU 4/4 | PASS（`raw/h2/h2-official-cpu-397.txt`） |
| 统一二进制 强制 CUDA 4/4 | PASS（`raw/h2/h2-official-cuda-397.txt`） |
| 统一二进制 auto 4/4 | PASS（`raw/h2/h2-official-auto-397.txt`） |
| 38 点矩阵逐点校验（auto 也校验） | 38/38 PASS（`raw/h2/hybrid-h2-397.json`） |
| 无可见 GPU 时 auto 回退 CPU | 4/4 PASS（`raw/h2/h2-official-nogpu-397.txt`） |
| 无可见 GPU 时强制 CUDA | 非零退出 + 明确报错（`raw/h2/h2-forced-nogpu-397.txt`） |
| auto 在 CPU 决策下不触碰 CUDA | `strace` 无 `libcuda` / `/dev/nvidia` 访问（`raw/h2/h2-no-cuda-init-397.txt`） |
| auto 在 CUDA 真实错误时 | 直接报错退出，不回退（本地证据 `raw/h2/local-auto-error-and-cpu-path.txt`） |

## 5. 性能结果

完整数据见 `BENCHMARK.md` 第 2/3 节，原始值见 `raw/h1/`、`raw/h2/`。

### 5.1 38 点三模式（Job 397，3 轮交错，中位数）

| 策略 | 累计耗时 | 相对始终 CPU | 相对逐点最优 |
|---|---:|---:|---:|
| 始终 CPU | 10.010 s | 1.000 | +24.1% |
| 始终 CUDA | 14.003 s | 1.399 | +73.6% |
| auto（交付） | 8.331 s | 0.832 | +3.27% |
| 逐点最优 | 8.067 s | 0.806 | — |

### 5.2 官方四组（`validate.py --runs 1`，秒）

| 模式 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| auto | 0.871 | 0.676 | 0.881 | 0.872 |
| 强制 CPU | 0.940 | 0.697 | 1.436 | 1.195 |
| 强制 CUDA | 0.877 | 0.579 | 0.872 | 0.867 |

官方四组上 auto 与“始终 CUDA”几乎相同（`small_mels` 选 CPU 多花约 0.1 s），
但正确性、语义与全矩阵表现都由同一套规则得到，没有为四个用例写特例。

### 5.3 调度器开销

- 调度逻辑：中位 **+0.75 ms**（含 `MELSPEC_DISPATCH_LOG` 输出；无日志更小）。
- 统一二进制多加载 CUDA/cuFFT DSO：约 **4 ms**（与选哪个后端无关）。
- CPU 决策下不打开任何 `/dev/nvidia*`，不做设备初始化。

## 6. 已知局限与后续

- 成本模型是按“单进程一次性完成”的部署画像标定的；换成常驻服务（复用 CUDA 上下文
  与 cuFFT plan）后 `T_cuda` 常数项会显著下降，交叉点左移，必须用
  `scripts/hybrid_bench.py` + `fit_hybrid_model.py --emit-cpp` 重标定。
- CPU 侧绝对耗时随节点/时段漂移：同一二进制同一配置跨作业 0.849 s vs 1.049 s（+24%），
  同一配置的另一用例 0.493 s vs 0.661 s（+34%）；CUDA 侧 ≤5%。交叉点附近的胜负因此
  可能落在噪声内，H2 中 `official-small_mels` 的 +21% 就是这种情形。
- 模型对 H1（训练）与 H2（留出）的逐点最优各命中 37/38，留一交叉验证 35/38；
  未使用阈值/特例去拟合单个点。合并两场会话重拟合的对比见 `DISPATCH_MODEL.md` 6.3 节。
- 模型不包含 GPU 争用项；多作业共享 GPU 时应以实测重标定或显式强制 CPU。
- 范围外推（`n_fft>16384`、输入 >49 MB）属于估计，需要重测。
