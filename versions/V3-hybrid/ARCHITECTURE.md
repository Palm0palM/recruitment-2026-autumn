# Hybrid 架构：CPU SIMD / CUDA 自适应后端

## 1. 身份与来源

| 部件 | 来源 | 固定身份 |
|---|---|---|
| CPU Backend | `v2/simd` FP32 Batch16, AVX-512F | 算法提交 `e448c989c921762884582804efe23d6607678ff9` |
| CPU 构建/ISA 策略 | SIMD 最终冻结 | `v2-simd-fp32-b16-final`（`780d8af`） |
| CUDA Backend | `v3/cuda` async stable | 算法提交 `3e11194688b4b581b8ca527c5d292dbee451b0f1` |
| Hybrid Worktree | 本目录，分支 `v4/hybrid` | 起点 `780d8af` |
| 调度模型 | 本工作测量的 H1 数据 | `versions/V3-hybrid/raw/h1/` |

未移植 `v3/cuda-sprint` 的后台设备分配、GPU 输入转换和 A/B/AB 实验。CUDA 配置固定为
FP32 / Batch512 / Fused Mel / Context Overlap / Async Output / Resize Thread / Staging16。
CPU 侧保持 FP32 Batch16 与 AVX-512 → AVX2 → FP64 的 ISA 回退。

## 2. 文件职责

```text
src/melspec.cpp            公开 compute_melspectrogram + MELSPEC_BACKEND + 成本模型
src/melspec_cpu.cpp        冻结 SIMD CPU 实现（入口 compute_melspectrogram_cpu）
src/melspec_cuda.cu        冻结 CUDA 实现（入口 compute_melspectrogram_cuda）
include/melspec.h          官方公开头，未修改
include/melspec_backend.h  CPU/CUDA 私有入口 + cuda_backend_available()
include/melspec_cuda.h     CUDA 私有声明
scripts/hybrid_bench.py    H1/H2 全进程交错 Benchmark
scripts/fit_hybrid_model.py H1 → NNLS 成本模型 → C++ 宏
scripts/summarize_hybrid.py auto / 始终 CPU / 始终 CUDA 对比
```

相对冻结提交的改动只有三处：

1. CPU 文件：包含 backend 头，把公开入口重命名为 `compute_melspectrogram_cpu`，
   函数体未改。
2. CUDA 文件：包含 backend 头，新增 `cuda_backend_available()`（只调用
   `cudaGetDeviceCount`），Kernel 与流水线未改。
3. 公开入口 `compute_melspectrogram` 改为唯一 Dispatcher。

`compute_melspectrogram`、`compute_melspectrogram_cpu`、
`compute_melspectrogram_cuda`、`hz_to_mel`、`mel_to_hz`、
`build_mel_filterbank` 均只有一个定义（见 `raw/backend-audit.txt`）。

## 3. 一次调用的控制流

```text
compute_melspectrogram(y, sr, n_fft, hop, n_mels, f_min, f_max, out)
        │
        ├─ MELSPEC_BACKEND=cpu ───────────────► compute_melspectrogram_cpu
        │
        ├─ MELSPEC_BACKEND=cuda
        │     ├─ cuda_backend_available()? ──no──► throw（明确报错）
        │     └─ yes ────────────────────────────► compute_melspectrogram_cuda
        │
        └─ MELSPEC_BACKEND=auto（默认）
              ├─ 成本模型预测 CPU ──────────────► compute_melspectrogram_cpu
              │     （不调用任何 CUDA API）
              └─ 成本模型预测 CUDA
                    ├─ 设备不可用 ─────────────► compute_melspectrogram_cpu
                    └─ 设备可用 ───────────────► compute_melspectrogram_cuda
```

每次调用只有一个 Backend 完整处理整个输入并写入 `output`；不存在 CPU/GPU 分工，
也不存在先跑一遍再决定。

## 4. 边界与错误语义

- **强制 CPU**：不探测、不初始化 CUDA 设备。CPU-only 二进制完全不链接 CUDA
  Runtime / cuFFT；统一二进制因链接关系仍会在启动时加载 CUDA DSO，
  但不会打开 `/dev/nvidia*`，也不建立上下文。
- **强制 CUDA**：设备不可用、或二进制未编译 CUDA 时抛 `std::runtime_error`，
  由 `main` 打印 `Error: ...` 并以非零码退出。
- **auto**：只有“设备不可用”这一环境事实允许回退 CPU；CUDA 运行期真实错误
  （Kernel launch、cuFFT、显存等）直接向上传播，不回退、不重算。
- 调度器不做事先试跑，不分配显存，不加锁，不启动线程；预测只是若干次浮点乘加，
  外加一次 `std::log2`。
- 资源生命周期完全沿用冻结实现（RAII 的 DeviceBuffer / PinnedBuffer / Stream /
  Event / cuFFT plan），Dispatcher 不介入。

## 5. 构建

```bash
make CUDA=0 -B -j16                      # CPU-only，可独立运行，无 CUDA DSO
make CUDA=1 -B -j16 CUDA_HOME=... CUDA_ARCH=89   # CPU+CUDA（L40）
```

- CPU ISA 由 `-march=native` 与运行期目标决定，Batch16/8/1 由 `Makefile` 自动选择。
- 对象目录按 `MELSPEC_FP32_BATCH` 与全部 CUDA 配置参数隔离，
  不同模式不会复用错误对象。
- 受保护文件 `src/main.cpp`、`validate.py`、`data/`、`include/melspec.h` 未修改。

## 6. 运行期开关

| 变量 | 取值 | 含义 |
|---|---|---|
| `MELSPEC_BACKEND` | `auto`（默认）/`cpu`/`cuda` | 后端选择模式，不改变 CLI |
| `MELSPEC_DISPATCH_LOG` | 未设/`1` | 向 stderr 输出一次决策（后端、特征、两侧预测），仅用于研发与审计 |

`MELSPEC_BACKEND` 解析一次/调用，非法值明确报错，不会静默退回 auto。
