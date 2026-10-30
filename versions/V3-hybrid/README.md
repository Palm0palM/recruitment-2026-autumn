# V4 Hybrid SIMD + CUDA — 准备工作

状态：**后端拆分、构建模式、运行模式和符号审计完成（准备阶段记录）。**
后续的交叉点测量、自适应调度与最终成绩见同目录 `ARCHITECTURE.md`、
`BENCHMARK.md`、`DISPATCH_MODEL.md`、`FINAL_HYBRID.md`。

## 1. Worktree 与来源

- Worktree：`/home/palm/projects/heptagon/melspec-v4-hybrid`
- 分支：`v4/hybrid`
- 起点：`780d8aff99d2e1074dff909cfae192528bd44902`（SIMD 最终冻结）
- CPU Backend：`v2/simd` 算法提交 `e448c98`，源码 SHA-256 `95aa8e56...`
- CUDA Backend：`v3/cuda` 稳定算法 `3e11194688b4b581b8ca527c5d292dbee451b0f1`
- 禁止移植的 sprint 实验：`v3/cuda-sprint` 的后台设备分配、GPU 输入转换、A/B/AB。
- 未使用 Git merge/cherry-pick；文件从冻结提交独立提取并做最小接口适配。

## 2. 源码布局

| 文件 | 作用 |
|---|---|
| `src/melspec.cpp` | 唯一公开 `compute_melspectrogram`，解析 `MELSPEC_BACKEND` 并分派 |
| `src/melspec_cpu.cpp` | 冻结 SIMD CPU Backend，算法主体与 `e448c98` 相同 |
| `src/melspec_cuda.cu` | 冻结 CUDA Backend，算法主体与 `3e11194` 相同 |
| `include/melspec_backend.h` | CPU/CUDA 私有入口与设备可用性探针 |
| `include/melspec_cuda.h` | 冻结 CUDA 内部声明 |
| `include/melspec.h` | 官方公开接口，未修改 |

CPU 文件相对 `e448c98` 只改了两处：

1. 增加 `melspec_backend.h` 声明；
2. `compute_melspectrogram` 内部名改为 `compute_melspectrogram_cpu`。

CUDA 文件相对 `3e11194` 只改了两处：

1. 增加 `melspec_backend.h`；
2. 增加 `cuda_backend_available()`，仅用于设备可用性探测。

没有重复定义 `hz_to_mel`、`mel_to_hz`、`build_mel_filterbank`、
`compute_melspectrogram` 或 `compute_rfft`。公共数学函数由 CPU Backend 提供，
CUDA Backend 链接同一份实现。

## 3. 构建模式

### CPU-only（无 CUDA Runtime/cuFFT 依赖）

```bash
make CUDA=0 -B -j4
ldd ./melspec | grep -E 'cuda|cufft'    # 应无输出
```

CPU-only 已验证 11 项 smoke 全 PASS。

### CPU+CUDA

```bash
export CUDA_HOME=/path/to/private/cuda
make CUDA=1 -B -j4 CUDA_HOME="$CUDA_HOME" CUDA_ARCH=89
```

对象目录按 `hybrid-cpu/fpN` 或 `hybrid-cuda-fpN-arch...` 独立。
CPU ISA 仍自动选择：AVX-512F → FP32 Batch16，AVX2+FMA → FP32 Batch8，否则 FP64。

CUDA 默认配置固定为：

```text
FP32 / batch512 / fused Mel / context-init overlap /
async staged output / resize thread / staging rows=16 /
pinned=0 / defer_resize=0 / profile=0
```

## 4. 运行模式

公开 CLI 不变，额外读取环境变量：

| `MELSPEC_BACKEND` | 行为 |
|---|---|
| `cpu` | 强制 CPU Backend；不执行 CUDA 设备探测或初始化 |
| `cuda` | 强制 CUDA；设备不可用或内部计算错误直接报错 |
| `auto`（默认） | 设备可用时当前先用 CUDA，否则 CPU；后续将替换为规模模型 |

当前 `auto_should_use_cuda()` 是准备阶段的占位策略：CUDA 可用则返回 true。
**尚未完成交叉点模型**，因此暂不宣称 auto 已按规模最优分派。

## 5. 验证结果

- CPU-only 构建：`make CUDA=0`，无 CUDA DSO，smoke 11/11 PASS。
- 目标 ISA 交叉构建检查：`-march=icelake-server` 自动选择 `MELSPEC_FP32_BATCH=16`，
  二进制中存在 `apply_batch16f` 与 `recover_power_batch16f` 符号，证明不是 V1 CPU。
- CPU+CUDA 构建：本地 sm_120、目标 L40 sm_89 均成功；动态链接 `libcufft.so.12`/`libcudart.so.13`。
- 符号检查：四个关键公开/后端函数各只有一份定义，无重定义。
- 无可见 GPU 环境：
  - `MELSPEC_BACKEND=cpu` 正常运行，输出与 CPU-only golden 字节一致；
  - `auto` 回退 CPU，输出与 golden 字节一致；
  - `cuda` 返回 1，错误信息明确为 “no CUDA device is available”。
- CPU 模式开销（192k 样本小输入，8 次 median）：CPU-only 3.219 ms，
  CPU+CUDA 二进制的 cpu 模式 6.366 ms；差值来自进程加载 CUDA/cuFFT DSO，不是 CUDA 初始化。
  对全量默认用例约 0.8s 的量级约 0.4% 固定开销；正式 CPU 性能测试仍应使用 CPU-only 构建。
- 本地 GPU：
  - `cuda` 输出对 CPU golden `allclose`，max|err|=6.10e-05；
  - `auto` 与强制 `cuda` 输出逐字节一致（当前占位策略）。
- 原始命令/哈希/链接信息见 `raw/backend-audit.txt`、`raw/symbols-and-linkage.txt`、
  `raw/backend-modes.txt`、`raw/cpu-*-build.log`、`raw/cpu-only-smoke.txt`。

## 6. 已知限制

1. `auto` 尚无规模交叉点模型，仅做设备可用性选择。
2. CPU+CUDA 二进制因动态链接会在进程启动时加载 CUDA Runtime/cuFFT DSO；`cpu` 模式不执行 CUDA 初始化，但若运行环境完全没有这些 `.so`，动态加载器会先失败。CPU-only 构建不受影响。
3. CUDA 二进制使用绝对 RUNPATH 指向构建时的 `CUDA_HOME`，且 sm_89 是 L40 专用；换机必须重新构建。
4. GPU 运行期不提供硬件缺失以外的自动精度回退；CUDA 内部计算异常原样传播。
5. 交叉点会受输入样本数、n_fft、n_mels、输出大小、GPU 固定启动成本和缓存影响；待下一步以同节点交错测试建模。
6. 当前没有把 CPU-only 和 CPU+CUDA 的所有目标 ISA 组合放入 CI；本地仅验证 AVX2 主机路径，sm_89 仅做编译验证。
