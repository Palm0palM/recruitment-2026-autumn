# V3 AMD / ROCm 初始化环境

## 1. 基线与工作区

- Local worktree：`/home/palm/projects/heptagon/melspec-v3-amd`
- Branch：`v3/amd`
- Inherited frozen CUDA commit：`58bca045f58461a2f53f8dcadb01e75729b2f008`
  （来自已推送 `v3/cuda`，不是有未提交改动的 `v3/cuda-sprint`）
- Remote private deployment：`/home/zsx/melspec-v3-amd`（仅本路线目录）
- Protected files unchanged：`src/main.cpp`、`validate.py`、`data`、`include/melspec.h`

AMD 资源和探测背景见用户提供的 [`../_cluster/amd-r9700.md`](../_cluster/amd-r9700.md)；
原始设备/HIP 证据已纳入本分支 `versions/_cluster/raw-amd-r9700/`。

## 2. 可复用与必须替换的设计

| CUDA 结构 | AMD/HIP 处理 |
|---|---|
| 整段 y 一次 H2D、GPU 直接 gather 重叠帧/周期 Hann | 算法可复用；`hipMalloc/hipMemcpy` 替换 |
| cuFFT batched R2C/D2Z，自然 `[batch][n_fft]` / `[batch][n_fft/2+1]` | 换 hipFFT/rocFFT；plan/batch/work area 需重新验证 |
| fused power+Mel warp reduction、first/last 稀疏区间、row-major 输出 | 数学与布局可复用；warp API/编译目标换 HIP |
| 16 outer workers、无内部库线程 | 可复用 |
| CUDA stream/event/pinned 异步 D2H + CPU double 转换 | HIP stream/event/`hipHostMalloc` 对应实现，需重新 A/B |
| `cudaFree(0)` context 初始化和 resize 线程 | HIP runtime 初始化语义不同，不能假设同一收益 |
| Batch512、FP32 主线、FP64 诊断 | 参数可复用，但 R9700 的 hipFFT 实际 batch 曲线需重测 |
| `sm_89`/`sm_120` | 必须换成 `gfx1201` |
| CUDA warpSize=32 假设 | R9700 wavefront 也是 32，数学 lane 假设可保留，但 intrinsic 名不同 |
| CUDA Graph / unified memory 等 | 未验证，不能继承假设 |

## 3. 实际 GPU 与 Slurm 边界

来自本 worktree 的 Slurm 作业 **329**（hepnode0，`COMPLETED / 0:0`，4 秒）：

- GPU：AMD Radeon AI PRO R9700，`gfx1201`，PCI `b3:00.0`，Navi 48。
- VRAM：`34,208,743,424 B` ≈ 31.86 GiB。
- HIP device properties：`multiprocessors=32`、`wavefront=32`、runtime/driver `70152801`。
- L2 8 MiB、L3 64 MiB；rocm-smi 报告低功耗状态，正式测量前必须预热。
- 作业实际 cpuset：`0-3,64-67`；Slurm `AllocTRES` 记录 8 CPU（请求 4）。
- **hepnode0 没有 GPU GRES**。不能写 `--gres=gpu` 或 `--gpus`。
  普通 Slurm 作业 `-w hepnode0` 独占节点后使用 GPU；不写 `--mem`；禁止登录 Shell 计算。
- 当前其它 GPU/CPU 作业也争用 hepnode0 的单一作业槽，必须串行、短作业、用完退出。

## 4. 私有 ROCm 环境

无 sudo、无容器、不改共享 Python。`scripts/setup_amd_rocm.sh` 用 `apt download`
和 `dpkg-deb -x` 将固定候选版本的 Ubuntu 26.04 ROCm 包解包到：

```text
/home/zsx/melspec-v3-amd/tmp/amd-gpu/root/   # 约 162 MiB
  usr/bin/hipcc
  usr/include/hip/, usr/include/hipfft/
  usr/lib/llvm-21/.../amdgcn/bitcode/
  usr/lib/x86_64-linux-gnu/
```

实测关键库：

| 组件 | 版本/状态 |
|---|---|
| HIP runtime | 7.1.52801-9999 |
| hipcc/clang | 7.1.1 / Ubuntu clang 21.1.8 |
| ROCm userspace packages | 7.1.1（ROCm SMI 7.2.0，HSA 7.1.0） |
| hipFFT | 7.1.1，`libhipfft.so.0` |
| rocFFT | 7.1.1，`librocfft.so.0` |
| HIP device libs | LLVM 21 `amdgcn/bitcode` |
| rocminfo / rocm-smi | 可用 |

`tmp/amd-gpu/env.sh` 设置 `ROCM_PREFIX`、`ROCM_PATH`、`HIP_PATH`、`HIP_DEVICE_LIB_PATH`、
`HIPCC`、`HIP_ARCH=gfx1201`、`PATH`、`LD_LIBRARY_PATH`、`LIBRARY_PATH`、`CPATH`。
完整包清单见 `raw/package-manifest.txt`，安装原始日志见 `raw/amd-setup.txt`。
不提交 .deb、ROCm 库、二进制或日志大文件。

## 5. 构建入口

```bash
cd /home/palm/projects/heptagon/melspec-v3-amd   # 或远端同一项目目录
bash scripts/setup_amd_rocm.sh
source tmp/amd-gpu/env.sh

# 生产 HIP 入口（G2 实现流水线，G3 完成系统优化，见 FINAL_AMD.md）
make -B -j4 HIP=1 EXEC=tmp/bin/melspec-hip

# 官方 V1 CPU 对照
make -B -j4 HIP=0 CUDA=0 EXEC=tmp/bin/melspec-cpu

# 独立环境 probe
make -B hip-probe
```

实测构建：

- `HIP=1` 成功编译并链接 `libamdhip64/libhipfft/librocfft`，
  二进制 SHA-256 `2ee3e94e43110103ecdd153f1c97db5a49c92d6a57ed640f3aed27520b122842`。
- `HIP=0 CUDA=0` 成功，二进制 SHA-256
  `91c160a230cef2aa32c35322270dff5986349f8d9962effca3f2fa537b8c1e0e`；
  本地 8 项 CPU smoke PASS。
- HIP 编译参数：`--offload-arch=gfx1201 --rocm-path=$ROCM_PATH
  --rocm-device-lib-path=$HIP_DEVICE_LIB_PATH`，host 侧 `-O3 -march=native -pthread`。
- G2 已将 `src/melspec_hip.cu` 替换为完整 HIP Melspec 后端，并通过官方四组；
  初始性能、实现细节与限制见 [G2-MELSPEC.md](G2-MELSPEC.md)。
- 初始 `-Xcompiler -O3,-march=native,-pthread` 被 hipcc 当成了非法整数值，调整为 clang 直接参数后成功。

## 6. 实机验证证据

```text
hip_devices=1 runtime=70152801 driver=70152801
name=AMD Radeon AI PRO R9700 arch=gfx1201 mem=34208743424 CUs=32 wavefront=32
HIP vector-add: PASS
hipFFT R2C: PASS
```

这是作业 329 的真实输出，证明 HIP runtime、显存访问、GPU kernel 与 hipFFT 不只是枚举。
原始 `rocminfo`、`rocm-smi`、probe、构建日志和包清单均在 `raw/`。

## 7. 生产配置与已解决/遗留问题

G3 后的默认构建参数（`make -B -j4 HIP=1`）：

| 参数 | 值 | 依据 |
|---|---|---|
| `HIP_PRECISION` | 32 | G1：FP64 明显更慢 |
| `HIP_BATCH` / `HIP_BATCH_LARGE` | 512 / 256 | G1 曲线 + 端到端复核 |
| `HIP_FUSED` | 1 | 不落功率矩阵 |
| `HIP_PINNED_INPUT` | 0 | 实测负优化（写 pinned 只有 ~2.4 GB/s） |
| `HIP_ASYNC_H2D` | 1 | 避免同步拷贝路径 |
| `HIP_PREWARM` | 1 | 进程级 GPU 初始化与 PCM 读取重叠，省 ~190 ms |
| `HIP_HUGEPAGE` | 1 | 对齐 `MADV_HUGEPAGE`，省 ~70–80 ms |
| `HIP_DIRECT_OUTPUT` | 1 | Mel 直接写 double，一次 D2H，省 ~50 ms |
| `HIP_RESIZE_THREAD` | 1 | 394 MB output 初始化与 GPU 计算重叠 |
| `HIP_STAGING_ROWS` | 16 | 仅在关闭 direct output 时使用 |

已解决：

1. G2 完成 gather/Hann、hipFFT batched R2C/D2Z、fused power+Mel、warp 归约与官方四组。
2. G1 的 batch 结论已端到端采用（8192→512，16384→256）。
3. pinned/async D2H、stream/event、resize/context 初始化重叠均已实测并取舍。
4. 已增加 HIP 专用阶段计时（`HIP_PROFILE=1`），足以定位到 H2D/plan/gather/FFT/Mel/D2H。
5. rocFFT 持久 kernel 缓存已作为环境前提写入作业脚本（`ROCFFT_RTC_CACHE_PATH`）。

遗留：

1. FP64 仍未做完整官方回归（主线是 FP32）。
2. ROCm profiler/rocprof 未安装，暂不能依赖硬件计数器。
3. hepnode0 无 GPU GRES，只能独占节点串行使用。
4. Ubuntu 26.04 不在 ROCm 官方 OS 支持表内；任何新库/升级都必须重新 probe。
