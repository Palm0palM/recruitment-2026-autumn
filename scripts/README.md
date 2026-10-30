# scripts/ —— 本地与集群的评测/分析工具

这些脚本是**工作流基础设施**，不参与被测代码本身；所有产物都落在 `tmp/`（已被 `.gitignore` 忽略）。

## 一次性配置

```bash
./scripts/setup.sh          # 建 uv venv（numpy/scipy/librosa）+ 链接数据 + 编译
source scripts/env.sh       # 之后可用 $PY、$ROOT
```

- Python 环境在 `tmp/.venv`（`uv venv --python 3.12` + `numpy scipy librosa`）。
- 仓库自带的 `data` 是指向集群 `/cache/recuritment/melspec/data` 的符号链接，本地（WSL）通常是断的；
  `setup.sh` **不会改这个链接**，而是在 `tmp/data/` 下建可用的副本链接。
  数据来源优先级：`$MELSPEC_DATA` > `./data` > `~/projects/heptagon/autumn26/cluster_melspec/data`。
- `clang-format` 通过 `uv tool install clang-format` 提供（`~/.local/bin/clang-format`），
  用于 `make format`。

## benchmark

```bash
$PY scripts/bench.py --case default --slice 5     # 开发期：截断 5 秒，秒级反馈
$PY scripts/bench.py --case all --runs 3          # 本地全量（朴素实现约十几分钟/组）
$PY scripts/bench.py ./melspec --case all --out tmp/bench/step2.md
```

参数组直接 import 自 `validate.py` 的 `TEST_CASES`，与官方一致：

| 用例 | n_fft | hop | n_mels | f_min | f_max | 输出 shape |
|---|---|---|---|---|---|---|
| default | 8192 | 128 | 256 | 0 | 22050 | (256, 192397) |
| small_mels | 8192 | 128 | 64 | 0 | 22050 | (64, 192397) |
| large_fft | 16384 | 128 | 256 | 0 | 22050 | (256, 192333) |
| narrow_band | 16384 | 128 | 256 | 300 | 8000 | (256, 192333) |

- 截断模式用 librosa 现算参考值；全量模式与 `data/expected/*.bin` 对比（`rtol=1e-3, atol=1e-4`）。
- 全量模式下默认**不跑** librosa（其峰值内存约 25 GB），需要时加 `--force-librosa`。

## 集群

```bash
# 登录节点：只编译与提交
git pull && make -j
mkdir -p logs
sbatch scripts/bench.slurm        # 正式成绩：计算节点上跑官方 validate.py
sbatch scripts/profile.slurm      # perf stat/record + 火焰图
```

- `VARIANT=./melspec_v3 sbatch scripts/bench.slurm` 可以指定被测二进制。
- 集群 Python 依赖用 spack（`spack load python py-numpy py-librosa`）或自己建 uv venv，
  通过 `PY=/path/to/python sbatch ...` 传入。
- 注意：`sbatch` 会读 `#SBATCH` 头里的 `--output=logs/...`，所以**先 `mkdir -p logs`**。
- 禁止在登录节点跑全量；`profile.slurm` 默认单例（`-n 1`）采样，避免线程干扰。

## 本地环境实测（2026-10-30）

| 项 | 版本/状态 | 备注 |
|---|---|---|
| CPU | i7-14650HX，12C/24T | 仅 AVX2 + FMA，**无 AVX-512** |
| 内存 | 15 GiB + 4 GiB swap | default 全量朴素实现 RSS ≈ 6.7 GB；large_fft 需 12.6 GB，本地会 swap |
| GPU | RTX 5060 Laptop | CUDA 13.4（`/opt/cuda/bin/nvcc`） |
| 编译器 | g++ 16.2.1 | `make` 已带 `-O3 -march=native -fopenmp` |
| FFTW | 3.3.11（double/float/omp/threads） | `-lfftw3 -lfftw3_omp` 本地实测可用 |
| BLAS | **无 OpenBLAS/cblas** | 需要时 `sudo pacman -S openblas` 或集群 spack 取用 |
| Python | venv 内 3.12.13 | numpy 2.5.4 / scipy 1.18.1 / librosa 1.0.0 |
| clang-format | 23.1.3（`uv tool`） | 现有源码格式已符合 `.clang-format`，`make format` 无 diff |
| perf | 可用（`perf_event_paranoid=2`） | 用户态采样可用，内核态不可 |

集群：`ssh hepnode`（222.20.97.229:2200）在 2026-10-30 上午不可达（`No route to host`），
建议开工前先 `ssh hepnode 'hostname; nproc; free -g'` 确认节点规格与 spack 里的 fftw/openblas 版本。
