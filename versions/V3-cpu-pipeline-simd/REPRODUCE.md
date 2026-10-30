# 复现与移植

## 1. 控制与源码补丁

- 冻结控制：`e448c989c921762884582804efe23d6607678ff9`。
- slab独立提交：`3011dbf`；HP独立提交：`ce56b4c`；安全修复：`cbf214d`。
- `patches/cpu-pipeline-simd.patch` 仅包含 src/melspec.cpp 差异，可应用到冻结点。
  不需要带上实验历史的大批raw日志。两种优化默认关闭，安全修复无条件生效。
- 标准C++17，无新链接依赖；slab仅作用于已有AVX-512 B16 fast path。
  HP仅在Linux有MADV_HUGEPAGE时编译，不支持/提示失败时正常resize。

应用前先检查，避免覆盖其他研发工作：

```bash
git apply --check /path/to/cpu-pipeline-simd.patch
git apply /path/to/cpu-pipeline-simd.patch
```

## 2. 正式二进制（必须在AVX-512节点）

```bash
mkdir -p tmp/bin tmp/control
git show e448c989:src/melspec.cpp > tmp/control/melspec.cpp
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  -DMELSPEC_FP32_BATCH=16 src/main.cpp tmp/control/melspec.cpp -o tmp/bin/frozen-b16
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  -DMELSPEC_FP32_BATCH=16 -DMELSPEC_PACK_REUSE=1 \
  src/main.cpp src/melspec.cpp -o tmp/bin/reuse-b16
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  -DMELSPEC_FP32_BATCH=16 -DMELSPEC_OUTPUT_HUGEPAGE=1 \
  src/main.cpp src/melspec.cpp -o tmp/bin/hugepage-b16
g++ -std=c++17 -O3 -march=native -pthread -Iinclude \
  -DMELSPEC_FP32_BATCH=16 -DMELSPEC_PACK_REUSE=1 -DMELSPEC_OUTPUT_HUGEPAGE=1 \
  src/main.cpp src/melspec.cpp -o tmp/bin/combined-b16
```

本机无AVX-512：只能 `-march=icelake-server` 交叉编译，不能执行该二进制。
通用回退使用 `-march=x86-64`；AVX2 B8需显式 `MELSPEC_FP32_BATCH=8`。
不要用普通make误替代FP32 B16控制，不启用fast-math。

## 3. 环境与校验

使用既有只读Python环境与数据链接；不执行setup.sh、不安装依赖。
当前实验本地 tmp/.venv、tmp/data 指向共享资源，tmp/session-env.sh 指向本worktree的缓存。
集群分别只读链接 ~/venv 与 /cache/recuritment/melspec/data。
所有Python前：

```bash
source tmp/session-env.sh
"$PY" scripts/cpu_pipeline_check.py tmp/bin/frozen-b16 tmp/bin/reuse-b16 --exact
"$PY" scripts/cpu_pipeline_check.py tmp/bin/frozen-b16 tmp/bin/hugepage-b16 --exact
"$PY" scripts/cpu_pipeline_check.py tmp/bin/frozen-b16 tmp/bin/combined-b16
"$PY" scripts/multi_bench.py frozen=tmp/bin/frozen-b16 reuse=tmp/bin/reuse-b16 \
  --runs 8 --out tmp/logs/reuse-ab.json
"$PY" validate.py tmp/bin/combined-b16 --runs 3
```

官方脚本即使FAIL仍可能返回0，包装层必须检查 `Result: 4/4 PASSED`。
独立check会校验68组任意参数/合成音频、全部16种B16尾数、超一页/多MiB输出和多worker，
单项配置要求冻结B16 bit-exact，组合按官方原始容差验收（尾帧微小差异见INTEGRATION.md），
另外对照librosa（长度1按题目Hann公式单独处理）。
补充API测试直接传非PCM量化的double和非空/带capacity的output，不只测空vector的CLI：

```bash
g++ -std=c++17 -O1 -g -march=native -pthread -Iinclude \
  -DMELSPEC_FP32_BATCH=16 -DMELSPEC_PACK_REUSE=1 -DMELSPEC_OUTPUT_HUGEPAGE=1 \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  scripts/cpu_pipeline_interface_check.cpp src/melspec.cpp -o tmp/bin/sanitize
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1 tmp/bin/sanitize
```

不能仅根据sanitizer默认退出码判断通过；首轮警告与修复证据保留在SAFETY.md及raw/。

## 4. 全阶段Profile

```bash
source tmp/session-env.sh
"$PY" scripts/cpu_pipeline_profile.py --source tmp/control/melspec.cpp --tag control --runs 3
"$PY" scripts/cpu_pipeline_profile.py --tag reuse --runs 2 --cppflag=-DMELSPEC_PACK_REUSE=1
"$PY" scripts/cpu_pipeline_profile.py --tag combined --runs 2 \
  --cppflag=-DMELSPEC_PACK_REUSE=1 --cppflag=-DMELSPEC_OUTPUT_HUGEPAGE=1
```

输出含read/write钩子、output resize、setup、每worker四阶段、tail、/usr/bin/time RSS/faults，
以及smaps_rollup的AnonHugePages。只在tmp生成DSP插桩副本，main直接编译原文件。
线程累计秒不能直接当作进程wall；插桩二进制绝不用于正式A/B。

## 5. 串行集群作业

专属目录 `~/melspec-cpu-pipeline-simd`；每次 `sbatch --wait` 完成后再提交下一项：

```bash
sbatch --wait -w hepnode3 --export=ALL,MODE=profile scripts/cpu_pipeline_simd.slurm
sbatch --wait -w hepnode3 --export=ALL,MODE=reuse scripts/cpu_pipeline_simd.slurm
sbatch --wait -w hepnode3 --export=ALL,MODE=reuse-confirm scripts/cpu_pipeline_simd.slurm
sbatch --wait -w hepnode3 --export=ALL,MODE=hugepage scripts/cpu_pipeline_simd.slurm
sbatch --wait -w hepnode3 --export=ALL,MODE=final scripts/cpu_pipeline_simd.slurm
sbatch --wait -w hepnode3 --export=ALL,MODE=combined-confirm scripts/cpu_pipeline_simd.slurm
```

脚本1task/16CPU/nomultithread、不写mem，使用节点本地独立scratch。
MODE=final重新测试安全修复后的三个配置、严格sanitizer、独立尺寸性能、组合增量和官方验证。
源码/控制/main/validate SHA与编译器/拓扑均打印；不共享其他实验构建产物。
HP收益需要目标Linux/THP配置与内存状态，不能承诺其他平台同幅度收益。
