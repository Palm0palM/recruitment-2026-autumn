# FFTW 复现与依赖说明

## 代码与资源隔离

共同 BASE：`15f90558b4ab83ad0e6500e84aa807a942564229`。
正式验证候选：`b4552f3962751b5a02d2aa56798907c5a1c29afc`（后续提交仅增加工具/记录，未改变内核）。
没有修改其他 worktree/master；本地共享 Python 与 tmp/data 只读取，不执行 setup/pip/uv 安装。

```bash
source tmp/session-env.sh       # 本地 worktree 已预备的隔离环境；非 Git 文件
make -B -j4 FFT_MODE=c2c FFTW_PLAN=estimate EXEC=tmp/bin/c2c-estimate
make -B -j4 FFT_MODE=c2c FFTW_PLAN=measure EXEC=tmp/bin/c2c-measure
make -B -j4 FFT_MODE=r2c FFTW_PLAN=estimate EXEC=tmp/bin/r2c-estimate
make -B -j4 FFT_MODE=r2c FFTW_PLAN=measure EXEC=tmp/bin/r2c-measure
git show 15f90558b4ab83ad0e6500e84aa807a942564229:src/melspec.cpp > tmp/build/v1.cpp
g++ -std=c++17 -O3 -march=native -pthread -Iinclude src/main.cpp tmp/build/v1.cpp -o tmp/bin/v1
$PY scripts/smoke.py tmp/bin/r2c-estimate
$PY scripts/fftw_check.py tmp/bin/c2c-estimate tmp/bin/c2c-measure tmp/bin/r2c-estimate tmp/bin/r2c-measure
$PY scripts/fftw_bench.py --runs 3 --out tmp/results/local-compare.json
```

新 clone 没有 session-env 时，自行准备私有 Python 环境并指定 `MELSPEC_DATA`；
不要修改题目 `data` 的集群链接，也不要在已有共享 venv 中安装依赖。
本地系统 FFTW `3.3.11-1`，pkg-config `fftw3`；无 sudo 复现也可使用下述自建库。

## 无 sudo 的集群静态库构建

`scripts/build_fftw.sh` 下载官方 FFTW 3.3.11 tarball（约 4 MB），校验
SHA-256 `5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1`。
源码、独立构建目录、安装前缀均在本实验 tmp 内。默认参数：

```text
CC=gcc CFLAGS="-O3 -march=native"
--enable-static --disable-shared --disable-fortran
--disable-threads --disable-openmp --disable-mpi
--enable-sse2 --enable-avx --enable-avx2
--enable-avx512  # 仅目标 CPU 暴露 avx512f 时加入
--prefix=<本实验目录>/tmp/opt/fftw-3.3.11
```

不需要 root、不依赖 Spack/预装数学库；构建在 Slurm 计算节点进行。
应用传入 `FFTW_PREFIX=<绝对前缀>`，Makefile 链接该目录 `lib/libfftw3.a` 和 libm。
没有链接 FFTW threads/omp 库；`fftw_init_threads` / `plan_with_nthreads` 从未调用。
`ldd` 仅有 libc/libm/libstdc++/libgcc 等运行时，无 FFTW .so 路径依赖。
运行时仍需要目标机兼容的 C/C++ 系统运行库；`-march=native` 库不能随意跨 CPU 搬运，需在目标机重建。

## 集群部署、验证、分析

可用本分支的 Git bundle/clone 部署到新目录。实际部署为
`/home/zsx/melspec-v2-fftw-agent-a`，未覆盖任何已有仓库；不得复用 V1 的部署目录。

```bash
git bundle create tmp/agent-a.bundle v2/fftw
rsync -av tmp/agent-a.bundle hepnode:melspec-agent-a.bundle
ssh hepnode 'git clone -b v2/fftw ~/melspec-agent-a.bundle ~/melspec-v2-fftw-new'
ssh hepnode 'cd ~/melspec-v2-fftw-new && mkdir -p tmp/logs && sbatch --wait --export=ALL,MODE=build scripts/fftw-cluster.slurm'
# 等待前一作业结束，再提交；也可用 afterok 依赖，确保只有一个实际运行。
ssh hepnode 'cd ~/melspec-v2-fftw-new && sbatch --wait --export=ALL,MODE=bench scripts/fftw-cluster.slurm'
ssh hepnode 'cd ~/melspec-v2-fftw-new && sbatch --wait --export=ALL,MODE=profile scripts/fftw-cluster.slurm'
ssh hepnode 'cd ~/melspec-v2-fftw-new && sbatch --wait --export=ALL,MODE=plans scripts/fftw-cluster.slurm'
```

脚本一节点、一 task、16 CPU、nomultithread，不写 mem；本次显式 `-w hepnode2`。
构建/执行全部在作业内。大型输出在 `/tmp/fftw-agent-a-$JOBID-*`，trap 仅清理自己创建的 scratch。
库和小日志保存在实验 home，供后续复测；官方 `validate.py` 不做修改，包装层检查 4/4 PASS。
MODE=bench 同时保存 V1/A1/A2 五版本 interleaved 比较；V1 由 BASE 导出并在同节点重新编译，绝不运行 V0。
集群 Python 使用已有 `~/venv/bin/python`，作业设置独立 pycache/Numba 缓存与 16 线程对照库上限。

本地重新 profile：

```bash
source tmp/session-env.sh
$PY scripts/fftw_profile.py --mode r2c --planner estimate
git clone --depth 1 https://github.com/brendangregg/FlameGraph.git tmp/tools/FlameGraph
$PY scripts/fftw_perf.py --out tmp/perf-fftw-new
```

perf 使用用户态软件时钟，不依赖 WSL 不支持的 PMU；自动拒绝丢样结果。
系统 FFTW 内部符号可能被剥离，局部符号缺失属于分析限制，不能伪造函数名。

## 许可证与归档

FFTW 为 **GPL-2.0-or-later**（官方 tarball 的 COPYING、源文件许可头；本地包 metadata 同样标注）。
分发静态链接可执行文件时，需核对原题代码的许可兼容性、保留版权/许可并履行 GPL 相应源码与构建材料义务；
不能仅随二进制带一个 URL 就假定所有义务已满足。本分支不替原题重新授权，也不发布可执行文件/库。
源码 tarball、COPYING 原文、依赖构建日志保留在本实验 tmp/deps；下载地址与完整复现脚本入库。
FlameGraph 为 Brendan Gregg 上游工具，生成图注明其来源/commit，工具 clone 仅留 tmp。

归档只含需要展示的文本/JSON/SVG，不提交 venv、音频、expected、目标文件、库、可执行文件或 perf.data。
原始文本归档可去除行尾空格及多余 EOF 空行，数值不做修改；原始未规范化输出仍在 tmp/logs。
