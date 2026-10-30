# MPI 环境与正式评测兼容性

调查日期：2026-10-30。主 Agent 负责所有 SSH/Slurm 操作；未在登录节点运行计算负载。

## 已验证的 MPI 运行环境

| 项目 | 实际证据 |
|---|---|
| Slurm | 25.11.2；MPI 插件 `none/cray_shasta/pmi2/pmix`，PMIx v5 |
| OpenMPI | `/cache/toolchain/openmpi-4.1.6`，Open MPI 4.1.6 |
| UCX | `/cache/toolchain/ucx-1.18.1`；系统 `ucx_info` 可运行 |
| wrapper | `mpicxx` 写死 `/opt/toolchain/openmpi-4.1.6`，直接调用失败；需要手工 g++ 编译/链接或修正私有 wrapper |
| 动态库 | 需 `LD_LIBRARY_PATH=/cache/toolchain/openmpi-4.1.6/lib:/cache/toolchain/ucx-1.18.1/lib`；设置 `OPAL_PREFIX/OPAL_LIBDIR` |
| singleton | 作业 98 在 hepnode3 直接运行 1 Rank，PASS |
| srun | 默认直接 `srun` 失败；`srun --mpi=pmix` 单节点 2 Rank、双节点各 1 Rank 均 PASS |
| mpirun | 单节点 2 Rank、双节点各 1 Rank 均 PASS；默认路径和显式 ob1/TCP 路径均通过 hello 测试 |
| 双节点 | 作业 103 分配 hepnode[2-3]，OpenMPI Rank 0/1 分别落在两个节点 |

hello 程序只执行 `MPI_Init`、打印 rank/size/host、barrier、finalize。上述结果证明 MPI 启动可行，
**不代表实际 Melspec 的通信性能、UCX 最优传输或强扩展收益已经证明**。

### 可复现环境要点

```bash
O=/cache/toolchain/openmpi-4.1.6
U=/cache/toolchain/ucx-1.18.1
export PATH="$O/bin:$PATH"
export LD_LIBRARY_PATH="$O/lib:$U/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$O" OPAL_LIBDIR="$O/lib"

# 在 Slurm 作业内编译；不要依赖写死 /opt 路径的 mpicxx。
g++ -std=c++17 -O3 -I"$O/include" probe.cpp \
  -L"$O/lib" -Wl,-rpath,"$O/lib" -lmpi -o probe

# 两种已验证启动方式：
srun --mpi=pmix -N2 -n2 --ntasks-per-node=1 -c16 --hint=nomultithread ./probe
"$O/bin/mpirun" -np 2 --map-by ppr:1:node:pe=16 --bind-to none ./probe
```

不写 `--mem`，因为集群存在 `RealMemory=1` 异常。一节点同时只能一个作业；
多节点性能测试在每节点 1 Rank、16 CPU 下串行提交。PMIx 的 munge/gds 警告出现在 stderr，
但成功路径返回 0，原始成功与失败日志都保留，不能把警告数量当成性能或正确性结论。

## validate.py 与多 Rank 的正式兼容性

`validate.py` 对二进制直接构造并执行：

```text
[program_path, data/audio.pcm, temporary_output, official options...]
```

没有 shell、MPI launcher、节点数或 Rank 映射参数。原始 `src/main.cpp` 也没有 `MPI_Init`、
Rank 分支或单一写出者。若直接 `mpirun -np N ./melspec`，每个进程都会读输入、计算并截断写同一路径，
会产生输出竞争，不能用“Rank 内容可能相同”来宣称正确。

`scripts/mpi-official-wrapper.sh` 可以在**受控 Slurm 研究作业内**为独立 driver 提供 launcher，
使未修改的 `validate.py` 检查汇集输出。但这证明的是 wrapper + 独立 driver 的研究口径，
并不证明赛事最终评测允许额外启动器或接受该入口。

**结论：目前只能将本路线标记为 MPI 实验原型，不能声称是不改受保护文件即可直接提交的最终方案。**
若最终只能证明多节点性能而没有合法正式启动路径，必须明确保留这一限制，不将其包装成正式加速。

## 证据文件

- `raw/mpi-probe-98.out/.err`：单节点 singleton、srun、mpirun 探测。
- `raw/mpi-two-node-103.out/.err`：双节点两种 launcher 探测。
- `raw/slurm-accounting.txt`：作业状态、节点、CPU 和耗时。

Kimi researcher 的只读审查还确认：系统 PATH 无 `mpicxx/mpirun`，历史 wrapper/PMIx 故障需要实测，
不能沿用“有 OpenMPI 目录”就推断可用。三个研发子 Agent 均使用项目配置的
`ppio/moonshotai/kimi-k3.1-pro-highspeed#high`，未自行 SSH 或提交作业。
