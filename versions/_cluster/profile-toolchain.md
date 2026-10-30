# 集群 profiling 工具链现状（2026-10-30 实测）

> 起因：`perf` 在集群上不可用，需要确认原因并找出可替代的 profiling 手段。
> 原始日志：`tmp/perf-probe.log`、`tmp/perf-alt.log`（不进 git）

## 1. 结论

**集群上 `perf` 完全不可用，且无法自行绕过。** 根因是 `kernel.perf_event_paranoid = 4`。

## 2. 证据

```
$ cat /proc/sys/kernel/perf_event_paranoid
4
$ perf stat -e cycles true
      Ignore mlock limit after perf_event_mlock_kb without CAP_IPC_LOCK
>= 2: Disallow kernel profiling          ← perf 的出错提示
$ perf record -o /tmp/x.data -- true
Error: Failure to open any events for recording.
```

直接调系统调用（`exclude_kernel=1` 也不行）：

```
perf_event_open(cpu cycles, exclude_kernel=1) -> -1  errno=13 (Permission denied)
perf_event_open(task-clock)                   -> -1  errno=13 (Permission denied)
```

`perf_event_paranoid` 语义：`-1` 全开 / `0` 禁 raw tracepoint / `1` 禁 CPU 事件 /
`2` 禁内核采样（用户态仍可）/ **`≥3` 禁一切 `perf_event_open`**。集群是 4。

无法绕过的原因：

| 绕过路径 | 实测结果 |
|---|---|
| 进程 capabilities | `capsh --print` → `Current: =`（有效集为空；`cap_perfmon` 只在 bounding set） |
| perf 二进制 file capability | `getcap /usr/bin/perf` → 空 |
| sudo | `Sorry, user zsx may not run sudo on hepnode0.` |
| bpftrace（eBPF 路线） | `Error reading events from /sys/kernel/tracing/available_events (Permission denied)` |
| 内核日志 | `dmesg_restrict = 1` |

注意：`perf stat` 不带 `-e` 时输出的 `seconds user / seconds sys` **来自 `getrusage`**，不是 perf 计数器，
容易误以为 perf 还能用。

## 3. 可用的替代手段（均已在本机/集群实测）

| 手段 | 位置 | 实测 | 用途 | 局限 |
|---|---|---|---|---|
| `__rdtsc()` + `clock_gettime` | 集群 ✅ | TSC ≈ 2.00 GHz | 分阶段 cycle/时间，算每帧 cycle、GFLOP/s | 需自己插桩 |
| `getrusage` / `/proc/self/stat` | 集群 ✅ | utime/minflt 正常 | CPU 时间、**页面故障（发现 TLB/大数组问题）** | 只有全局值 |
| `/proc/self/schedstat` | 集群 ✅ | 有值 | 线程 runqueue 等待（判断超订/绑核错误） | 每进程累积 |
| `strace -c -f` | 集群 ✅ | 正常 | 系统调用耗时（I/O 占比） | 看不到计算热点 |
| `gprof -pg` | 集群 ✅ | 能定位到函数级 | 单线程函数级热点 | OpenMP 下不准、内联不可见 |
| `perf record/report` | **本地 WSL** ✅（paranoid=2） | 可用 | 火焰图、热点 | 本地 CPU 无 AVX-512，结论需回集群复验 |
| `bpftrace` / `valgrind` / `gdb` | ❌ | — | — | 权限或未安装 |

## 4. 由此确定的工作方式

1. **热点定位在本地做**（`perf record -g` + 火焰图）：本地 `perf_event_paranoid = 2`，可用。
   但本地是 i7-14650HX（**仅 AVX2，无 AVX-512**），集群是 Xeon Gold 6338（AVX-512），
   向量化相关结论必须在集群重新 A/B 验证。
2. **集群只做两件事**：正确性（`validate.py`）+ A/B 计时（`rdtsc`/`chrono` 分阶段）。
3. 分阶段计时至少覆盖：读入 / 分帧加窗 / FFT / 功率谱 / Mel 投影 / 写盘，
   并把 `minflt`、`schedstat` 一起记下来（后者能直接暴露「超线程绑核错误」导致的排队）。
4. 若确实需要集群上的 perf：只能找管理员在**计算节点**上
   `sysctl -w kernel.perf_event_paranoid=1`，或 `setcap cap_perfmon+ep /usr/bin/perf`。
   `perf_event_paranoid` 是 per-node 的，改登录节点不影响计算节点。

## 5. 对已完成的 V0 分析的影响

V0 的热点结论（Mel 投影占 94%）是在**本地**用 `perf record -g` + `n_mels` 扫描得到的，
其「Mel 稠密乘远慢于 FFT」这一结论属于算法/访存层面的定性结论，
不受 CPU 架构差异影响，可以直接迁移到集群；但具体倍率要以集群 A/B 为准。
