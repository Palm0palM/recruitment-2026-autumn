# MPI 数据划分、输出顺序与最小原型设计

## 1. 按帧而不是按 PCM 字节均分

设全局样本数 `N`、FFT 长度 `n=n_fft`、步长 `h=hop_length`、总帧数：

```text
F = 1 + floor((N-n)/h),  N >= n
```

Rank `r`（共 `P` 个）负责：

```text
begin_r = floor(F*r/P)
end_r   = floor(F*(r+1)/P)
count_r = end_r-begin_r
```

非空 Rank 的 PCM 切片：

```text
sample_begin = begin_r*h
L            = (count_r-1)*h+n
sample_end   = sample_begin+L = (end_r-1)*h+n
```

代入现有 worker 的帧数公式：`1+(L-n)/h = count_r`，且第 `u` 个局部帧恰好是全局第 `begin_r+u` 帧。
因此无需 padding、裁帧或修改 FFTW 数学内核。`count=0` 的 Rank 不读 PCM、不调用 DSP，只参与集合通信。
`N<n` 必须失败，不能静默输出空文件。`hop>n_fft` 时切片之间允许有空洞，不能补成连续音频。

相邻 Rank 在 `n>=h` 时只重复右侧 halo：`n-h` 个样本。对全体非空 Rank，总局部样本量为
`sum L = F*h + K*(n-h)`，其中 `K=min(P,F)`。任何分发方式都必须保留尾部不足一帧样本的忽略语义。

## 2. 三种输入路径

- **bcast**：root 读取并归一化完整 PCM，广播完整 double `y`。控制简单，但每个 Rank 持有完整输入，流量和内存最大。
- **scatter**：root 读取完整 `y`，按上述公式向各 Rank 发送带 halo 的局部 double 切片。非 root 仅保留局部输入。
- **shared**：各 Rank 从共享文件系统 seek 并读取自己的局部 int16 区间，再除以 32768.0 转为 double。
  没有 MPI PCM 消息，但真实文件系统吞吐、缓存和 seek 必须实测。

三条路径使用完全相同的帧区间和 worker，用于分离通信、I/O 和转换成本，而不是根据音频内容选择算法。

## 3. 输出汇总：单次 Gatherv + Derived Datatype

局部 double 输出为 `[M,count_r]` 行主序。**不能直接把该缓冲区交给下面的列接收类型**：
接收类型按“帧列”消费数据，因此先在每个 Rank 打包为 `[count_r,M]`：

```text
packed[t*M+m] = float(result[m*count_r+t])
```

这次打包同时完成 double→float32，与单进程最后转换的舍入一致，把通信量减半；
转换和布局成本单独计时，不能藏进网络带宽或忽略。

root 创建一个接收类型：

```cpp
MPI_Type_vector(M, 1, F, MPI_FLOAT, &column);
MPI_Type_create_resized(column, 0, sizeof(float), &column_resized);
```

设置：

```text
recvcounts[r] = count_r
displs[r]     = begin_r
```

因为 resized extent 恰为一个 float，第 `j` 个接收实例的第 `m` 个元素落在：

```text
global[m*F + begin_r + j]
```

这正是局部 `result[m*count_r+j]` 应到的位置；发送流则来自 `packed[j*M+m]`。
非整除和空 Rank 都使用同一 receive type。
**MPI_Gatherv 只有一个 receive datatype，且 displacement 以 resized extent 计**；
若改用每 Rank 一个 extent=count 的类型，就不能直接套上述 displacement，属于另一套接口/证明。

root 是唯一写出者，最终 buffer 为 float32 小端 `[M,F]` 行主序，无文件头。
任何 Rank 抛错都调用 `MPI_Abort`，不允许 root 写出貌似成功的部分输出。

## 4. 计时与实验边界

独立 `scripts/mpi_melspec.cpp` 记录每 Rank：MPI 初始化、读取、分发、worker、float 转换、
Gather、写出、同步、Finalize 和进程时间；外部计时包住整个 `mpirun`，还包含 launcher。
正式比较使用同作业、同节点池、同编译参数的 A2 单进程与 MPI 1/2/4 Rank。

worker 仍是固定 FFTW A2 ESTIMATE，没有改 FFT 数学内核。原型不修改 `src/main.cpp`、`validate.py` 或 `data`。
研究 wrapper 不等于正式评测启动资格，结论须与性能分开报告。

## 5. 正确性矩阵

小规模检查覆盖：1/2/4 Rank；三种输入路径；随机官方参数；精确 1/2 帧与空 Rank；
非整除；`hop>n_fft`；`n_fft=1/8`；短输入失败。每个合法用例与独立 A2 串行输出逐字节一致，
再进入完整官方 expected 和强扩展测试。

子任务完整分析见 `research/`。它们是设计材料，不冒充实测结果。
