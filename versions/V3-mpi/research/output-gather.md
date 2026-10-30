# V3 MPI 输出汇集原型：局部连续帧到全局行主序

> **主 Agent 复核更正：** 本文原始证明遗漏了发送布局：局部 `[M,count]` 行主序不能直接被
> `vector(M,1,F)` 的列接收类型消费。正式原型先打包为 `[count,M]`，再执行 Gatherv；
> 正确公式和成本见 `../DESIGN.md`。本文其余关于 extent、recvcounts/displs 和空 Rank 的分析保留。

> 仅作设计说明，未编译、未运行、未测试，不包含性能结论。

## 1. 已核对的现有语义

- `src/melspec.cpp` 以 `output[m * n_frames + t]` 写结果，布局为行主序 `[n_mels, n_frames]`；算法过程使用 `double`。
- `src/main.cpp::write_output` 最后逐元素 `static_cast<float>`，再连续写出，无文件头。
- `include/melspec.h` 同样声明最终二进制为 float32 小端行主序。
- `validate.py` 用 `np.frombuffer(..., dtype="<f4").reshape(shape)` 检查精确形状和顺序，容差为 `rtol=1e-3, atol=1e-4`。
- `versions/V3-mpi-spike/README.md` 要求汇集顺序确定，输出为 float32 小端、无文件头的 `[n_mels, n_frames]`。

记：

- `M = n_mels`，`F = n_frames`，`P = MPI communicator size`；
- rank `r` 的连续全局帧区间为 `[b_r, e_r)`，`c_r = e_r - b_r`；
- 局部 double/float 数组均为连续行主序 `[M, c_r]`，即
  `L_r[m * c_r + j]` 对应全局元素 `G[m * F + b_r + j]`。

固定划分可直接使用：

```text
b_r = floor(F * r / P)
e_r = floor(F * (r + 1) / P)
c_r = e_r - b_r
```

计算时用 64 位中间值，避免 `F * r` 的 `int` 溢出。该划分天然覆盖 `[0,F)`，不需要额外 Gather 帧数；非整除时各块大小最多相差 1，`F < P` 时允许出现空区间。

## 2. 推荐的通用方案：一次 `MPI_Gatherv`

### 2.1 关键类型：`count = 1` 的 vector tile

仅 root 创建：

```cpp
MPI_Datatype column, column_resized;
MPI_Type_vector(M, 1, F, MPI_FLOAT, &column);
MPI_Type_create_resized(column, 0, (MPI_Aint)sizeof(float),
                        &column_resized);
MPI_Type_commit(&column_resized);
MPI_Type_free(&column);
```

这正是题目指定的

```text
MPI_Type_vector(n_mels, count, n_frames, MPI_FLOAT)
+ resized extent = count floats
```

在 `count=1` 时的形式。一个 `column_resized` 实例表示全局输出的一整个帧列：

- 有 `M` 个 block；
- 每个 block 只有一个 float；
- block 间距为 `F` 个 float；
- resized extent 为一个 float，因此第 `j` 个实例相对前一实例只前进一帧。

所有 rank 的发送侧仍是普通连续 `MPI_FLOAT`：

```text
sendcount_r = M * c_r
sendbuf_r   = 局部 double 在通信前转换得到的连续 float [M,c_r]
```

root 的接收参数为：

```text
recvtype       = column_resized
recvcounts[r]  = c_r          // 完整帧列的数量，不是 M*c_r
displs[r]      = b_r          // extent 恰为 1 float
recvbuf        = 最终全局 float [M,F] 的首地址
```

于是只需一次所有 rank 都参与的 `MPI_Gatherv`。完整独立函数原型见同目录 `gather_prototype.cpp`。

### 2.2 为什么一般不能直接使用 `count = c_r`

`MPI_Gatherv` 只有**一个** root `recvtype`，不能像 `recvcounts` 那样给每个 rank 传一个不同 datatype。非整除时若在 root 分别构造

```text
vector(M,c_r,F) + extent(c_r floats)
```

并不能把这些不同类型放进同一次 `Gatherv`。

也不能简单取 `C=max(c_r)`：短 rank 没有 `M*C` 个有效元素，补齐后还会使 root 的不同 rank 接收区域重叠或越过行尾。

取 `C=1` 后类型对所有 rank 完全相同，而 `recvcounts[r]=c_r` 表达任意块长，因此非整除和空区间都不需要第二次通信或额外打包。

### 2.3 等长块的另一种合法参数化

仅当所有 `c_r=C>0`（典型情形是 `F % P == 0`，`C=F/P`）时，可用一个 tile 对应一个 rank：

```cpp
MPI_Type_vector(M, C, F, MPI_FLOAT, &tile);
MPI_Type_create_resized(tile, 0, (MPI_Aint)C * sizeof(float),
                        &tile_resized);
MPI_Type_commit(&tile_resized);
```

此时：

```text
recvcounts[r] = 1
displs[r]     = b_r / C = r   // 连续等长划分
sendcount_r   = M * C
```

**注意 displacement 的单位：**`Gatherv::displs` 以 resized receive type 的 extent 为单位，不以字节为单位，也不天然以单个 float 为单位。因此：

- 通用 `C=1` 方案：`displs[r]=b_r`；
- 等长 `C>1` 方案：若 extent 为 `C` floats，则 `displs[r]=b_r/C`，连续等长时即 `r`；不能直接填 `b_r`，否则地址会放大 `C` 倍。

若确需非等长的 `C>1` rank-specific tile，只能改成其他通信结构（例如逐 rank 的接收），不再是单次 `Gatherv`。推荐统一使用 `C=1`，无需保留两套路径。

## 3. 输出顺序证明

### 3.1 通用 `C=1` 方案

发送数组的元素顺序为：

```text
L_r[0:c_r]                 // m=0 的局部连续帧
L_r[c_r:2*c_r]             // m=1
...
L_r[(M-1)*c_r:M*c_r]       // m=M-1
```

第 `j` 个 receive-type 实例消费发送流中的第 `j` 组 `M` 个 float，其第 `m` 个元素就是：

```text
L_r[m*c_r+j]
```

root 的 `column_resized` extent 为 1 float，且第 `j` 个实例的第 `m` 个 block 位于：

```text
recvbuf + displs[r] + j + m*F
        = recvbuf + b_r + j + m*F
```

所以通信后恰有：

```text
G[m*F+b_r+j] = L_r[m*c_r+j]
             = value(m, b_r+j)
```

对任意 `0<=m<M`、`0<=j<c_r` 成立。

各区间 `[b_r,e_r)` 两两不交且并集为 `[0,F)`，故所有全局元素恰好被写一次；不存在依赖消息到达先后的覆盖。最终线性数组严格为 `[M,F]` 行主序，root 连续写出即与 baseline 文件布局一致。

### 3.2 等长 `C>1` 方案

一个 tile 实例的第 `(m,j)` 个元素位于：

```text
recvbuf + C*displs[r] + m*F + j
        = recvbuf + b_r + m*F + j
```

发送流按局部 `[M,C]` 行主序排列，与 tile 的 block 顺序一一对应，因此得到同一个全局下标公式。各等长区间同样不交且完整覆盖。

## 4. root、非 root 与空区间处理

### root

1. 分配最终 `std::vector<float> global(M*F)`。
2. 构造并 commit `column_resized`；准备长度为 `P` 的 `recvcounts`、`displs`。
3. root 自己也持有一个独立的局部 `[M,c_root]` float send buffer，调用同一次 `Gatherv`。
4. 完成后释放 datatype，直接把 `global` 的 `4*M*F` 字节写入输出文件。

不建议此路径使用 `MPI_IN_PLACE`：局部 `[M,c_root]` 与全局 `[M,F]` 的布局不同，root 的连续局部发送缓冲和非连续全局接收区域应当分离。

### 非 root

- 只分配局部 double 和局部 float send buffer，不分配全局数组。
- `recvbuf/recvcounts/displs` 可传 `nullptr`；被忽略的 `recvtype` 可传 `MPI_FLOAT`。
- 必须与 root 一样调用同一个 collective，不能因局部为空而提前跳过。

### 空区间 `c_r=0`

- `sendcount=0`，send buffer 可为空；
- root 设置 `recvcounts[r]=0`；
- `displs[r]=b_r` 仍保留该 rank 的逻辑位置，可以为 `F`，因为不接收任何元素；
- 不创建 `blocklength=0` 的特殊类型，所有 rank 继续使用 root 的同一个 `C=1` receive type；
- 原始有效输入保证 `F>=1`，所以 root 最终全局数组非空。

### 其他实现边界

- MPI 普通 count 接口要求每个 rank 的 `M*c_r` 以及 root 的 `c_r`、`b_r` 可由 `int` 表示；全局线性索引和分配大小使用 `size_t`。
- `MPI_Type_vector` 的 `M`、`F` 参数也是 `int`。超大问题需要 big-count MPI 接口或分段方案，不能静默截断。
- 集群按同构 IEEE-754 binary32 环境考虑；root 在目标小端平台直接写 `MPI_FLOAT` 对应的 float 数组。

## 5. 为什么在通信前转换 double → float

准确的执行位置是**每个发送 rank 转换自己的局部块**；root 对自身的局部块也这样做。root 无法在通信前替其他 rank 转换其尚未收到的数据。

### 5.1 精度依据

baseline 对每个最终 double 元素只做一次：

```cpp
float x = static_cast<float>(double_x);
```

本方案也只在发送前做同一次逐元素转换。随后 `MPI_FLOAT` 仅传输该 binary32 值，不做累加、重排运算或二次舍入；root 直接写收到的 float。

因此，在各 rank 的局部 double 计算结果与串行对应元素一致的前提下：

- “发送前转换后汇集”与“先汇集 double、root 最后转换”对每个元素执行的是相同的单次 double→float 舍入；
- 汇集本身不引入新的数值误差；
- 最终 float 值和元素顺序均与 baseline 的最后输出阶段一致，而不是仅依赖 `allclose` 容差推断正确性。

分布式 FFT/划帧是否使 double 中间结果一致属于计算部分的独立正确性问题，不由 gather 保证。

### 5.2 通信量依据

每个元素 double 为 8 字节，float32 为 4 字节：

| 项目 | 汇集 double | 通信前转 float |
|---|---:|---:|
| 每 rank 发送元素数 | `M*c_r` | `M*c_r` |
| 每 rank 发送载荷 | `8*M*c_r` | `4*M*c_r` |
| collective 的逻辑总载荷（含 root 自身贡献） | `8*M*F` | `4*M*F` |
| root 最终文件 | `4*M*F` | `4*M*F` |

所以跨 rank 的有效通信载荷减少 50%；root 自身贡献通常不经过真实网络。这里只给出字节数关系，不声称通信时间或端到端时间必然减半。

Derived datatype 的 stride/hole 只描述 root 内存布局，不增加发送数据：发送签名仍恰好是每列 `M` 个 float，网络不会因全局行 stride `F` 而传输空洞。

## 6. 内存成本

设某 rank 块长为 `c_r`：

### 推荐方案

- 每 rank 原局部 double：`8*M*c_r`；
- 转换期间新增局部 float send buffer：`4*M*c_r`；
- root 额外最终接收数组：`4*M*F`；
- root counts/displacements：`O(P)`；
- datatype：`O(1)` 描述开销，不需要 root 的全局打包临时数组。

因此转换阶段：

```text
非 root 主要输出相关内存 = 8*M*c_r + 4*M*c_r
root 主要输出相关内存   = 8*M*c_root + 4*M*c_root + 4*M*F + O(P)
```

转换完成后若后续计算不再需要局部 double，可在 collective 前释放它，以降低汇集期间的常驻内存；但转换瞬间两块仍需同时存在。

### 与朴素 double 汇集比较

若 root 先收完整 double `[M,F]`，再生成 float 输出，root 可能同时持有：

```text
8*M*F 的 double 汇集数组 + 4*M*F 的 float 写出数组
```

推荐方案不创建 root 全局 double 汇集数组，直接得到最终 float 输出。

### 与额外连续打包比较

rank-major 的临时 float 汇集若还需 root 重排成 `[M,F]`，会额外占约 `4*M*F`，并增加一次完整数组读写。Derived datatype 方案省去该显式临时数组；MPI 实现内部如何缓冲属于实现细节，不作零内部开销假设。

## 7. 其他方案对比

### 7.1 逐 mel 行 `Gatherv`

对每个 `m`，各 rank 的一行局部帧天然连续：

```text
sendbuf = local + m*c_r
recvbuf = global + m*F
recvcounts[r] = c_r
displs[r] = b_r
datatype = MPI_FLOAT
```

调用 `M` 次 `Gatherv` 即可直接形成全局输出。优点是简单、无 derived type、无重排临时数组；缺点是 collective 次数为 `M`，同步和延迟暴露更多。通信字节数与推荐 float 方案相同，不预设其实际性能优劣。

### 7.2 打包成连续接收缓冲后重排

一次普通 float `Gatherv` 若按 rank 拼接，会得到类似 `[rank,m,local_frame]` 的布局，而不是 `[m,global_frame]`。即使各 rank 先转置为 frame-major，rank 拼接得到的也是 `[global_frame,m]`，仍需转置。

要满足现有输出语义，需要：

1. root 额外连续临时数组；
2. 按 `G[m*F+b_r+j] = tmp[...]` 完整重排；
3. 再写最终数组。

这是概念简单的单次通信备选，但有明确额外内存和内存流量，不能省略最后重排。

### 7.3 MPI-IO 共享单文件

可为每 rank 建立文件视图：文件表示 `[M,F]`，rank `r` 的可见区域用 `vector(M,c_r,F,MPI_FLOAT)` 描述，文件 displacement 对应 `b_r` 个 float。局部 `[M,c_r]` 连续写出，文件视图把各帧块放到正确全局行位置。

优点：

- 不必在 root 保存 `4*M*F` 全局接收数组；
- 各 rank 直接写最终逻辑位置，无 root 重排；
- 文件视图与本文下标公式相同，可保证逻辑顺序。

注意：

- 必须使用无头二进制视图并保证最终文件长度恰为 `4*M*F`，覆盖运行时避免旧文件尾部残留；
- 目标小端同构环境使用匹配 native float32 的表示，不能随意换成不同字节序的数据表示；
- 需要按 MPI-IO 语义选择独立或 collective write、检查所有 rank 错误并正确关闭文件；
- 共享文件系统、 strips/锁和 collective buffering 开销必须另行测量，本方案不预设优于 root 写出。

### 7.4 每 rank 独立文件

各 rank 写自己的 `[M,c_r]` 文件最容易实现且无需汇集，但 official CLI 要求一个无头 `[M,F]` 文件。后处理必须按 mel 行拼接各 rank 的帧区间，不能只按 rank 文件顺序直接 `cat`。

因此它适合诊断或离线流水线，不是无需后处理的现有 CLI 替代品；额外合并 I/O 和存储也应计入端到端成本。

## 8. 结论

通用的首选原型是：

```text
发送：各 rank 连续 float [M,c_r]，sendcount=M*c_r
接收：root vector(M,1,F,MPI_FLOAT)，resized extent=1 float
参数：recvcounts[r]=c_r，displs[r]=b_r
结果：一次 Gatherv 直接写满 root 的最终 float [M,F]
```

它在非整除、任意空区间下都保持同一 receive type，不需要全局临时打包或逐行 `M` 次 collective，并由逐元素下标等式证明输出顺序。等长块可用 `count=C` 的一 rank 一 tile 形式，但此时 displacement 必须按 `C`-float extent 缩放，不能照搬 `displs=b_r`。
