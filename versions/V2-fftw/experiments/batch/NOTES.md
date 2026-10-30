# FFTW `plan_many` R2C 独立原型（未编译、未运行）

## 状态与边界

- 只读参考了指定的 A2 `src/melspec.cpp`、`include/fftw_support.h`、`versions/V2-fftw/PROFILE.md`、`README.md` 和 `scripts/fftw_check.py`。
- 本目录是独立实验：候选只替换 `melspec.cpp` 的实现，继续与受保护的 `src/main.cpp` 链接。
- **本轮没有执行 shell、编译、CPU 测试、Git 或集群操作；没有任何 timings、speedup 或实测 RSS。** 默认 `B=8` 只是可编译的功能默认值，不是性能推荐值。
- 支持编译宏 `MELSPEC_FFTW_BATCH=1/4/8/16/32`；其他值在预处理期报错。

## 文件

- `replacement.cpp`：实现 `melspec.h` 的全部公开函数；热点为 double `fftw_plan_many_dft_r2c`。
- `check_boundaries.py`：供主 Agent 后续串行执行的确定性边界/非法输入检查，不是基准。
- `NOTES.md`：设计、数学、内存、构建和验收说明。

## 实现计划

1. 保留公开 `hz_to_mel`、`mel_to_hz`、`compute_rfft`、`build_mel_filterbank` 和 `compute_melspectrogram` 符号。单帧公开 RFFT 仍用原生 R2C；批处理只用于 `compute_melspectrogram`。
2. `B` 帧组成一批。每个 worker 有自己的：
   - `B*n_fft` 个 double 输入；
   - `B*n_bins` 个 `fftw_complex` 输出；
   - `B*n_bins` 个 double 功率 scratch。
3. 所有 scratch 在规划与启动线程前分配。输入/输出都由 `fftw_malloc` 分配，各 worker 尺寸、对齐、in/out-of-place 属性和布局一致。
4. 主线程串行创建一个共享 plan，worker 只调用 new-array `fftw_execute_dft_r2c`。析构也在 join 后的串行路径。没有 `fftw_init_threads`、`fftw_plan_with_nthreads`、OpenMP 或 FFTW threads 后端。
5. 布局无隐藏 padding：`rank=1`、`howmany=B`、`istride=1`、`idist=n_fft`、`ostride=1`、`odist=n_bins`，两个 embed 指针均为 `nullptr`。
6. 沿用 V1/A2 的 worker 区间和线程数策略，最多 16 个外层 worker。每个 worker 都可能有尾批；无效行显式置零，共享 plan 仍执行完整 `B` 行，但功率/Mel/输出只处理 `valid` 行。
7. 每批 FFT 后立即计算模平方并逐帧做稀疏 Mel；只存在 `W*B` 帧功率，不物化 `n_frames*n_bins` 全量功率谱。
8. 输出仍为 double 累加、行主序 `[n_mels,n_frames]`，索引 `m*n_frames+t`；后续 float32 转换仍由未修改的 main 完成。
9. 周期 Hann 为 `0.5*(1-cos(2*pi*i/n_fft))`。`n_fft=1` 时唯一窗系数为零，直接走原功率/Mel 顺序，不创建 FFT plan。过短输入、非正 hop、非法 FFT 大小等在除法/规划前拒绝。

## 数学与帧覆盖论证

对批内行 `b`，加窗输入为

\[
I_{b,i}=y[(t_0+b)h+i]\,w_i,\qquad 0\le i<N.
\]

R2C plan 对每一行独立计算同一个未归一化前向 DFT：

\[
X_{b,k}=\sum_{i=0}^{N-1} I_{b,i}e^{-2\pi jki/N},
\qquad 0\le k\le N/2.
\]

`idist=N` 与 `odist=N/2+1` 使不同行完全不重叠；尾批补零行不会与有效行发生数据耦合。候选继续计算
`re*re + im*im`，再按递增 `k` 做 `acc += weight*power`，因此保持 double Mel 累加顺序。HTK 换算、三角滤波器和 Slaney `2/(f_r-f_l)` 面积归一化与 A2 相同。

worker `i` 的半开区间为

\[
[\lfloor Fi/W\rfloor,\ \lfloor F(i+1)/W\rfloor),
\]

相邻区间端点相同，所以并集恰好为 `[0,F)` 且无重叠。区间内 `t0 += B`，`valid=min(B,end-t0)`；只对 `t0 <= t < t0+valid` 写出，故尾批不会重复、遗漏或越界输出。批处理改变 FFTW 计划/内核，**数学等价不保证与 A2  bitwise 相同**，仍需规定容差验证。

## Scratch 内存公式与数字

记 `N=n_fft`、`K=N/2+1`、`B=batch`、`W=worker` 数。忽略 allocator 元数据、vector/thread 对象和 FFTW plan 内部存储：

\[
S_{worker}=8BN+16BK+8BK=8B(N+3K)\ \text{bytes}.
\]

`N>1` 且为 2 的幂时，`K=N/2+1`，所以

\[
S_{worker}=(20BN+24B)\ \text{bytes},\qquad
S_{all-workers}=W(20BN+24B)\ \text{bytes}.
\]

`N=1,K=1` 时直接使用通式，每 worker 为 `32B` bytes（虽然 FFT 路径被绕过，原型仍分配该 scratch）。

计算期其余主要常驻存储为：

\[
S_{persistent}\approx 8MF + 8MK + 8N + 8M
\]

分别对应输出、Mel 权重、窗口以及 first/last 两个 int 数组。因此常驻近似总量为
`S_persistent + S_all-workers + FFTW plan 内部开销`；构建滤波器时的临时频率数组、输入 PCM、main 的写出 float 缓冲和运行时开销另计。

以下为公式数字，**不是 RSS 测量**。单位 MiB=`2^20` bytes，取 `N=8192,K=4097,W=16`：

| B | 每 worker scratch MiB | 16 worker 合计 MiB |
|---:|---:|---:|
| 1 | 0.156273 | 2.500366 |
| 4 | 0.625092 | 10.001465 |
| 8 | 1.250183 | 20.002930 |
| 16 | 2.500366 | 40.005859 |
| 32 | 5.000732 | 80.011719 |

另一个边界数字：`N=16384,W=16,B=32` 时，每 worker `10.000732 MiB`，合计 `160.011719 MiB`。相比 A2 每帧 scratch，理想数组量随 `B` 线性增加；大 `B` 的 cache/page-fault 代价必须实测，不能由公式推断加速。

## 后续串行构建命令（本轮未执行）

以下命令使用系统 FFTW 头文件和 `libfftw3`，所有新产物都留在本目录。若目标机使用非默认静态库，主 Agent 需换成该节点已批准的包含/库路径；**不要添加 `fftw3_threads` 或 `fftw3_omp`**。

```bash
cd /home/palm/projects/heptagon/melspec-v2-fftw
mkdir -p tmp/fftw-lab/batch/obj tmp/fftw-lab/batch/bin

g++ -std=c++17 -O3 -march=native -pthread \
  -DMELSPEC_FFTW_MEASURE=0 -Iinclude \
  -c src/main.cpp -o tmp/fftw-lab/batch/obj/main.o

for B in 1 4 8 16 32; do
  g++ -std=c++17 -O3 -march=native -pthread \
    -DMELSPEC_FFTW_MEASURE=0 -DMELSPEC_FFTW_BATCH=${B} -Iinclude \
    -c tmp/fftw-lab/batch/replacement.cpp \
    -o tmp/fftw-lab/batch/obj/replacement-b${B}.o
  g++ -pthread \
    tmp/fftw-lab/batch/obj/main.o \
    tmp/fftw-lab/batch/obj/replacement-b${B}.o \
    -lfftw3 -o tmp/fftw-lab/batch/bin/melspec-b${B}
done
```

可单独构建当前 A2 对照，但它绝不能与 replacement 同时链接进同一候选：

```bash
g++ -std=c++17 -O3 -march=native -pthread \
  -DMELSPEC_FFTW_MEASURE=0 -DMELSPEC_FFTW_R2C=1 -Iinclude \
  -c src/melspec.cpp -o tmp/fftw-lab/batch/obj/a2-reference.o
g++ -pthread tmp/fftw-lab/batch/obj/main.o \
  tmp/fftw-lab/batch/obj/a2-reference.o -lfftw3 \
  -o tmp/fftw-lab/batch/bin/a2-reference
```

默认采用冷 `FFTW_ESTIMATE`。若另测 `MEASURE`，须单独输出目录/文件名并计入规划成本；不能沿用上述二进制或宣称冷启动收益。诊断 profile 可额外加 `-DMELSPEC_PROFILE=1`，其插桩结果不能当正式成绩。

## 正确性矩阵（全部待执行）

`check_boundaries.py` 对每个 `B` 生成确定性的 `n_fft=8, hop=3, n_mels=16` 随机 PCM，逐元素比较行主序 float 输出，容差与项目一致 (`rtol=1e-3, atol=1e-4`)。

| 维度 | 帧数/参数 | 目的 |
|---|---|---|
| 单 worker 批边界 | `F=1,B-1,B,B+1,2B-1,2B,2B+1`（去除非正值） | 空尾批之外的不足、恰好满批、多一帧和多批尾批 |
| worker 数阈值 | 对 `W=2,4,8,16`，`F=256W-1,256W,256W+1` | V1 `F/256` 策略的 worker 数变化及区间接缝 |
| worker 内尾批 | `F=256W+B` | 多 worker 下非均匀末端与各 worker 自己的尾批 |
| 非单位 hop | 全部上述用例 `N=8,hop=3` | 检查输入帧地址没有混用 batch/frame 单位 |
| 小 FFT 特例 | `n_fft=1`，`F=B,B+1,1024+B` | 周期窗唯一系数为零、无 plan 分支与批边界 |
| 非法/过短 | `N<n_fft`；`hop=0,-1`；`n_fft=0,3`；`n_mels=0`；`sr=0` | 非零退出、不越界、不执行非法规划 |

多 worker 覆盖要求目标机 `hardware_concurrency()` 至少为对应 `W`；否则这些帧数仍正确，但不能声称覆盖了 16-worker 路径。建议后续在 profile 输出中核对实际 worker 行数，而不只依据用例名。

串行验收命令示例（假定已有脚本默认的 V1 `tmp/bin/v1`）：

```bash
cd /home/palm/projects/heptagon/melspec-v2-fftw
source tmp/session-env.sh
$PY tmp/fftw-lab/batch/check_boundaries.py \
  --reference tmp/bin/v1 \
  --programs tmp/fftw-lab/batch/bin/melspec-b1 \
             tmp/fftw-lab/batch/bin/melspec-b4 \
             tmp/fftw-lab/batch/bin/melspec-b8 \
             tmp/fftw-lab/batch/bin/melspec-b16 \
             tmp/fftw-lab/batch/bin/melspec-b32

$PY scripts/fftw_check.py --reference tmp/bin/v1 \
  tmp/fftw-lab/batch/bin/melspec-b1 \
  tmp/fftw-lab/batch/bin/melspec-b4 \
  tmp/fftw-lab/batch/bin/melspec-b8 \
  tmp/fftw-lab/batch/bin/melspec-b16 \
  tmp/fftw-lab/batch/bin/melspec-b32

$PY validate.py tmp/fftw-lab/batch/bin/melspec-b8
```

还应对五个 B 分别跑完整四组官方验证，而非只验 B8。公开 helper 建议另加小单元测试：`compute_rfft` 对 `N=1,2,8` 与朴素 double DFT 比较，HTK 正逆换算及滤波器权重与 A2 比较；CLI 不一定覆盖这些独立入口。

## 集成风险与待验收/测量点

1. **链接冲突**：候选只能链接 replacement，不能再链接正式 `src/melspec.cpp`。
2. **对齐契约**：new-array execute 依赖相同 alignment/layout。不得把 FFTW 输入/输出随意改成普通 vector 或改变距离后继续复用该 plan。
3. **线程安全**：共享 plan 只执行 new-array execute；规划和销毁保持串行。只链接普通 `libfftw3`，外层 `std::thread` 是唯一并行层。
4. **数值差异**：plan-many 可选不同 codelet/分解，需全部容差测试，不能假设 bitwise 一致。
5. **scratch 增长**：最大配置在 `N=16384,W=16` 时公式已达约 160 MiB，可能改变 cache、TLB、page fault 和峰值 RSS；输出与 main I/O 固定成本仍不可忽略。
6. **规划成本**：`B` 改变 plan，ESTIMATE 时间也不能假定为 A2 的约 2 ms；MEASURE 更需要单独冷启动测量。
7. **尾批开销**：每 worker 最后一批可能计算无效零帧；这是为有界 scratch 和共享完整批 plan 做的明确取舍，收益未知。
8. **硬件覆盖**：线程创建失败会回退到主线程，正确性应保持，但性能解读必须记录实际 worker 数。

待主 Agent 串行验收后才可测量：五档 B 的构建/正确性、plan 与 setup 时间、完整 compute/官方计时、峰值 RSS，以及 pack/FFT/power/Mel 分项。应在同一目标机、相同 FFTW 库和冷进程下与 A2 对照，不能用单帧微基准乘以批大小或 16 worker 推算加速。**当前没有证据断言任一批大小会加速。**

---

## Layout 扩展：`replacement_layout.cpp`（新增，未编译、未运行）

原 `replacement.cpp` 保持不变。主 Agent 已反馈原 unpadded 版本的首轮结果；本节的新布局原型尚未由本 worker 执行，**没有新布局的 PASS、timing、speedup 或 RSS 结果**。

新文件只允许 `MELSPEC_FFTW_BATCH=4/8/16`，并增加显式 `MELSPEC_BATCH_LAYOUT`：

- `0`（默认）：padded-row。
- `1`：interleaved。
- 其他值在预处理期报错。

两种布局仍实现相同公开接口、double HTK/Slaney/周期 Hann、最多 16 个外层 worker、同一 V1 帧区间、一个串行创建并由所有 worker 共享的 new-array plan。规划和析构均持 planner mutex；不使用 FFTW 内部线程。每批尾行仍完整零填充，功率/Mel/输出仅访问有效行。

### 1. Padded-row（layout 0）

令

\[
K_{pad}=4\left\lceil K/4\right\rceil=(K+3)\,\&\,\sim 3.
\]

地址与 plan 参数为：

- 输入 `input[b*N+i]`：`istride=1,idist=N`，共 `B*N` doubles。
- FFT 输出 `output[b*Kpad+k]`：`ostride=1,odist=Kpad`，共 `B*Kpad` complexes。
- 功率 `power[b*K+k]`：不补 padding，共 `B*K` doubles。

`Kpad` 是 4 个 complex 的倍数，所以相邻 FFT 输出行首相差 `4*16=64` bytes；所有行保持 `fftw_malloc` 基址的对齐类别。严格的绝对对齐上界仍由 FFTW allocator 提供，这里不额外伪造 aligned pointer。padding 区域从不参与功率或 Mel。

每 worker scratch：

\[
S_{row}=8BN+16BK_{pad}+8BK
      =8B(N+K+2K_{pad})\ \text{bytes}.
\]

总 scratch 为 `W*S_row`。相对同 B 的 interleaved，额外量为
`16*B*(Kpad-K)` bytes/worker。例：`N=8192,K=4097,Kpad=4100,W=16,B=16` 时，padded-row 每 worker 约 `2.501099 MiB`、合计约 `40.017578 MiB`；interleaved 分别为约 `2.500366` 和 `40.005859 MiB`。这些仍是公式值，不是实测。

### 2. Interleaved（layout 1）

地址与 plan 参数为：

- 输入 `input[i*B+b]`：`istride=B,idist=1`，共 `B*N` doubles。
- FFT 输出 `output[k*B+b]`：`ostride=B,odist=1`，共 `B*K` complexes。
- 功率 `power[k*B+b]`：与频谱交织布局一致，共 `B*K` doubles。

pack 在每个样本 `i` 上连续写 `b`；模平方在每个 bin `k` 上连续读写 `b`，可让编译器跨独立帧 SIMD，而不需要跨 lane 做 Mel 归约。Mel 仍逐帧、逐滤波器、按 `k` 严格递增执行一个 double `acc`，没有改成跨 bin 向量归约或改变加法结合顺序。代价是 Mel 的 power 读取步长为 `B`。

每 worker scratch：

\[
S_{interleaved}=8BN+16BK+8BK=8B(N+3K)\ \text{bytes},
\]

总 scratch 为 `W*S_interleaved`；与原 unpadded 数组公式相同。

### 3. 尾批、循环溢出与数学顺序

padded-row 对无效 `b` 的整行 `N` 个输入置零；interleaved 在每个 `i` 上对 `b=valid..B-1` 置零。二者都执行完整 B 行 plan，但功率只写有效帧，padding 和无效功率从不读取。

新文件的批循环使用 `t0 += valid`，而不是原文件的 `t0 += B`。由于 `valid>=1` 且最后一次恰到达 `end`，可修复 `t0+B` 在接近 `INT_MAX` 时的有符号溢出风险。按要求，原文件未回改。

两种布局都没有改变以下逐帧数学操作：

1. `y[start+i]*window[i]`；
2. R2C 后 `re*re+im*im`；
3. 每个 `(frame,mel)` 从 `first[m]` 到 `last[m]` 按 k 递增 double 累加；
4. 写出 `output[m*n_frames+t]`。

不同 plan 布局仍不承诺与 A2 或彼此 bitwise 一致，必须做容差验收。

### 4. 可能改善或变差的原因（仅假设）

**padded-row 可能改善：** 每行频谱连续，行首保持 64-byte 距离，可能减少多行访问时的对齐/缓存集合冲突，并保持类似原实现的逐帧功率和 Mel 连续读取。

**padded-row 可能变差：** FFTW 面对不同 `odist` 可能选择不同 codelet；额外 padding 增加 scratch 和地址跨度，且 plan-many 本身的首轮回退未必能由对齐解决。

**interleaved 可能改善：** 加窗拷贝和模平方均在 b 维连续，可跨帧 SIMD；功率布局无需转置即可供这种访问方式使用。

**interleaved 可能变差：** 对单条 DFT 而言输入/输出 stride 都为 B，可能降低 FFTW 内核效率或改变计划选择；Mel 中 stride-B 的 power 读取也可能抵消 pack/power 的收益。以上都必须通过目标机全流程测量判断，不能从布局名称推断加速。

### 5. 精确构建宏与命令（待主 Agent 串行执行）

```bash
cd /home/palm/projects/heptagon/melspec-v2-fftw
mkdir -p tmp/fftw-lab/batch/obj tmp/fftw-lab/batch/bin

g++ -std=c++17 -O3 -march=native -pthread \
  -DMELSPEC_FFTW_MEASURE=0 -Iinclude \
  -c src/main.cpp -o tmp/fftw-lab/batch/obj/main-layout.o

for B in 4 8 16; do
  g++ -std=c++17 -O3 -march=native -pthread \
    -DMELSPEC_FFTW_MEASURE=0 -DMELSPEC_FFTW_BATCH=${B} \
    -DMELSPEC_BATCH_LAYOUT=0 -Iinclude \
    -c tmp/fftw-lab/batch/replacement_layout.cpp \
    -o tmp/fftw-lab/batch/obj/layout-row-b${B}.o
  g++ -pthread tmp/fftw-lab/batch/obj/main-layout.o \
    tmp/fftw-lab/batch/obj/layout-row-b${B}.o -lfftw3 \
    -o tmp/fftw-lab/batch/bin/melspec-row-b${B}

  g++ -std=c++17 -O3 -march=native -pthread \
    -DMELSPEC_FFTW_MEASURE=0 -DMELSPEC_FFTW_BATCH=${B} \
    -DMELSPEC_BATCH_LAYOUT=1 -Iinclude \
    -c tmp/fftw-lab/batch/replacement_layout.cpp \
    -o tmp/fftw-lab/batch/obj/layout-inter-b${B}.o
  g++ -pthread tmp/fftw-lab/batch/obj/main-layout.o \
    tmp/fftw-lab/batch/obj/layout-inter-b${B}.o -lfftw3 \
    -o tmp/fftw-lab/batch/bin/melspec-inter-b${B}
done
```

候选不得同时链接正式 `src/melspec.cpp` 或原 `replacement.cpp`。仅链接普通 `-lfftw3`，不得添加 FFTW threads/OpenMP 后端。默认是 `FFTW_ESTIMATE`；任何 MEASURE/profile 变体需另名构建并单独解释。

### 6. 新布局正确性矩阵（全部待执行）

| 维度 | 用例 | 检查点 |
|---|---|---|
| B/L 全组合 | `B=4,8,16` × `layout=0,1` | 六个独立二进制的编译、链接与完整输出 |
| 批尾 | 每个 B 的 `F=B-1,B,B+1,2B-1,2B,2B+1` | 无效行零填充、不读 padding、不重复/遗漏帧 |
| worker 边界 | `W=2,4,8,16`，`F=256W-1,256W,256W+1,256W+B` | V1 分区接缝、每 worker 自己的尾批 |
| R2C padding | `N=2(K=2,Kpad=4),4(3,4),8(5,8),256(129,132),2048(1025,1028)` | 不同 padding 量和 odist；interleaved 同参数对照 |
| 输入寻址 | `n_fft=8,hop=3`，随机、零、impulse、Nyquist | interleaved 的 `i*B+b` 与 padded 的 `b*N+i` |
| 小 FFT | `n_fft=1`，跨 B 和 worker 边界 | 零周期窗分支、两种 power 布局在 K=1 时一致 |
| 输出语义 | 全向量 shape/stride 检查 | row-major `[n_mels,n_frames]`，double Mel 后由 main 转 float |
| 安全 | 过短输入、hop 0/-1、非法 N/M/sr | 非零退出，无非法 plan 或越界 |
| 宏约束 | B=1/2/32、layout=-1/2 的预期编译失败 | 防止误用未实现布局 |

现有 `check_boundaries.py` 的帧数集合是上述通用超集，可传入六个新程序；`fftw_check.py` 继续覆盖随机/极值信号和短输入。多 worker 结论仍需目标机 hardware concurrency 与 profile worker 行数佐证。

通过正确性后才测量六个组合的 plan/setup、pack/FFT/power/Mel、完整官方计时和峰值 RSS，并与同节点 A2 及原 unpadded 候选比较。**当前新增布局未测，不标记 PASS，不宣称加速。**
