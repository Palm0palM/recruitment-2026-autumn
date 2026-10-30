# FFTW ISA / codelet 审查与受控实验计划

状态：**仅静态审查，未运行 shell、编译、benchmark、Git 或远程任务；本文所有新实验均为未测。**
只允许主 Agent 在统一调度 CPU 后审查、修改并执行本目录中的草案。

## 1. 证据边界与已有实际 plan

审查输入仅有 `PROFILE.md`、作业 64 原始输出、现有 FFTW 构建脚本、plan probe 和
`fftw_support.h`。根目录所述的 `raw/cluster/fftw-64.out` 不存在；实际读到的是
`versions/V2-fftw/raw/cluster/fftw-64.out`。

作业 64 的环境是 Xeon Gold 6338、FFTW 3.3.11 静态 double 库，库构建参数为
`gcc -O3 -march=native`，`fftw_version` 列出 sse2/avx/avx2/avx2_128/avx512。
这些字段以及 `HAVE_AVX*` 只说明**编译进库的 ISA 变体**，不说明某次 transform 选择了哪些
codelet。实际选择必须看同一次 plan 的 `fftw_sprint_plan`；下列名称均来自原始输出，不是反汇编猜测。

| 路线 | n_fft | planner | 实际分解与 SIMD codelet（摘要） | copy+execute |
|---|---:|---|---|---:|
| half-C2C | 8192 | ESTIMATE | CT 32；`t3fv_32_avx512`；buffered 128-x32；`n1fv_128_avx512` | 21.0706178 us |
| half-C2C | 8192 | MEASURE | CT 64；`t2fv_64_avx512`；buffered 64-x64；`n2fv_64_avx` | 15.3858956 us |
| R2C | 8192 | ESTIMATE | RDFT2 CT 2 + AVX2 前端；CT 32；`t3fv_32_avx512` + `n1fv_128_avx512` | 22.3327296 us |
| R2C | 8192 | MEASURE | 同一 AVX2 RDFT2 前端；CT 64；`t2fv_64_avx512` + `n2fv_64_avx` | 16.2234752 us |
| half-C2C | 16384 | ESTIMATE | CT 32；buffered 256-x32；内层 CT 16；三个 AVX-512 codelet | 46.5230710 us |
| half-C2C | 16384 | MEASURE | 顶层 buffered 8192；CT 8 / 内层 CT 64；AVX-512 twiddle + `n1fv_16_avx2`；rank0 memcpy | 32.6700240 us |
| R2C | 16384 | ESTIMATE | RDFT2 CT 4 + AVX2 前端；vrank x2；CT 32；两个 AVX-512 codelet | 46.1264326 us |
| R2C | 16384 | MEASURE | RDFT2 CT 2 + AVX2 前端；CT 8 / 内层 CT 64；AVX-512 twiddle + `n1fv_16_avx` | 34.4146820 us |

关键解读：

* ESTIMATE 已经选中 AVX-512 codelet，不能把它描述成“AVX2 基线”。MEASURE 也仍混用
  AVX/AVX2/AVX-512，并未“全换成 AVX-512”。R2C 的 real/hc2c 前端在这四组中始终是 AVX2。
* planner 同时改变分解、codelet 与 copy/buffer 结构。因此表中的 MEASURE/ESTIMATE 差异不是
  一个纯 ISA 开关实验，更不能归因为某一个后缀。
* half-C2C 的两组 ESTIMATE 都有 `dft-buffered`；MEASURE 8192 也有 buffered 64-x64，
  MEASURE 16384 有顶层 buffered 8192 和 `rdft-rank0-memcpy/16384`。R2C 的四份文本没有
  `dft-buffered` 节点。buffered 节点表示 FFTW 内部的缓冲/重排路径；plan 文本本身不能给出其耗时。
* probe 的计时还包括每次用 `std::vector` 样本刷新 FFTW 输入的应用侧拷贝。因此
  “copy+execute”既可能含应用拷贝又含 FFTW 内部 buffered-copy，不能当作纯 codelet 时间。
* 这是**单线程、5000 次输入刷新 + FFT execute** 的微基准，不是 16-worker 全程序并发结果，
  不含 Mel、功率、读写和分配。不能将每帧差额乘 16 或乘帧数推导端到端成绩。
* 同库 MEASURE 相对 ESTIMATE 的单线程 copy+execute 时间降幅约为 26.96%、29.78%、27.35%、
  25.40%（按上表路线顺序），但其冷规划耗时为 0.453–0.823 s；ESTIMATE 为约
  0.000155–0.001925 s。这里是 planner/wisdom 权衡，不是 AVX2 对 AVX-512 的测量。
* 目前没有 AVX2-only 构建的数字，也没有任何“AVX-512 相对 AVX2 加速比”。
  本地 WSL 只有 AVX2/FMA、无 AVX-512/PMU，不能补测或推断集群 AVX-512、IPC、cache/TLB、
  降频等数据。

## 2. 当前封装、plan 生命周期和对齐

`fftw_support.h` 使用 double `fftw_plan_dft_r2c_1d` 或半长原地 C2C，默认 ESTIMATE，
编译宏可切到 MEASURE。plan/destroy 由互斥锁串行化，worker 通过 new-array execute 共享 plan，
每个 worker 使用自己的 `fftw_malloc` buffer；FFTW 执行线程后端被禁用，外层线程是唯一并行层。

作业 64 所有 planning buffer 的 `fftw_alignment_of` 都是 0。这里的 0 是 FFTW 的 alignment
class，表示该指针对 FFTW 而言对齐，**不是“地址模 64 等于 0”的直接报告**。它也只证明 probe
当时的 planning 输入/输出，不自动证明应用每个 worker、每次 new-array execute 或未来 batch 每行
都具有相同布局。当前 `Plan::apply` 没有运行时 alignment 断言。

`FFTW_UNALIGNED` 不应作为静默修复加入：它会改变 planner 约束和可能选择的 codelet，必须成为
单独实验。若 new-array 的地址/stride 与 plan 假设不一致，仅“执行没有崩溃”也不是布局正确性的证明。

## 3. AVX2-only 对 AVX-512：必须分清的两种 claim

1. **只禁用 FFTW AVX-512 codelet**：configure 不启用 `--enable-avx512`，但 native 编译仍可能让
   FFTW 的通用 C 代码或应用代码出现其他 AVX-512 指令。这不能称为“全进程 AVX2-only”。
2. **FFTW 与应用都限制为 AVX2**：除不启用 AVX-512 codelet 外，库和 probe/app 都不能用
   `-march=native`。草案对库使用共同 `-O3 -march=x86-64-v3`，AVX2 变体再加
   `-mno-avx512f`；probe/app 始终用 AVX2 基线编译。这样才是较严格的 AVX2-only 设计。

AVX-512 库需要编译器接受 `-mavx512f`，所以其通用 C 代码也可能与 AVX2 库不同。
即使应用编译 flags 固定，这一 residual confound 仍存在；结果只能先表述为“两个完整 FFTW
构建的差异”。只有在 plan 相同、执行路径相同且反汇编审查支持时，才可进一步缩小归因，
仍不能宣称所有差异都来自 codelet。禁止用“关闭 codelet 但保留 native 其他 AVX-512 指令”
来冒充严格的 AVX2-only。

反向也成立：库里编译了 AVX-512 不代表 plan 会选它，选中 AVX-512 也不保证更快。分解、内部
copy、cache、内存布局、频率和并发都可能抵消局部向量宽度收益。

## 4. 受控实验矩阵（全部由主 Agent 调度）

### 4.1 固定条件和独立构建

* 在同一目标计算节点/同一 CPU 型号运行；先在已分配、允许的 CPU 上构建，不在登录节点构建。
* 固定 FFTW 3.3.11 源码 SHA256、同一 gcc 版本、`-O3`、静态库、无 FFTW threads/OpenMP/MPI。
* 两个 prefix 完全独立，例如 `prefix/avx2-double` 与 `prefix/avx512-double`；build 目录、
  probe binary、map、日志、wisdom 也分开。绝不覆盖或重建现有 `tmp/opt/fftw-3.3.11`。
* SIMD configure 除 `--enable-avx512` 外保持一致。AVX2 变体不得使用 native；AVX-512 变体只在
  实际暴露 avx512f 的目标机构建/运行。
* probe/app 自身用固定 `-O3 -march=x86-64-v3 -mno-avx512f`，以绝对路径链接待测 `.a`，保留
  link map/链接 trace，防止误链系统 FFTW 或另一 prefix。
* 主对照是这两个匹配构建之间的 A/B。现有 native AVX-512 库可另列第三组，但不能与匹配构建
  混称同一实验。

`build_fftw_variant.sh` 是独立 prefix 的构建草案；它不执行任何现有库或共享环境的写操作。
`isa_plan_probe.cpp` 打印版本、编译 flags、实际 plan、alignment class、地址模 16/32/64、
batch 每行布局，并可选择与原 probe 同类的单线程 copy+execute 计时。

### 4.2 冷 plan 与实际 codelet

对 `n_fft={8192,16384}`、`kind={r2c,half-c2c}`、`planner={estimate,measure}`、两个 ISA 构建，
使用独立进程，每组先 `forget_wisdom`，不加载 wisdom。记录：

* 完整 `fftw_sprint_plan`，而不只记录 `fftw_version`/configure 宏；
* plan 创建时间、import 时间（若有）、execute/copy+execute 时间分开；
* planning buffer 和第二套 new-array buffer 的 alignment/地址；
* 静态库和最终 probe 的哈希、link map、configure `config.log`。

微基准先固定单线程和同一 CPU affinity，使用相同输入、repetitions、warmup；交替 A/B 顺序并做
多个独立进程，报告每轮原始值，不只挑最优值。现有 5000 次结果可作历史对照，但不能和新工具
不同 warmup/布局的数字直接拼接。不要通过“总时间减 copy-only”宣称精确纯 execute 时间。

然后在 16-worker 全程序中复测，保持 ESTIMATE 或 MEASURE、线程数、输入、精度和其余代码不变，
跑所有官方用例/PASS，并纳入冷 plan、读写、分配与写出。微基准方向不能代替这一步。

### 4.3 wisdom 必须分库、分精度、分布局

至少区分以下三种计时，不能混为一个数字：

1. 无 wisdom 的冷 ESTIMATE；
2. 无 wisdom 的冷 MEASURE（生成成本计入本进程）；
3. 离线生成 wisdom 后，新进程 import + 创建 plan + execute。

wisdom 生成和运行必须使用完全相同的 FFTW 库/ISA 构建、精度、n、kind、planner flags、
batch/howmany、stride、对齐与布局。AVX2/AVX-512、double/float、single/plan-many 分别使用
不同文件；import 和 plan 创建分别计时，并把生成成本及部署条件写入结论。import 失败必须显式
记录并回退 ESTIMATE，不能继续把它标成 wisdom-MEASURE。

wisdom 不是跨 CPU、跨 FFTW 构建的通用性能承诺。它也不含也不应含 PCM/expected 特征。
作业 64 的冷 MEASURE 执行潜力支持做这个实验，但离线 wisdom 的运行时 plan 时间、全程序收益
目前均**未测**。

### 4.4 batch / plan-many 是另一个变量，不与 ISA 收益相乘

建议先做 `B=1` 的原 single API 对 plan-many API，再做 `B={2,4,8}` 的小批 R2C；每次仍打印实际
plan。连续布局可定义为输入 `[B][n]`、输出 `[B][n/2+1]`，但须保持 Mel/功率访问与中间存储受限，
不得物化全量功率谱。

新布局可能改变 SIMD、buffered-copy、cache 和后续遍历，planner 也可能选择完全不同的分解。
因此不能假设“ISA 收益 × batch 收益 × wisdom 收益”。每个组合都需独立实测，并检查逐帧数值
和官方输出。

尤其注意 R2C double 的未 padding 输出行字节跨度为 `8*n+16`：n=8192/16384 时相邻行会改变
64 字节相位（通常依次相差 16 字节），即使基底由 `fftw_malloc` 对齐。输入行 `8*n` 则保持 64
字节相位。plan-many 的 base `fftw_alignment_of=0` 不能概括每一行的模 64 地址。
若加 padding，必须将真实 padded stride 传给 planner，并把额外内存、Mel/写出布局成本计入；
它是新布局实验，不是免费对齐修复。

### 4.5 float 与 ISA 分开测

现有构建是 double-only，没有任何 float 新数据。float 需要独立 `--enable-float` prefix、
`fftwf_*` API、`libfftw3f.a` 和 wisdom 文件。probe 以 `FFTW_PROBE_FLOAT=1` 编译。

先分别做 double AVX2/AVX-512 与 float AVX2/AVX-512，不能把 double 的 ISA 比例套到 float。
float 改变向量宽度、流量、对齐、plan 分解以及应用输入转换/功率/输出精度，不只是替换一个库。
任何 float 候选都需完整数值验证；batch float 的复数行跨度还会呈现不同的模 64 相位。
float、batch、wisdom、ISA 的收益均不可假设相乘。

## 5. 如何发现意外 ISA、错误链接和对齐问题

`audit_isa.sh` 提供只读 artifact 审查草案（输出只放本目录）：

* 保存 `file`、SHA256、ELF dynamic/notes/symbol 信息和完整反汇编；
* 用 zmm、opmask、avx512 名称等筛选可疑 AVX-512 位置，供人工结合 opcode/操作数复核；
* 检查最终 executable 和静态 archive，而不是只检查 configure 输出；应用、其他静态库、汇编
  源文件都可能是意外 ISA 来源。

文本筛选是启发式，不是“无 AVX-512”的形式证明：AVX-512 可使用 xmm/ymm + EVEX/mask，
也会有关键字误报。ELF GNU property 只表达所需/兼容属性，不能充当所有指令的上界。
在 AVX2 机器上没触发 illegal instruction 同样只覆盖已执行路径。必要时由主 Agent 安排更完整的
反汇编审计；本 WSL 无 PMU，不编造硬件计数器。

最终链接必须检查 map/trace 和 ELF `NEEDED`，确认没有误用系统 `libfftw3`、错误精度库或旧 prefix。
运行期再核对 probe 打出的 `fftw_version`、`fftw_cc` 与 plan 名称。

## 6. 草案工具边界

* `isa_plan_probe.cpp`：默认只建 plan/检查布局；`--repeats` 大于 0 时才做**单线程**
  copy+execute 微基准。它不做官方数值验证，也不是全程序并发 benchmark。
* 其 half-C2C 仍表示 n_fft 实输入对应的 n_fft/2 复数原地变换，不包含 A1 手工频谱恢复。
* 输出中的每帧 batch 时间只是总 batch 时间除以 B，不是独立单帧测量。
* 第二套 buffer 只用于布局审查；真正应用仍须检查每个 worker 的 new-array 参数。
* 所有脚本均尚未执行，需主 Agent 审查命令、依赖、目标 CPU 和路径后再运行。

## 7. 推荐最小执行顺序

1. 审查并构建 double AVX2/AVX-512 两个独立 prefix；audit archive 与 probe，确认无错误链接。
2. 无 wisdom 跑 16 个冷 plan 组合（2 ISA × 2 n × 2 kind × 2 planner），保存完整 plan/布局。
3. 只对实际生产相关的 R2C 做重复单线程 A/B，再做 16-worker 全程序验证。
4. 分别生成/导入 ESTIMATE、MEASURE wisdom，计 import+plan+execute 和总进程成本。
5. 依次加入 plan-many B=1、小 batch、必要的新 stride；每步重新检查 plan 和每行 alignment。
6. 若数值与精度实验获准，再以独立 float prefix 重复，不继承 double wisdom 或性能结论。

本轮没有执行以上任何步骤，结论均限于已有作业 64 与静态源码审查。
