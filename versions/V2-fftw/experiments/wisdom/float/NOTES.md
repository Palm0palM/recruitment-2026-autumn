# FFTWf + Wisdom 组合私有原型（未编译、未测试）

本目录只实现 float R2C + Wisdom 组合。基线算法源是只读归档 `versions/V2-fftw/experiments/precision/replacement.cpp`；信封设计来自已修复 initial APIC ID 的 double 私有原型。**没有修改父目录的 double 交付源，没有修改正式 `src/`/`include/`，也没有执行 Shell、编译、测试、Git 或远程操作。**

主 Agent 转告的作业 93 double 结果和 float ESTIMATE 结果只是本组合的动机，不是本组合测量。**不得把 double Wisdom 的约 8–12% 与 float 路线收益相乘或相加；float + Wisdom 必须独立做端到端测量。**

## 文件

- `combo.cpp`：归档 float replacement 的完整私有副本。唯一算法行为改动是 `FftwfR2CPlan` 构造函数调用 `melspec_fftwf::plan_r2c_with_float_wisdom`；planner/destructor mutex 保持。`MELSPEC_FFTWF_DOUBLE_POWER=0/1` 两个分支均保留。
- `float_wisdom.h` / `float_wisdom.cpp`：fftwf 信封、CPU/库/布局绑定、单文件/目录导入、明确诊断和 ESTIMATE 回退。
- `float_wisdom_generate.cpp`：独立离线 R2C MEASURE/PATIENT 生成器，无 PCM 参数或音频读取。
- `NOTES.md`：构建、生成和组合验收要求。

## 运行时设计与环境变量

三种状态互斥：

1. **未配置 Wisdom**：`MELSPEC_FFTWF_WISDOM` 与 `MELSPEC_FFTWF_WISDOM_DIRECTORY` 均未设置/为空，直接 fftwf ESTIMATE，不打印 Wisdom 日志。
2. **单文件**：`MELSPEC_FFTWF_WISDOM=/path/file.wisdom`，校验该文件与当前 R2C size。
3. **目录**：`MELSPEC_FFTWF_WISDOM_DIRECTORY=/path/dir`，只构造 `/path/dir/r2c-<n_fft>.wisdom`。不扫描目录、不搜索近似尺寸、不选择其他 planner 文件。

单文件和目录同时非空是配置错误，打印 `reason=wisdom-source-ambiguous` 并回退 ESTIMATE；没有隐式优先级。目录模式下每个尺寸一个精确文件，缺失即 `file-missing-or-unreadable` 回退。MEASURE/PATIENT 应放在不同目录，避免同名文件互相覆盖。

两种模式都必须另设独立：

```text
MELSPEC_FFTWF_LIBRARY_ID=<actual libfftw3f identity>
```

它不使用 `MELSPEC_FFTW_LIBRARY_ID`，防止与 double 库混淆。建议使用实际链接 `libfftw3f.a` 的 SHA-256。只设置 double library id 不能满足 float 导入条件。

### 离线/运行编译隔离

只有定义 `MELSPEC_FFTWF_WISDOM_GENERATOR=1` 的生成器目标才编译离线生成函数及 MEASURE/PATIENT 分支。正常 combo 构建不包含该函数。

有效导入后只调用：

```cpp
fftwf_plan_dft_r2c_1d(n, input, output,
                      FFTW_WISDOM_ONLY | FFTW_ESTIMATE);
```

无适用 plan、文件/身份/layout/checksum 失配时，清除本次 fftwf wisdom 并调用 fftwf ESTIMATE。运行路径不会偷偷 MEASURE/PATIENT，也不会生成或导出 Wisdom。

诊断前缀为 `FFTWf wisdom`：

- 成功：`status=loaded transform=r2c size=N precision=float generated_with=measure|patient runtime=WISDOM_ONLY|ESTIMATE`
- 回退：`status=fallback planner=ESTIMATE precision=float reason=...`

### 信封绑定

沿用 `MELSPEC_FFTW_WISDOM_V1` 的 13 字段结构，但 payload 和身份全部来自 fftwf：

- `fftw_version=fftwf_version`、`fftw_cc=fftwf_cc`；
- `precision=float`；double 文件不能作为 float 文件导入；
- 独立 float `library_id`；
- CPUID leaf 0/1/7.0/extended-1 与 XCR0；
- leaf 1 EBX 仅清除 bits 31:24 的 per-logical-CPU initial APIC ID，保留其余字段；
- `transform=r2c`、实际 `n_fft`、`placement=out-of-place`；
- `fftwf_alignment_of` 的输入/输出 alignment class；
- 离线 planner、payload 长度和 FNV-1a64。

CPUID 身份绑定的是执行上下文兼容性，不是主机名。hybrid/异构 vCPU 的 feature 或 XCR0 差异仍应保守失配。生成与运行应使用相同目标机、相同 float 库及一致 cpuset/core type。

### 内存、plan 和 public API

helper 全部使用 `fftwf_malloc/fftwf_alloc_*`、`fftwf_free`、`fftwf_destroy_plan` 和 `fftwf_alignment_of`，没有混用 double allocator/API。生成器在 MEASURE/PATIENT plan 完成后才写入与 PCM 无关的零值并执行一次。

`combo.cpp` 保留原 float 语义：double 窗口和 double 输入乘积后转 float、double Mel weights/accumulator、最多 16 worker、稀疏 Mel、`n_fft==1` 特例，以及两个 POWER 模式。每次 compute 仍在 worker 启动前持 mutex 创建一个 plan，所有帧通过 new-array execute 复用；worker 不规划。`melspec.h` 和 protected main CLI 不变。

## 精确构建命令（主 Agent 串行执行）

从 worktree 根目录执行，使用系统 fftw3f。不要编译原 `src/melspec.cpp` 或归档 replacement。

```bash
mkdir -p tmp/fftw-lab/wisdom/float/build
mkdir -p tmp/fftw-lab/wisdom/float/generated/measure
mkdir -p tmp/fftw-lab/wisdom/float/generated/patient

g++ -std=c++17 -Wall -Wextra -O3 -march=native -pthread \
  -DMELSPEC_FFTWF_DOUBLE_POWER=1 \
  -Iinclude -Itmp/fftw-lab/wisdom/float $(pkg-config --cflags fftw3f) \
  src/main.cpp \
  tmp/fftw-lab/wisdom/float/combo.cpp \
  tmp/fftw-lab/wisdom/float/float_wisdom.cpp \
  -o tmp/fftw-lab/wisdom/float/build/combo-double-power \
  $(pkg-config --libs fftw3f) -lm -pthread

g++ -std=c++17 -Wall -Wextra -O3 -march=native -pthread \
  -DMELSPEC_FFTWF_DOUBLE_POWER=0 \
  -Iinclude -Itmp/fftw-lab/wisdom/float $(pkg-config --cflags fftw3f) \
  src/main.cpp \
  tmp/fftw-lab/wisdom/float/combo.cpp \
  tmp/fftw-lab/wisdom/float/float_wisdom.cpp \
  -o tmp/fftw-lab/wisdom/float/build/combo-float-power \
  $(pkg-config --libs fftw3f) -lm -pthread

g++ -std=c++17 -Wall -Wextra -O3 -march=native -pthread \
  -DMELSPEC_FFTWF_WISDOM_GENERATOR=1 \
  -Iinclude -Itmp/fftw-lab/wisdom/float $(pkg-config --cflags fftw3f) \
  tmp/fftw-lab/wisdom/float/float_wisdom.cpp \
  tmp/fftw-lab/wisdom/float/float_wisdom_generate.cpp \
  -o tmp/fftw-lab/wisdom/float/build/float-wisdom-generate \
  $(pkg-config --libs fftw3f) -lm -pthread
```

静态库构建时，将 `$(pkg-config --cflags fftw3f)` 换为 `-I/absolute/path/to/FFTW_PREFIX/include`，将链接处 `$(pkg-config --libs fftw3f)` 换为 `/absolute/path/to/FFTW_PREFIX/lib/libfftw3f.a`。生成器和两个 combo 必须使用同一 float 库；不要误链 `libfftw3.a`。

## 精确离线生成命令

以下 size=2048 是公开算法参数示例，不是 PCM 特征调优。应对验收所需的每个公开 `n_fft` 分别生成；PATIENT 可能耗时很长，必须串行。

```bash
# 静态库示例；实际路径和哈希必须由主 Agent 替换/核实。
FLOAT_LIB_ID="$(sha256sum /absolute/path/to/FFTW_PREFIX/lib/libfftw3f.a | cut -d ' ' -f1)"

tmp/fftw-lab/wisdom/float/build/float-wisdom-generate \
  --size 2048 --planner measure --library-id "$FLOAT_LIB_ID" \
  --output tmp/fftw-lab/wisdom/float/generated/measure/r2c-2048.wisdom

tmp/fftw-lab/wisdom/float/build/float-wisdom-generate \
  --size 2048 --planner patient --library-id "$FLOAT_LIB_ID" \
  --output tmp/fftw-lab/wisdom/float/generated/patient/r2c-2048.wisdom
```

生成器 `planning_s` 只覆盖该次离线 fftwf planner 调用，不含分配、执行、导出、写文件及进程启动，不能当作 combo 端到端成绩。

单文件运行示例：

```bash
MELSPEC_FFTWF_LIBRARY_ID="$FLOAT_LIB_ID" \
MELSPEC_FFTWF_WISDOM="$PWD/tmp/fftw-lab/wisdom/float/generated/measure/r2c-2048.wisdom" \
  tmp/fftw-lab/wisdom/float/build/combo-double-power \
  input.pcm tmp/fftw-lab/wisdom/float/build/out.bin \
  --sr 44100 --n_fft 2048 --hop_length 512 --n_mels 128 \
  --f_min 0 --f_max 22050
```

目录运行示例（目录内需为所有被调用尺寸准备精确文件）：

```bash
MELSPEC_FFTWF_LIBRARY_ID="$FLOAT_LIB_ID" \
MELSPEC_FFTWF_WISDOM_DIRECTORY="$PWD/tmp/fftw-lab/wisdom/float/generated/measure" \
  tmp/fftw-lab/wisdom/float/build/combo-double-power \
  input.pcm tmp/fftw-lab/wisdom/float/build/out.bin \
  --sr 44100 --n_fft 2048 --hop_length 512 --n_mels 128 \
  --f_min 0 --f_max 22050
```

## 组合验收要点（全部待主 Agent 执行）

### 1. 先验证无 Wisdom 的 float 基线

两个 POWER 二进制分别运行，三个 float Wisdom 环境变量全部 unset，应直接 ESTIMATE：

```bash
source tmp/session-env.sh
env -u MELSPEC_FFTWF_WISDOM -u MELSPEC_FFTWF_WISDOM_DIRECTORY \
    -u MELSPEC_FFTWF_LIBRARY_ID \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/float/build/combo-double-power

env -u MELSPEC_FFTWF_WISDOM -u MELSPEC_FFTWF_WISDOM_DIRECTORY \
    -u MELSPEC_FFTWF_LIBRARY_ID \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/float/build/combo-float-power
```

再分别跑完整官方 validate，不能只沿用归档 float 结果代替本副本验证。

### 2. 单文件与目录分别验证组合

- 两个 POWER × MEASURE/PATIENT：匹配 size 必须 `status=loaded`，完整输出重新通过 expected。
- 单文件 size 与 CLI `n_fft` 不同：`transform-size-mismatch` 后 ESTIMATE，输出仍须通过。
- 目录模式：准备多个公开尺寸，逐一确认选中 `r2c-N.wisdom`；删除一个尺寸文件，应对该尺寸明确回退，不选择其他尺寸。
- 单文件和目录同时设置：`wisdom-source-ambiguous` 回退。
- 同机不同逻辑 core：initial APIC ID 不应再造成误失配；hybrid feature 差异仍允许保守回退。
- 两个 POWER 共用同一算法 size 的 Wisdom 文件是允许的，因为信封不绑定 POWER；但正确性和性能必须分别测，不能假定两分支收益一致。

### 3. float/double 隔离及失配矩阵

| 测试 | 预期诊断/行为 |
|---|---|
| float source 存在但缺 `MELSPEC_FFTWF_LIBRARY_ID` | `library-id-required`，ESTIMATE |
| 只设置 double `MELSPEC_FFTW_LIBRARY_ID` | 仍视为 float library id 缺失，ESTIMATE |
| 把 double 库哈希填入 float id | `library-id-mismatch`，ESTIMATE |
| 把 double Wisdom 文件传给 float source | version/compiler 或 `precision-mismatch`，不得导入为 fftwf |
| 修改 float 文件 `precision=double` | `precision-mismatch`，ESTIMATE |
| version/cc、CPU/XCR0、size、placement、alignment 失配 | 对应明确原因，ESTIMATE |
| 文件缺失、空、截断、checksum 错、非法字段 | 明确原因，ESTIMATE |
| 导入成功但无适用 plan | `no-applicable-imported-plan`，清 Wisdom 后 ESTIMATE |
| 同机跨 core | 不因 initial APIC ID 失配 |
| `n_fft=1/2/8`、短输入、随机/零/满幅/冲激/Nyquist | 保留 float replacement 语义并通过相应回归 |

### 4. 独立测量组合性能，禁止收益相乘

正确性全部通过后，至少对两个 POWER 分支分别交错比较：

1. float ESTIMATE（无 Wisdom 环境变量）；
2. 相同 float 二进制 + MEASURE Wisdom；
3. 相同 float 二进制 + PATIENT Wisdom。

要求同节点、同 float 库、同编译 flags、同 cpuset、相同完整输入和输出位置；四组完整用例，每组至少 3 次，保留每次 wall、min/median 和 expected 校验。单文件与目录可先分别做正确性；性能比较固定一种部署方式并说明。导入、信封解析和 plan 重建必须计入完整进程时间，离线生成时间另列。

double Wisdom 的既有改进和 float ESTIMATE 的既有改进只能作为分层对照，**不得推算 float+Wisdom 收益**。在组合实测前，不声明更快、等效收益或可叠加性。

## 未验证限制与风险

1. 本目录尚未编译或运行；fftwf 符号、RAII 类型、导入行为、诊断、两个 POWER 分支及目录拼接均待主 Agent 验证。
2. float library id 仍是部署方提供的身份，程序无法从静态进程自证哈希；必须使用实际 `libfftw3f` 身份并保存证据。
3. FNV-1a64 不是安全签名，文件必须来自可信私有目录。目录路径只做字符串拼接，不做沙箱或递归扫描。
4. 单文件和目录同时设置被有意拒绝，不会猜测优先级；每目录同名文件只能保存一种 planner 产物。
5. 当前只支持 R2C basic 1D、interleaved complex、默认 aligned/preserve-input 布局；不支持 half-C2C、batched、stride 或 UNALIGNED。
6. 非 x86 只记录架构名，未实现完整 feature 身份；hybrid/vCPU 环境可能有意保守回退。
7. `fftwf_forget_wisdom` 清除的是 float 全局 planner wisdom，不是 double；当前单组件模型安全，未来嵌入多组件需重新设计所有权。
8. 新增 worker buffer alignment 检查不属于本任务，因此 `combo.cpp` 保持归档实现；仍依赖 `fftwf_malloc` 提供 new-array execute 所需的一致 alignment。
9. Wisdom 可能改变浮点舍入，不能要求与 float ESTIMATE bitwise 相同，必须按官方容差重新验证。
10. 生成器固定 `OUTPUT.tmp`，不支持并发生成同一输出，目标目录须预先存在。

## 许可证

FFTW/FFTWf 为 GPL-2.0-or-later。本目录不重新授权原题或归档实验代码，也不发布二进制。分发链接后的可执行文件时须核对许可兼容性、保留版权/许可并提供对应完整源码与构建材料。
