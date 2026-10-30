# FFTW Wisdom 私有原型（未编译、未测试）

基线：A2 ESTIMATE 正式候选 `b4552f3962751b5a02d2aa56798907c5a1c29afc`。
本目录是独立原型，不修改 `src/`、`include/`、`src/main.cpp`、validator 或数据。
**当前没有执行 Shell、编译、正确性测试或性能测量，不能据此声称正确性或加速。**

## 文件

- `melspec.cpp`：A2 私有副本；保留 double、周期 Hann、HTK/Slaney、稀疏 Mel、最多 16 worker、线程分区及 `n_fft==1` 特例。另检查所有 new-array worker buffer 的 FFTW alignment class。
- `fftw_support.h`：私有 planner/buffer 包装。每次 `compute_melspectrogram` 仍只创建一个 plan，plan 在 worker 启动前创建，输入在 plan 后填充。
- `fftw_wisdom.h` / `fftw_wisdom.cpp`：版本化 wisdom 信封、CPU/库/布局身份、严格导入、运行时 `WISDOM_ONLY` 与 ESTIMATE 回退，以及离线生成实现。
- `fftw_wisdom_generate.cpp`：独立离线 CLI；只接受 transform/size/planner/library-id/output，**没有 PCM 输入，也不读取音频**。
- `NOTES.md`：设计、构建、验收矩阵和限制。

## 设计

### 离线生成与运行时严格分离

只有 `generate_wisdom_file` 能选择 `FFTW_MEASURE` 或 `FFTW_PATIENT`，且该函数及声明仅在生成器构建定义 `MELSPEC_FFTW_WISDOM_GENERATOR=1` 时编译；运行程序目标文件不包含离线生成函数。生成器在目标机上：

1. `fftw_forget_wisdom()`，分配与运行时相同的 double/basic-1D 布局；
2. 用所选 MEASURE/PATIENT 创建一个 plan；
3. **规划完成后**才以与 PCM 无关的零值初始化并执行一次；
4. 导出 FFTW wisdom，外加私有元数据信封，先写 `OUTPUT.tmp` 再 rename。

运行时环境变量：

- `MELSPEC_FFTW_WISDOM`：可选，单个信封文件路径；未设置/为空就是原 A2 ESTIMATE 路径。
- `MELSPEC_FFTW_LIBRARY_ID`：设置 wisdom 路径时必填，必须与信封中的库标识完全一致。建议使用实际链接 `libfftw3.a` 的 SHA-256。

设置 wisdom 路径后，运行时先清空本次全局 wisdom，再校验并导入信封。只有校验和导入都成功才调用：

```cpp
fftw_plan_dft_*(..., FFTW_WISDOM_ONLY | FFTW_ESTIMATE)
```

FFTW 官方允许较高 rigor 生成的 wisdom 用于较低 rigor 请求，因此 MEASURE/PATIENT wisdom 可用于 ESTIMATE 查询；`WISDOM_ONLY` 禁止缺失 wisdom 时进行实际测量或启发式规划。运行路径没有任何 MEASURE/PATIENT planner 调用。

以下情况均在 stderr 给出 `status=fallback planner=ESTIMATE reason=...`，随后明确创建 ESTIMATE plan：

- 文件缺失、不可读、过大、截断或格式错误；
- 缺少/错误的 library id；
- FFTW 版本、构建 flags、精度、CPU、transform、尺寸、placement 或 alignment 失配；
- payload 长度/FNV 校验失败、FFTW 拒绝导入；
- 导入成功但 `WISDOM_ONLY` 无适用 plan（先 `fftw_forget_wisdom()`，避免部分 wisdom 污染回退）。

成功导入打印 `status=loaded ... runtime=WISDOM_ONLY|ESTIMATE`。运行时不生成、不导出、不自动“修复”wisdom。CLI 参数及 `melspec.h` public API 不变。

### 身份和布局绑定

信封固定为 `MELSPEC_FFTW_WISDOM_V1`，含：

- `fftw_version` 与 `fftw_cc`；
- `precision=double`；
- 调用方提供的 `library_id`；
- x86 上 CPUID leaf 0/1/7.0/extended-1 与 XCR0（涵盖模型和 SIMD/OS 状态）；leaf 1 EBX 的 initial APIC ID 已屏蔽；
- `transform=r2c|half-c2c`、实际 FFTW plan size；
- R2C out-of-place / C2C in-place placement；
- 规划输入/输出的 `fftw_alignment_of` class；
- 离线 planner（measure/patient）、payload 长度和 FNV-1a64。

R2C 的 size 是 `n_fft`；半长 C2C 的 size 是 `n_fft/2`（`n_fft==1` 时原实现也用 half=1）。payload 是 FFTW planner knowledge，不是序列化机器码；导入后仍需重建 plan。

### CPU 身份复核与主 Agent 审查修复

主 Agent 审查发现：初版把 CPUID leaf 1 的整个 EBX 写入身份，其中 bits 31:24 是**当前逻辑 CPU 的 initial APIC ID**。生成器与运行进程被调度到不同 core 时，该字段可能不同，造成同机误报 `cpu-mismatch`。现已仅执行 `b &= 0x00ffffffu` 后再记录 leaf 1，保留 EBX bits 23:16 的 CLFLUSH line size、bits 15:8 的 logical processor count 和 bits 7:0 的 brand index。协议结构和其余实现不变；修复前生成的信封会因 CPU 身份编码变化而保守回退，必须重新生成，不能手工复用旧身份。

其余已查询字段中没有另一个像 initial APIC ID 这样专门标识“当前逻辑 CPU”的字段：leaf 0 是 vendor/max leaf，leaf 1 EAX 是 family/model/stepping，leaf 1 ECX/EDX、leaf 7.0 和 extended leaf 1 主要是 feature/capability 信息。它们在同构 x86 机器上通常跨 core 一致，但 hybrid CPU、异构 vCPU 或特殊虚拟化配置可能按 core type/vCPU 暴露不同能力；XCR0 也是当前逻辑 CPU 的 OS SIMD 状态，通常全机一致但未被强行归一化。

因此这里绑定的是**实际执行上下文的严格兼容性**，不是可与任意 core 无关的“主机名”。不屏蔽剩余 feature/model/XCR0 差异：异构环境下仍可能 `cpu-mismatch`，这是为避免载入不适用 SIMD wisdom 的保守回退。生成器与 worker 应在相同目标机、相同库及一致的 cpuset/core type 下运行。该身份不包含当前 APIC ID、频率、进程号或 PCM 特征。

### plan 与 buffer 生命周期

热路径延续 A2：所有线程的 scratch 先分配，在主线程持 planner mutex 创建一个 plan，再启动 worker。所有帧和所有 worker 通过 new-array execute 复用同一 plan，每 worker 使用独立 `fftw_malloc` buffer。私有副本额外验证这些 buffer 与规划 buffer alignment class 相同。输入拷贝/加窗始终发生在 plan 后。没有跨 public API 调用的静态 plan cache，也没有 worker planner 调用。

## 构建（主 Agent 串行执行）

从 worktree 根目录执行。以下使用系统 FFTW；命令不会编译原 `src/melspec.cpp`。

```bash
mkdir -p tmp/fftw-lab/wisdom/build tmp/fftw-lab/wisdom/generated

g++ -std=c++17 -Wall -Wextra -O3 -march=native -pthread \
  -DMELSPEC_FFTW_R2C=1 -Iinclude -Itmp/fftw-lab/wisdom \
  $(pkg-config --cflags fftw3) \
  src/main.cpp \
  tmp/fftw-lab/wisdom/melspec.cpp \
  tmp/fftw-lab/wisdom/fftw_wisdom.cpp \
  -o tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c \
  $(pkg-config --libs fftw3) -lm -pthread

g++ -std=c++17 -Wall -Wextra -O3 -march=native -pthread \
  -DMELSPEC_FFTW_R2C=0 -Iinclude -Itmp/fftw-lab/wisdom \
  $(pkg-config --cflags fftw3) \
  src/main.cpp \
  tmp/fftw-lab/wisdom/melspec.cpp \
  tmp/fftw-lab/wisdom/fftw_wisdom.cpp \
  -o tmp/fftw-lab/wisdom/build/melspec-wisdom-c2c \
  $(pkg-config --libs fftw3) -lm -pthread

g++ -std=c++17 -Wall -Wextra -O3 -march=native -pthread \
  -DMELSPEC_FFTW_WISDOM_GENERATOR=1 \
  -Iinclude -Itmp/fftw-lab/wisdom $(pkg-config --cflags fftw3) \
  tmp/fftw-lab/wisdom/fftw_wisdom.cpp \
  tmp/fftw-lab/wisdom/fftw_wisdom_generate.cpp \
  -o tmp/fftw-lab/wisdom/build/fftw-wisdom-generate \
  $(pkg-config --libs fftw3) -lm -pthread
```

**为什么需要私有 `melspec.cpp` 副本：** `#include "fftw_support.h"` 对 quoted include 会先搜索包含它的源文件所在目录。只传 `-Itmp/fftw-lab/wisdom` 而继续编译 `src/melspec.cpp`，仍会先命中 `include/fftw_support.h`（项目原有 quoted-include 搜索路径），从而错误使用正式头。上述命令编译私有源副本，因此命中同目录私有头；`src/main.cpp` 仍通过 `-Iinclude` 使用只读 public 头。

使用实验自建静态 FFTW 时，将三条命令中的 `$(pkg-config --cflags fftw3)` 替换为：

```bash
-I/absolute/path/to/FFTW_PREFIX/include
```

并把链接处的 `$(pkg-config --libs fftw3)` 替换为：

```bash
/absolute/path/to/FFTW_PREFIX/lib/libfftw3.a
```

不要用不同 FFTW prefix 编译生成器和运行程序。可选诊断构建给 melspec 两条编译命令增加 `-DMELSPEC_PROFILE=1`；其 `r2c_plan_s`/`c2c_plan_s` 会包含信封读取、校验、导入和 plan 创建，不等同于纯 FFTW planner 时间。

## 离线生成与运行示例

示例 size=2048 只是算法参数示例，不是根据 PCM 特征选择。对所需公开 FFT 参数分别生成文件；半长 C2C 的原始 `n_fft=2048` 应传 `--size 1024`。

```bash
# 只有实际库哈希确为此值时才使用文档中的既有集群库 ID：
LIB_ID=c487eddeb264013f9f051af9fb15ce66d60fc6d15d32ff7eb31501f593554ef5

# 新构建应重新计算，不能照抄旧哈希：
sha256sum /absolute/path/to/FFTW_PREFIX/lib/libfftw3.a

tmp/fftw-lab/wisdom/build/fftw-wisdom-generate \
  --transform r2c --size 2048 --planner measure \
  --library-id "$LIB_ID" \
  --output tmp/fftw-lab/wisdom/generated/r2c-2048.measure.wisdom

tmp/fftw-lab/wisdom/build/fftw-wisdom-generate \
  --transform r2c --size 2048 --planner patient \
  --library-id "$LIB_ID" \
  --output tmp/fftw-lab/wisdom/generated/r2c-2048.patient.wisdom

MELSPEC_FFTW_LIBRARY_ID="$LIB_ID" \
MELSPEC_FFTW_WISDOM="$PWD/tmp/fftw-lab/wisdom/generated/r2c-2048.measure.wisdom" \
  tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c \
  input.pcm tmp/fftw-lab/wisdom/build/out.bin \
  --sr 44100 --n_fft 2048 --hop_length 512 --n_mels 128 \
  --f_min 0 --f_max 22050
```

生成器输出的 `planning_s` 仅是该次离线 FFTW planner 调用的 wall time，不含进程启动、buffer 分配、一次执行、导出和写文件；不能当作端到端成绩。PATIENT 可能很慢，必须在获准的离线阶段串行生成。

## 主 Agent 验收命令

先按上文构建。无 wisdom 环境变量时必须走 ESTIMATE：

```bash
source tmp/session-env.sh
env -u MELSPEC_FFTW_WISDOM -u MELSPEC_FFTW_LIBRARY_ID \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c
env -u MELSPEC_FFTW_WISDOM -u MELSPEC_FFTW_LIBRARY_ID \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/build/melspec-wisdom-c2c
```

对 size=2048 的有效 R2C 文件运行 smoke：CLI defaults/对应 2048 用例应报告 loaded，其他 FFT size 应明确回退且仍通过。MEASURE 与 PATIENT 分开执行：

```bash
MELSPEC_FFTW_LIBRARY_ID="$LIB_ID" \
MELSPEC_FFTW_WISDOM="$PWD/tmp/fftw-lab/wisdom/generated/r2c-2048.measure.wisdom" \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c

MELSPEC_FFTW_LIBRARY_ID="$LIB_ID" \
MELSPEC_FFTW_WISDOM="$PWD/tmp/fftw-lab/wisdom/generated/r2c-2048.patient.wisdom" \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c
```

缺失文件/库 ID 的安全回退（程序应正常完成，stderr 有原因）：

```bash
MELSPEC_FFTW_LIBRARY_ID="$LIB_ID" \
MELSPEC_FFTW_WISDOM="$PWD/tmp/fftw-lab/wisdom/generated/does-not-exist" \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c

MELSPEC_FFTW_WISDOM="$PWD/tmp/fftw-lab/wisdom/generated/r2c-2048.measure.wisdom" \
  $PY scripts/smoke.py tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c
```

确定性边界/短输入回归（需主 Agent 先按原 REPRODUCE 构建独立 V1 reference）：

```bash
$PY scripts/fftw_check.py --reference tmp/bin/v1 \
  tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c \
  tmp/fftw-lab/wisdom/build/melspec-wisdom-c2c
```

完整官方验证：

```bash
env -u MELSPEC_FFTW_WISDOM -u MELSPEC_FFTW_LIBRARY_ID \
  $PY validate.py tmp/fftw-lab/wisdom/build/melspec-wisdom-r2c
```

若要对四组都使用 wisdom，应仅按各组公开的 `n_fft`/transform 参数提前生成对应文件并逐组运行；单文件实现不会对其他尺寸自动生成，而是回退 ESTIMATE。

性能只在正确性和失配矩阵通过后测量。至少将 A2 ESTIMATE、有效 MEASURE wisdom、有效 PATIENT wisdom 在同节点同库同 cpuset 交错运行，覆盖四组完整输入，每组至少 3 次，保留 min/median、完整进程 wall 和每次输出校验。离线生成耗时单独报告；运行进程内的导入+重建 plan 成本必须计入成绩。不能以生成器 planning_s 或单线程 FFT probe 推算总加速。

## 测试矩阵（全部待主 Agent 执行）

| 类别 | 情况 | 预期 |
|---|---|---|
| 构建 | R2C、half-C2C、generator；系统/静态 double FFTW | 无原正式文件参与覆盖；无新增 warning/error（若有需记录） |
| 基线 | 两个 wisdom 环境变量均 unset | 无 wisdom 日志，ESTIMATE；输出与 A2/V1 在题设容差内 |
| 有效导入 | R2C，MEASURE 与 PATIENT，各匹配 size | `status=loaded`，runtime=`WISDOM_ONLY|ESTIMATE`，输出通过 |
| C2C | 对 hot half-C2C 传实际 half size | loaded 且恢复频谱/输出通过；size 传错则回退 |
| 缺失 | 路径不存在、无权限、空文件、超过 16 MiB | 明确原因，ESTIMATE，正常计算 |
| 库身份 | library id 缺失、错误；FFTW version/cc 失配 | 不导入 payload，ESTIMATE |
| 同机跨 core | 生成器与 worker 在同构机器的不同逻辑 CPU 上运行 | initial APIC ID 不参与身份，不应仅因调度 core 不同而失配；输出通过 |
| 机器 | 不同 CPUID/XCR0、异构 core type 或在私有副本修改 cpu 字段 | `cpu-mismatch`，ESTIMATE；异构保守失配属于绑定限制 |
| 精度 | precision 字段非 double | `precision-mismatch`，ESTIMATE |
| 布局 | R2C/C2C 交叉、size/placement/alignment 失配 | 对应明确原因，ESTIMATE |
| 完整性 | payload 截断/改字节、长度错、FNV 错、NUL | 导入前拒绝或 FFTW import 拒绝，ESTIMATE |
| 信封 | magic/字段缺失/重复/未知字段/非法 planner | 严格拒绝，ESTIMATE |
| 无适用 plan | 有效信封和可导入 payload，但无对应 plan | `no-applicable-imported-plan`，清空后 ESTIMATE |
| 通用语义 | `n_fft=1/2/8`、精确帧边界、随机/零/满幅/冲激/Nyquist | smoke + fftw_check 全通过；短输入安全拒绝 |
| 正式正确性 | 四组官方输入，无 wisdom/有效 wisdom/各回退 | 官方容差全 PASS |
| 生命周期 | profile 检查每次 compute 一条 plan 计时，worker 无 plan 日志 | plan 仅在 worker 前一次创建；所有帧复用 |
| 性能 | 三候选同作业交错完整进程测量 | 记录实测 min/median；未测前无结论 |

## 风险与未验证限制

1. **尚未编译。** 类型、FFTW 3.3.11 实际导入行为、CPUID/XCR0 代码、诊断和所有正确性均待主 Agent 验证。
2. `library_id` 是操作约束，程序无法从静态链接进程反推并证明它等于实际库哈希。生成/运行环境若填写错误，库身份保护会被绕过；部署脚本应绑定真实哈希并保存证据。
3. FNV-1a64 只防意外损坏，不是签名。元数据也可被有意重写；wisdom 文件必须来自可信私有目录，不能视为不可信输入安全边界。
4. x86 身份较严格；非 x86 当前只记录架构名，未实现完整 CPU feature 绑定，因此该原型在非 x86 上不应被视为满足可移植身份要求。
5. 信封绑定的是 basic 1D、interleaved complex、默认 aligned/preserve-input 布局。未来若改 advanced/batched、stride、destroy-input、float 或 `FFTW_UNALIGNED`，必须升级格式，不能复用本文件。
6. 每进程只接受一个 wisdom 文件。当前 CLI 一次只计算一个 FFT size，所以足够；批量多 size 或常驻服务需要受控多文件/缓存设计，不能隐式拼接未验证 payload。
7. FFTW wisdom 是全局 planner 状态。本实现按原程序单次 plan 模型串行处理；若未来其他线程/库同时使用 FFTW planner，需要扩大同步和所有权设计。
8. `fftw_forget_wisdom()` 不影响已创建 plan，但会清除进程内其他已积累 wisdom。当前程序没有多组件 wisdom 所有者；嵌入其他应用时这是 public API 语义风险。
9. 离线 PATIENT/MEASURE 选择可能受目标机测量噪声影响，数值舍入也可能与 ESTIMATE 不同；必须重新验证，不能假定 bitwise 一致。
10. 生成器用固定 `.tmp` sibling，不支持并发生成同一输出；主 Agent 应串行执行。目标目录需预先存在。
11. wisdom 导入并非零成本，仍需文件 I/O、解析和 plan 重建。MEASURE/PATIENT 微内核优势是否覆盖这些成本未知，不能从已有 probe 直接推断 16-worker 总收益。
12. 新增成功/回退 stderr 日志便于审计；若未来外层协议把 stderr 当机器可读输出，需要单独规范，但当前受保护 CLI 不依赖 stderr 作为数据输出。

## 许可证

FFTW 是 GPL-2.0-or-later。这里不重新授权原题或 A2 代码，也不发布二进制。若分发静态/动态链接 FFTW 的可执行文件，须核对原题代码许可兼容性、保留版权和许可，并履行对应完整源码与构建材料义务；只附下载 URL 不足以假定合规。
