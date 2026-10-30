# fftwf R2C 精度专项

## 状态与边界

- 本目录只有独立实验实现和说明；**本轮未编译、未运行、未测性能，没有任何 PASS 结论**。
- `replacement.cpp` 与受保护的 `src/main.cpp` 链接，**不得同时链接正式 `src/melspec.cpp`**，否则会产生重复符号并混淆实验对象。
- 保持 `compute_melspectrogram(..., std::vector<double>& output)`、公共 helper 和 CLI 不变；`main` 最后仍执行 double 到 float32 的既有转换。
- 正式 A2（`b4552f3962751b5a02d2aa56798907c5a1c29afc`）只作 double 诊断参照。A2 已通过不能推出任何一个 fftwf 变体通过；两个变体必须分别验收。

## 实现和编译选项

共同路径：

1. 周期 Hann 用 double 构造，HTK mel 点、Slaney 面积归一化权重和 FFT bin 频率均用 double 构造。
2. 每点先计算 `double(y[start+i]) * double(window[i])`，再转换为 `float` 写入 fftwf 输入。
3. 一个 `fftwf_plan_dft_r2c_1d(..., FFTW_ESTIMATE)` 被所有 worker 共享，使用 new-array `fftwf_execute_dft_r2c`。不调用 FFTW 内部线程 API，不导入 wisdom，不启用 MEASURE。
4. 每线程有独立的 `fftwf_malloc` 输入、输出 buffer；planner 和 destructor 串行加锁，worker 内不分配、不规划。
5. 最多 16 个外层 worker，线程划分、稀疏 Mel 遍历顺序和 `[mel][frame]` 输出布局与 A2 一致。
6. 不物化全量功率谱；每 worker 只有一个 `vector<double>(n_bins)` 功率 scratch。

两个互斥编译变体：

| 宏 | FFT 和输入 | 模平方 | Mel 权重/累加 | 隔离目的 |
|---|---|---|---|---|
| `-DMELSPEC_FFTWF_DOUBLE_POWER=1` | float | 先将 float re/im 转成 double，再 double 平方和加法 | double | 主要观察输入量化 + float FFT，不引入额外 float 功率舍入 |
| `-DMELSPEC_FFTWF_DOUBLE_POWER=0` | float | float re/im 平方、float 相加，再转 double | double | 在上一项之上加入 float 模平方误差 |

默认宏值为 1，但正式实验构建应显式传宏。不要用 fast-math；两个变体必须使用同一编译器、ISA 和 FFTW 库。最终 CLI 的 float32 转换可能掩盖很小的内部差异，所以“两个二进制最终文件相同”不等于内部功率完全相同。如需分辨被输出量化掩盖的差异，可在本目录另加临时诊断 harness 直接写出 `vector<double>`，但正式验收仍以受保护 CLI 的 float32 输出为准。

### 边界语义

- `n_fft=1`：保留周期单点 Hann `w[0]=0` 的 A2/V1 语义，非零输入也产生零功率；不改成对称窗或 `w[0]=1`。
- `n_fft=2`：走原生 R2C，两个频 bin 都保留；周期窗为 `[0,1]`。
- 短输入：`N<n_fft` 在帧数除法之前拒绝；`N=n_fft` 恰好一帧。
- 尾帧：`n_frames=1+(N-n_fft)/hop_length`，只处理完整帧，不把不完整尾部补零成额外帧。
- `compute_rfft` 也改用 fftwf R2C，返回时把频谱分量转成 double；它不是 double FFT 的等价替代品，若被独立测试也必须按 float FFT 预期审查。

## 误差传播

记加窗后的 double 值为 `x_i=y_i w_i`，float 输入为 `~x_i=x_i+δx_i`。对正常浮点数、round-to-nearest，输入转换有

```text
|δx_i| <= u_f |x_i|,    u_f = 2^-24 ≈ 5.96e-8
```

次正规数应使用绝对半 ULP 界，而不是强行套用该相对界。这里不假设 FTZ/DAZ。由于窗乘仍在 double 中完成，本方案避免了“先把 PCM 和 Hann 都舍成 float 再相乘”的额外一次输入舍入，但不能消除乘积转 float 的误差。

R2C 是线性变换。仅看输入量化，某个频点的保守界为

```text
|δX_k| <= Σ_i |δx_i| <= u_f Σ_i |x_i|
```

该最坏界通常很松；独立零均值量化误差更适合用随 `sqrt(Σ_i x_i^2)` 增长的随机模型。float FFT 各加法、乘法、旋转因子还会增加执行舍入误差。其实际界随分解和阶段数变化，不能用“只有最后一次 float 舍入”或简单的单次 `u_f` 解释。

设 float FFT 返回 `~X_k=re+i·im`：

- **DOUBLE_POWER=1**：float 到 double 转换精确，`double(re)*double(re)+double(im)*double(im)` 只增加 double 量级的功率运算舍入。因此相对 A2 的主要可见误差来自 float 输入和 float FFT。
- **DOUBLE_POWER=0**：还包含 float 两次平方和一次加法的舍入。无溢出/下溢时，这是 `O(u_f)` 的额外相对功率扰动；绝对扰动随 `|X_k|^2` 放大。FMA 与否会改变最后几 ULP，故对照时不能混用编译选项。

Mel 输出满足

```text
M_m = Σ_k W_mk P_k
δM_m ≈ Σ_k W_mk δP_k + double_reduction_rounding
```

权重和功率均非负，Mel 求和本身没有异号大数抵消；主要绝对误差上界可按 `Σ |W_mk δP_k|` 分析。窄带滤波器只累加少数 bin，不代表其输入量化误差必然更小。double 累加保持求和舍入远低于 float FFT 项，但不会消除已经产生的频谱误差。

### 高能量时为什么绝对误差可以很大

相对误差小并不意味绝对误差接近 `1e-4`。例如仅作数量级说明（不是实测结果）：满幅常数经周期 Hann 后，DC 幅度约为 `A·N/2`，功率约为 `A^2·N^2/4`。当 `A≈1, N=16384` 时，该功率约为 `2^26≈6.71e7`。float 在该区间的相邻数间距为 4；即使只有 1 ULP 的功率表示差，也已经是绝对值 4。位于相邻幅值区间的 bin-centered tone 也可产生约 `2^24` 的功率，对应 float 间距 2。

这些 bin 误差还会乘 Mel 权重并累加，最后输出 float32 也有随数值增长的 ULP。因此高能量/full-scale 用例必须同时看绝对误差和相对误差，不能用固定的微小绝对误差上限判断，也不能把大绝对差自动当作算法错误。上面的数值是误差尺度推导，不是候选误差或验收结果。

### 低能量抵消与 relative + absolute 容差

低能量信号中，FFT 的复数频谱可能来自许多异号项相消。真实 `X_k` 很小时，输入量化和 FFT 中间舍入不必同比缩小，结果可出现很大的**相对**误差，甚至符号/小频谱形状变化；平方后仍是非负，但相对功率误差可能显著。Mel 的非负加权和不会恢复已经丢失的频谱细节。

现有检查采用逐元素条件：

```text
|got-ref| <= atol + rtol·|ref|,    rtol=1e-3, atol=1e-4
```

低能量时 `atol` 防止无意义地要求近零值保持极小相对误差；高能量时 `rtol·|ref|` 允许与信号尺度相称的绝对误差。还必须检查 shape 一致和 NaN/Inf；不能靠 allclose 的默认行为替代显式有限性审查。**不降低也不放宽这两个容差，不用 double A2 的历史结果替代 float 候选验收。**

## 构建命令（待主 Agent 执行）

在 worktree 根目录执行。以下直接编译避免 Makefile 对 double FFTW 的既有配置；输出只放本实验目录。两个命令都不包含 `src/melspec.cpp`。

```bash
mkdir -p tmp/fftw-lab/precision/bin

c++ -std=c++17 -O3 -march=native -pthread \
  -DMELSPEC_FFTWF_DOUBLE_POWER=1 \
  -Iinclude -I"$FFTW_PREFIX/include" \
  src/main.cpp tmp/fftw-lab/precision/replacement.cpp \
  -L"$FFTW_PREFIX/lib" -lfftw3f -lm \
  -o tmp/fftw-lab/precision/bin/melspec-fftwf-double-power

c++ -std=c++17 -O3 -march=native -pthread \
  -DMELSPEC_FFTWF_DOUBLE_POWER=0 \
  -Iinclude -I"$FFTW_PREFIX/include" \
  src/main.cpp tmp/fftw-lab/precision/replacement.cpp \
  -L"$FFTW_PREFIX/lib" -lfftw3f -lm \
  -o tmp/fftw-lab/precision/bin/melspec-fftwf-float-power
```

若库装在 `lib64`，相应修改 `-L`。自建依赖只安装 static 库时，上述 `-lfftw3f` 会选 `libfftw3f.a`；也可显式替换为 `"$FFTW_PREFIX/lib/libfftw3f.a"`。不要链接 `-lfftw3`、`-lfftw3f_threads`、OpenMP 或 MPI FFTW 后端。`-march=native` 要求构建与运行 ISA 兼容。

### 无 sudo 的单精度 FFTW 3.3.11 构建参数

仅在获得依赖构建执行安排后，在本实验独立目录解包源码并运行；不安装系统包、不改共享 venv：

```bash
# SOURCE_DIR 是独立解包的 fftw-3.3.11 源码目录；PREFIX 必须是绝对路径。
cd "$SOURCE_DIR"
./configure \
  --prefix="$PREFIX" \
  --enable-float \
  --disable-shared --enable-static \
  --disable-threads --disable-openmp --disable-mpi
make -j4
make install
export FFTW_PREFIX="$PREFIX"
```

`--enable-float` 生成 `fftwf` API 和 `libfftw3f`，不是 double `libfftw3`。单精度和 double 构建应使用不同源码/build/prefix，避免产物混淆。SIMD codelet 参数可按目标机单独追加，但不得在部署机不支持时强行启用 AVX/AVX2/AVX-512；构建与验收日志需保存实际 configure 参数、库和可执行文件哈希。当前没有执行这些命令。

## 可证伪测试矩阵与验收待办

所有合法用例对**两个变体分别**检查：退出成功、shape/帧数正确、输出全有限，并满足原 `rtol=1e-3, atol=1e-4`。先跑现有对照，再跑官方 expected；任何一个变体失败即记录失败，不能取两者较好者冒充共同通过。

| 类别 | 输入/参数 | 必须揭露的问题或解析检查 |
|---|---|---|
| 短输入 | 默认 `n_fft=2048`，`N=0,1,2047`；各边界 N 再测 `N=n_fft-1` | 安全返回失败，不崩溃、不产生伪一帧 |
| 精确帧/尾帧 | `N=n_fft+q·hop+r`，`q=0,1,2`，`r=0,1,hop-1` | 帧数严格为 `q+1`；不完整尾部不影响输出 |
| 单点 | `n_fft=1`，zero、random、full-scale，`hop=1` 和大 hop | 所有合法输出为零；无窗特例回归 |
| 两点 R2C | `n_fft=2`，impulse、正负满幅、random | DC/Nyquist 均处理；解析上每帧只由第二个加窗样本贡献，两个 bin 功率相等 |
| silence | 全零，跨官方参数和 `n_fft=1/2/8/2048` | 输出精确零，无 NaN、非规格化异常或假 Mel 能量 |
| full-scale | 全 `32767`、全 `-32768`，以及两者交替 | 高能量绝对/相对误差、DC、Nyquist、float 功率有限性 |
| impulse | 位于 `0,1,n_fft/2,n_fft-1`、hop 边界、最后一完整帧、被忽略的尾部；幅值正负满幅 | 窗零点冲激应给零帧；帧对齐、符号平方和尾帧正确 |
| Nyquist | `[+A,-A,+A,-A...]` 及反相版本；`n_fft=2/8/2048/8192/16384` | Nyquist bin 不被漏掉或重复，能量尺度合理 |
| tones | on-bin `k·sr/n_fft` 与 off-bin `(k+0.5)·sr/n_fft`，低/中/高频，满幅和 PCM 幅度 `1/16/256` | 相消、频谱泄漏、低能量 atol 区间、高能量 rtol 区间 |
| random PCM | 固定多个 seed 的均匀 int16、低幅 random、稀疏 random；覆盖所有官方参数 | 通用数值误差，不能只拟合官方输入或 expected 特征 |
| 多 worker | 足够多帧使 1/多 worker 分区均出现；重复运行并改变可用 CPU 数 | 输出不随分区变化，无跨线程 buffer/plan 竞争 |

建议执行顺序：

1. 独立构建两个变体，确认链接的是 `libfftw3f`，且未链接正式 `src/melspec.cpp`。
2. 用本地 V1 作现有脚本参照，并保留 A2 double 对照作误差归因：
   ```bash
   source tmp/session-env.sh
   $PY scripts/fftw_check.py --reference tmp/bin/v1 \
     tmp/fftw-lab/precision/bin/melspec-fftwf-double-power \
     tmp/fftw-lab/precision/bin/melspec-fftwf-float-power
   ```
   该脚本报告的是最终 float32 对 V1 的误差，不是内部功率误差。
3. 跑未修改的官方 validator 和全部 official cases；每个二进制单独记录，不引用 A2 的通过日志。
4. 补齐上表 tone、低幅、尾帧和解析边界测试。对高能量同时报告 `max_abs_err`、失败元素的相对误差和容差预算；对低能量报告绝对误差，避免只报一个 max 值。
5. 如需隔离误差来源，比较 `DOUBLE_POWER - A2` 与 `FLOAT_POWER - DOUBLE_POWER`，并注意最终 float32 量化可能掩盖后者；必要时在本目录使用 double-output 诊断 harness。
6. 重复运行检查确定性，再进行主 Agent 安排的串行正式验收。未执行前不得写 PASS，不得提供或暗示性能数字。

若任何用例超出原容差，应保留输入参数、原始输出、误差索引和构建身份，判定该变体未通过并分析原因；不得针对该输入硬编码修正或自行放宽容差。

## 独立强数值反例搜索：`stress_precision.py`

主 Agent 已报告的 official/local/cluster 结果只说明那些输入上的验收结果，不能代替这里的泛化压力检查。本脚本仍未在本轮执行；它固定生成 **54 个小输出用例**，覆盖 `n_fft=256/2048/8192/16384/32768/65536`，每个 N 同时包含 single-frame 和四帧（三个 hop）输入：

- 满幅、严格整数周期 4 和周期 8 信号；
- int16 量化后的 bin-centered tone、half-bin off-bin tone 和 near-Nyquist tone；
- 4 LSB quiet tone；
- 全 `-32768` DC 中恰有一个样本增加 1 LSB 到 `-32767`；
- 强 tone 与 4 LSB 弱 off-bin tone 在量化前叠加；
- 窗零点、窗内、hop 边界和不完整尾部的稀疏正负冲激；
- `n_mels=256` 全频带、`n_mels=64` 全频带以及 `100–4000 Hz` narrow-band。

四帧输出最多只有 `256×4=1024` 个 float32，PCM 最大也只有 `N+3·N/4` 个 int16，不构造全时长功率谱或大 expected 文件。高 N、高 Mel 和 narrow-band 用例全部照常执行，没有按尺寸、信号或既有结果硬编码跳过。

待主 Agent 在 worktree 根目录串行运行（reference 必须指向独立构建的 double A2）：

```bash
source tmp/session-env.sh
export TMPDIR=/path/to/orchestrator-private/tmp
$PY tmp/fftw-lab/precision/stress_precision.py \
  --reference /path/to/a2-estimate \
  --programs \
    double-power=/path/to/melspec-fftwf-double-power \
    float-power=/path/to/melspec-fftwf-float-power \
  --out /path/to/orchestrator-private/stress-report
```

每个 case 先运行 A2 reference，再按命令行顺序逐 candidate、逐进程运行；没有 Python 或 shell 并发。PCM 与临时 `.f32` 只在 `TMPDIR` 下自动清理的 `TemporaryDirectory` 中，`--out` 仅写 `summary.json`。脚本不使用官方 expected 特征，也不改变输入或容差。

### 判定和 JSON 数据

对 reference/candidate 都检查进程返回、文件存在、预期元素数和有限性。合法数组逐元素使用原条件：

```text
abs(got-ref) <= 1e-4 + 1e-3*abs(ref)
normalized_error = abs(got-ref)/(1e-4 + 1e-3*abs(ref))
```

`normalized_error>1` 就是容差失败。每条比较记录最差 normalized error 的 `abs/rel/maxindex/ref/got`，另记录 `max_abs` 及其索引和 `num_failed`。当 reference 为零而 candidate 非零时，相对误差在 JSON 中为 `null`，由 absolute/normalized error 完整表示；NaN/Inf 标量以字符串保存，避免生成非标准 JSON。shape、非有限值、进程失败、timeout 和 reference 异常均作为明确失败原因记录。

单个失败不会中止后续用例。最终 `summary.json` 包含全部 case 参数、信号频率/幅值/相位所需信息、PCM 摘要、每次进程 stdout/stderr/returncode、全部比较和每 candidate 的最差记录；终端只打印进度和汇总。任何 candidate 失败、不可比较或 reference 异常都会使脚本最终返回非零，同时保留完整报告。

这些测试只是**反例搜索**：全部满足容差也不构成对其他频率、幅值、帧数或参数的数学证明。若极端大 `n_fft`、高 `n_mels` 或 narrow-band 暴露失败，应原样保留并归因，不能删除用例、因 official 已通过而忽略、硬编码修正或放宽容差。
