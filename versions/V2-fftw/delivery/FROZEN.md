# 冻结与交付：float R2C + float power + PATIENT Wisdom

本轮停止大规模实验，仅冻结已验收候选。机器可读清单：[FREEZE.json](FREEZE.json)。

## 1. 冻结身份

| 项目 | 值 |
|---|---|
| 生产代码 commit | `d8234a51f37dbe89a222f8101f73deae1f63f4c0` |
| 候选可执行文件 SHA-256 | `a9d994473813cdc44b5588a8510007c5b2ee73072ba235221b880440c030030c` |
| 运行配置 | R2C + float FFT + float 模平方 + PATIENT Wisdom + 16 外层 worker |
| double 静态库 SHA-256 | `c487eddeb264013f9f051af9fb15ce66d60fc6d15d32ff7eb31501f593554ef5` |
| float 静态库 SHA-256 | `15c7fb154e1a65ed84858cf1776ce8d8a18ec46ddc55f556208f12e098fd761a` |
| FFTW 源码 archive SHA-256 | `5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1` |
| 验收 | Job 151 五轮同作业 Benchmark；Job 209 官方 4/4 PASS |
| 峰值 VmHWM | default 787508 KiB；large_fft 796680 KiB |

生产源文件逐文件哈希见 `../experiments/production-source-sha256.json`；冻结 Wisdom 的逐文件哈希见
`../experiments/raw/cluster/final-wisdom/sha256.json`。交付提交只增加文档/脚本，不改变生产源码。

## 2. 从干净 Checkout 复现

以下命令均在**本实验自己的目录**内执行，不使用 sudo，不改共享 venv。依赖编译必须放在 Slurm 分配内。

```sh
git clone <repo-url> melspec-fftw
cd melspec-fftw
git checkout 05955bf   # 或冻结后的交付提交；生产源与 d8234a5 一致，可用下方校验确认
```

校验生产源码未漂移：

```sh
git diff --exit-code d8234a5 -- src include Makefile
git diff --exit-code BASE -- src/main.cpp validate.py   # BASE=15f90558...
```

获取 FFTW 3.3.11 并校验（仓库不提交第三方源码/二进制）：

```sh
mkdir -p tmp/deps/downloads tmp/deps/src
curl -fL --retry 2 https://www.fftw.org/fftw-3.3.11.tar.gz -o tmp/deps/downloads/fftw-3.3.11.tar.gz
echo "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1  tmp/deps/downloads/fftw-3.3.11.tar.gz" | sha256sum -c -
tar -xzf tmp/deps/downloads/fftw-3.3.11.tar.gz -C tmp/deps/src
```

在 Slurm 作业中构建 float 静态库（double 对照用 `scripts/build_fftw.sh`）：

```sh
mkdir -p tmp/deps/build-fftw-float
cd tmp/deps/build-fftw-float
CC=gcc CFLAGS='-O3 -march=native' ../src/fftw-3.3.11/configure \
  --prefix="$SLURM_SUBMIT_DIR/tmp/opt/fftw-3.3.11-float" \
  --enable-float --enable-static --disable-shared --disable-fortran \
  --disable-threads --disable-openmp --disable-mpi \
  --enable-sse --enable-avx --enable-avx2 --enable-avx512
make -j8 && make install
sha256sum "$SLURM_SUBMIT_DIR/tmp/opt/fftw-3.3.11-float/lib/libfftw3f.a"
# 期望 15c7fb154e1a65ed84858cf1776ce8d8a18ec46ddc55f556208f12e098fd761a
```

编译候选（Release，无插桩）：

```sh
make -B FFT_PRECISION=float FFT_POWER=float FFTW_WISDOM=1 \
  FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11-float" \
  EXEC=tmp/bin/fftw-float-patient
sha256sum tmp/bin/fftw-float-patient   # 同一工具链/目标应复现冻结 SHA
```

Job 251 已在同一集群环境从该提交重新构建，二进制 SHA 与冻结值完全一致
（`a9d99447…c030030c`），说明构建流程可复现，而不是只保留了一次构建产物。

## 3. 部署 PATIENT Wisdom

Wisdom 与 FFTW 库、精度、CPU 特性、尺寸和布局绑定，**不可跨机器强行复用**。

```sh
W="$PWD/versions/V2-fftw/experiments/raw/cluster/final-wisdom/patient"
LIB_ID=$(sha256sum "$PWD/tmp/opt/fftw-3.3.11-float/lib/libfftw3f.a" | cut -d' ' -f1)
export MELSPEC_FFTWF_LIBRARY_ID="$LIB_ID"          # 必须是实际链接库的哈希
export MELSPEC_FFTWF_WISDOM_DIRECTORY="$W"         # 内含 r2c-8192.wisdom / r2c-16384.wisdom
unset MELSPEC_FFTWF_WISDOM                         # 单文件与目录模式互斥
```

同一目标机可直接使用冻结文件；换 CPU/库/编译器时，用生成器重新生成，**不要改 envelope 指纹**：

```sh
make FFT_PRECISION=float FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11-float" wisdom-generator
mkdir -p tmp/wisdom-patient
tmp/bin/fftwf-wisdom-generate --size 8192 --planner patient \
  --library-id "$LIB_ID" --output tmp/wisdom-patient/r2c-8192.wisdom
```

生成成本（hepnode2 实测）：8192 为 2.29 s，16384 为 3.63 s。**该成本不计入每次 CLI 计时**；
若部署无法预先准备，改用无 Wisdom 的 float ESTIMATE（约 1.15×）。

## 4. 运行

```sh
tmp/bin/fftw-float-patient input.pcm output.bin \
  --sr 22050 --n_fft 8192 --hop_length 128 --n_mels 256 --f_max 22050
```

成功加载时 stderr 必须出现：

```
FFTWf wisdom status=loaded ... precision=float generated_with=patient runtime=WISDOM_ONLY|ESTIMATE
```

## 5. 交付前校验（轻量）

```sh
# 逐用例：加载 PATIENT -> 4/4 期望输出；错误 library id -> 明确 fallback 且输出仍通过
python3 scripts/fftw_verify_deployment.py \
  --program tmp/bin/fftw-float-patient \
  --wisdom-dir "$W" --library-id "$LIB_ID" --out tmp/deploy-check.json
```

要求：4 个官方用例 `passed=true`、`status=loaded` 无 fallback；fallback 用例同样通过原容差。
更全面的 envelope 失配矩阵可运行 `scripts/fftw_wisdom_check.py`（19 项）。
`scripts/fftw-profile-final.slurm` 已把上述部署校验与一次目标版本 Profile 放在同一轻量作业中。

**Job 251 已完成该校验**：4/4 `status=loaded` 且无 fallback；错误 library id 时明确
`reason=library-id-mismatch` 回退，输出仍满足原容差。原始记录：
`../experiments/raw/cluster/deploy-final-251.json`；同作业的目标版本 Profile 见
`../PROFILE.md` 第 0 节。

## 6. 交付边界

- 默认 `make` 仍是 double A2；float 是显式近似算术选项，不替代全 double 语义。
- 不提交 FFTW 二进制、可执行文件、PCM、expected 数组或 perf.data。
- FFTW 为 GPL-2.0-or-later；分发二进制时须同时满足对应源码/构建提供义务。
- 不修改其他 worktree，不合并 master；本分支只做交付记录。
