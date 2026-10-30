# V2 Agent A — FFTW 路线（实验记录）

- Worktree：`melspec-v2-fftw`；分支 `v2/fftw`。
- 严格共同 BASE：`15f90558b4ab83ad0e6500e84aa807a942564229`。
- 不修改 `src/main.cpp`、`validate.py`、`data`；double、周期 Hann、HTK/Slaney、稀疏 Mel 和 V1 线程划分不变。
- AI：OpenCode / GPT-6.1 Sol；实现、测试、测量与报告均在本分支。

## A1：半长 C2C

只将手写 FFT 换成 `fftw_plan_dft_1d(n_fft/2)`；保留奇偶打包、共轭恢复与稀疏 Mel。
每次 `compute_melspectrogram` 只创建一个 plan，所有 worker 用 new-array execute 共享只读 plan。
每线程独立 `fftw_malloc` 缓冲，满足 FFTW 的相同布局/对齐约束；只有外层最多 16 线程。
不链接 FFTW threads/omp 后端。规划/销毁串行并加 mutex，worker 只执行 FFT，不调用 planner。
所有输入在规划完成后填充，以免 `FFTW_MEASURE` 破坏输入。保留单点 FFT 的题意边界。

```bash
source tmp/session-env.sh
make EXEC=tmp/bin/a1-estimate FFTW_PLAN=estimate -j4
make EXEC=tmp/bin/a1-measure FFTW_PLAN=measure -j4
$PY scripts/smoke.py tmp/bin/a1-estimate
$PY scripts/smoke.py tmp/bin/a1-measure
```

初次有效阶段：本地两种 planner 各 8 项 smoke 全 PASS（随机 PCM 的官方四组、CLI 默认值、
`n_fft=1/2/8`）；原始输出见 `raw/a1-*-smoke.txt`。此时尚未测全量性能，不据此宣称收益。

本地已安装 FFTW `3.3.11-1`（pkg-config 3.3.11）。集群不可假定预装库，
通过 `scripts/build_fftw.sh` 无 sudo 自建静态 double 库，依赖均放本实验的 `tmp/`。
编译程序时传 `FFTW_PREFIX=/absolute/path/to/tmp/opt/fftw-3.3.11`，避免运行时 .so 搜索路径依赖。
tarball SHA-256：`5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1`。
默认 `-O3 -march=native`，SSE2/AVX/AVX2，在支持的目标机上启用 AVX-512；不启用 FFTW 内部线程。
源代码/库不提交 Git；FFTW 为 GPL-2.0-or-later，分发链接后的可执行文件时须遵循 GPL、
提供相应完整源码/构建材料并保留许可和版权说明；本实验不重新授权原题代码，也不擅自发布二进制。

后续补充 A2、ESTIMATE/MEASURE 对照、正式集群四组验证与新瓶颈分析。失败实验同样保留。

资料：FFTW 官方 [下载](https://www.fftw.org/download.html)、
[线程安全](https://www.fftw.org/fftw3_doc/Thread-safety.html)、
[new-array execute](https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html)、
[规划标志](https://www.fftw.org/fftw3_doc/Planner-Flags.html)。
