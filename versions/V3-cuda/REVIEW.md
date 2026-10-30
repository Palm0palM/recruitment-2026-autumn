# Kimi 只读复审与修正

复审子会话：`ses_e738c54c5ffdkmg5XW2tozdhCv`（`kimi-researcher`，high Thinking）。
只读检查正式 CUDA 源和构建文件，没有编辑或执行测试。

主 Agent 已处理：

1. `ContextInit` 先声明/构造 `error` 再启动 thread，消除 worker 在成员初始化前写入的风险。
2. 可执行文件增加 order-only `FORCE` 链接触发，避免切换 CPU/CUDA 或参数时误用旧二进制；
   FORCE 不进入链接输入。CPU 与 CUDA 连续构建后分别通过 8 项 smoke。
3. profile 的每次同步都使用 `cuda_check`，不再丢失诊断构建的同步错误。
4. `CUDA_BATCH` 编译期要求正值；输入超过受保护 CLI 的 int 上限时显式拒绝；
   batch×n_fft/n_bins 的主机算术改 size_t。
5. 文档将区分 host wall 阶段、cuFFT event 与完整进程时间；`cudaMemGetInfo` 只表示设备级使用量，
   pinned 的 D2H 阶段还包含 host staging 分配，不能当作纯传输对照。

上述修正不改变 FP32/FP64 数学路径；初始化重叠变体在修正后重新通过 smoke。
正式 Benchmark 使用修正后新 commit，不以复审前的初始化重叠结果作为最终成绩。
