# 冻结版本的空 batch 引用问题

作业348的接口测试中，UBSan报告 `n_fft=1` 时 `batch_buffers[id]` 绑定空引用。
冻结代码只有 n_fft>1 才分配 batch_buffers，却无条件在 worker 开头取得 a 的引用。
最终单帧计算虽然由tail完成，引用本身也已构成UB。AVX2/Batch4回退存在同样问题。

该问题也出现在最初本地 hugepage 接口日志；UBSan默认recover会打印警告但返回0，
所以其中的“PASS 21”只表示数值比较通过，**不是 sanitizer PASS**。保留原始证据，不删除警告。

独立修复：在所有SIMD worker取得batch引用之前，若 `n_batches==0` 直接return。
原有单帧tail随后照常完成，CLI/参数/输出不变；正式四组 n_batches>0，不靠此修复获得加速。
非SIMD单帧回退没有 batch变量，预处理条件确保仍能编译。

后续全部sanitizer作业设置 `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`，
并扩大到24组复用vector/非PCM任意double输入及68组合成CLI测试。
冻结对照保持原始e448c989，不偷偷给对照打补丁；实验补丁必须包含此安全修复。

作业380：修复后的slab、HP、组合各通过68组独立合成输入对照（组合不要求bit-exact，
但官方容差与独立librosa均不变），三个配置的24组API检查均在严格ASan/UBSan下通过，
没有runtime error或sanitizer报告。三个配置还分别完成未修改validate.py的4/4 PASS。
