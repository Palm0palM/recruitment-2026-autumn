# B 帧块伪代码（仅供主 Agent 审查）

范围：固定 A2 原生 R2C + ESTIMATE，只替换 hot worker 的循环组织。不是可直接编译的补丁，未修改正式源码。

## 1. 保持不变的设置

- `window`、`mel_fb`、`first/last`、`output.resize`、`n_threads` 和现有整数分区公式不变。
- 每 worker 仍只有一份 `Buffer<double> input(n_fft)` 和一份 `Buffer<fftw_complex> fft_out(n_bins)`，均由 `fftw_malloc` 分配。
- 仍然只创建一个现有单变换 `Plan plan(n_fft,input_buffers[0].get(),fft_buffers[0].get())`；worker 仅调用 new-array execute。
- 不使用 `plan_many`，不创建 B 个 plan，不为每个 b 分配 FFTW 子 buffer。

新增/替换的仅 power 空间，必须在线程启动前完成所有分配：

```cpp
// B 是本次独立实验的固定正整数：1、2、4、8 或 16。
// n_threads、n_bins 已按当前 BASE 计算。
const size_t power_block_len = static_cast<size_t>(B) *
                               static_cast<size_t>(n_bins);
std::vector<std::vector<double>> power_buffers(
    n_threads, std::vector<double>(power_block_len));
```

核心 scratch 公式为 `S(N,B)=(16+4B)N+8B+16` 字节/worker；不分配全量 `[n_frames,n_bins]` 功率谱。

## 2. worker 主循环

```cpp
auto worker = [&](int id) {
    FFTW_COUNTERS(counters);

    auto& input   = input_buffers[id];  // 现有唯一一份 R2C 输入
    auto& fft_out = fft_buffers[id];    // 现有唯一一份 R2C 输出
    auto& power   = power_buffers[id];  // B x n_bins，frame-major

    const int begin = static_cast<int>(
        static_cast<long long>(n_frames) * id / n_threads);
    const int end = static_cast<int>(
        static_cast<long long>(n_frames) * (id + 1) / n_threads);

    // 在每个 worker 自己的连续分区内成块；不是全局分块后再分发。
    for (int block_begin = begin; block_begin < end; ) {
        const int block_frames = std::min(B, end - block_begin);

        // -------------------------------------------------------------
        // A. B 次彼此独立的单帧 pack / R2C execute / power。
        //    input 与 fft_out 每次复用；只保留 B 行 power。
        // -------------------------------------------------------------
        for (int b = 0; b < block_frames; ++b) {
            const int t = block_begin + b;
            const int start = t * hop_length; // 保持 BASE 的索引语义
            double* const power_b =
                power.data() + static_cast<size_t>(b) * n_bins;

            if (n_fft == 1) {
                // 必须保留 BASE 的单点周期 Hann 特例，window[0] == 0。
                const double sample = y[start] * window[0];
                power_b[0] = sample * sample;
            } else {
                FFTW_START(pack_start);
                for (int i = 0; i < n_fft; ++i)
                    input[i] = y[start + i] * window[i];
                FFTW_ADD(counters, 0, pack_start);

                FFTW_START(fft_start);
                // 每个 b 只执行一次现有单变换 Plan；绝不是 Batch FFT。
                plan.apply(input.get(), fft_out.get());
                FFTW_ADD(counters, 1, fft_start);

                FFTW_START(power_start);
                for (int k = 0; k < n_bins; ++k) {
                    power_b[k] = fft_out[k][0] * fft_out[k][0] +
                                 fft_out[k][1] * fft_out[k][1];
                }
                FFTW_ADD(counters, 2, power_start);
            }
        }

        // -------------------------------------------------------------
        // B. m -> block frame -> k。
        //    固定一条权重行，在块内复用；输出沿 t 连续。
        // -------------------------------------------------------------
        FFTW_START(mel_start);
        for (int m = 0; m < n_mels; ++m) {
            for (int b = 0; b < block_frames; ++b) {
                const int t = block_begin + b;
                const double* const power_b =
                    power.data() + static_cast<size_t>(b) * n_bins;

                double acc = 0.0;
                // k 的起点、终点、顺序和累加公式与 BASE 完全一致。
                for (int k = first[m]; k < last[m]; ++k)
                    acc += mel_fb[m][k] * power_b[k];

                output[static_cast<size_t>(m) * n_frames + t] = acc;
            }
        }
        FFTW_ADD(counters, 3, mel_start);

        // 用实际块长推进，自动处理每 worker 的尾块。
        block_begin += block_frames;
    }

    FFTW_PRINT(counters, id);
};
```

## 3. 保持现有失败回退

线程启动代码不需要因 B 改变：

```cpp
std::vector<std::thread> threads;
threads.reserve(n_threads - 1);
int launched = 1;
try {
    for (; launched < n_threads; ++launched)
        threads.emplace_back(worker, launched);
} catch (const std::system_error&) {
    // 保留 BASE：资源不足时未启动分区由主线程完成。
}

worker(0);
for (int id = launched; id < n_threads; ++id)
    worker(id);
for (auto& thread : threads)
    thread.join();
```

所有 power 分配都发生在此之前，worker 不做分配、不创建 plan、不抛预期异常。分配/plan 失败仍在线程启动前向外传播，不新增运行时 B 回退。

## 4. 集成审查清单

- [ ] B=1 时每 worker scratch 与 BASE 相同，且每帧仍按 pack→FFT→power→Mel 的次序完成。
- [ ] B>1 只把 Mel 延后到本块 B 次单帧 FFT 完成，没有减少或合并任何 FFT execute。
- [ ] `n_fft==1` 不调用通用 R2C 路径，每个 b 都写自己的 `power[b][0]`。
- [ ] `n_samples==n_fft`、n_frames=1、音频短尾、每 worker 尾块和 begin==end 均不越界、不漏写。
- [ ] 输出索引含 `(size_t)m*n_frames`，power 索引含 `(size_t)b*n_bins`。
- [ ] 不把连续 slab 中可能未保持 FFTW 对齐的帧子指针传给 `plan.apply`。
- [ ] 不与其他 Mel、线程、NUMA、FFT backend 优化混入同一 B 扫描。
- [ ] 主 Agent 独立执行 smoke、随机/边界、官方四组与同节点 A/B；本原型没有实测收益结论。
