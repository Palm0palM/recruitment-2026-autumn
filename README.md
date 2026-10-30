# 七边形 2026 年秋季招新题目！

> 本仓库的开发/集群评测流程见 [scripts/README.md](scripts/README.md)，优化记录见
> [versions/README.md](versions/README.md)。下文保留题目说明；测试规模以 `validate.py` 为准。

嗨，欢迎来做七边形的招新题！

招新题是一个C++ 小项目，使用 Make 构建。你需要对这个项目做你能想到的性能优化，使其在保持正确性的前提下，跑得尽可能快。

* 我们将会提供七边形的机器供大家在上面编程和调试。最终加速比（也就是性能表现）将会以在七边形机器上测试的结果为准。
* 集群使用 [spack](https://spack.io/) 进行包管理，位置在`\cache\spack`，你可以通过spack找到一些你想要的包。
* 默认的机器（即登录节点）仅有一台，且没有 GPU。集群通过 [slurm](https://slurm.schedmd.com/documentation.html) 进行管理，我们提供了一份示例脚本，请通过slurm提交运行作业，禁止在登录节点上运行。
* 队内集群中数个节点包含GPU。如果你编写了很酷的多机并行算法，或者编写了很酷的 GPU 加速模块，可以使用 slurm 运行你的应用程序，我们允许使用整个集群的资源。
* 如果遇到无法解决的问题，可以联系管理员获取相关支持。

请仔细阅读下面的规则、题目详情和提交方式。

## 题目大意
Mel谱图将原始音频波形转换为基于人耳感知（Mel尺度）频率带的时间-频率表示。流程有6个步骤：

- **步骤1：读取和归一化**

将输入文件读取为16位有符号整数（小端序）并归一化为浮点数：

$$y[n] = \frac{\text{pcm}[n]}{32768}$$

- **步骤2：分帧**

将信号分成重叠帧，长度为`n_fft`，连续帧起始之间的`hop_length`个样本。**不进行零填充**（center=False）：

$$\text{n\\_frames} = 1 + \left\lfloor \frac{N - \text{n\\_fft}}{\text{hop\\_length}} \right\rfloor$$

其中 $N$ 是总样本数。帧 $t$ 包含样本 $y[t \cdot \text{hop\\_length} : t \cdot \text{hop\\_length} + \text{n\\_fft}]$ 。

- **步骤3：加窗**

对每个帧应用**周期性Hann窗口**：

$$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{\text{n\\_fft}}\right), \quad n = 0, 1, \ldots, \text{n\\_fft} - 1$$

> 重要：这是*周期性*形式（周期 = `n_fft`），不是对称形式（分母为`n_fft - 1`）。在scipy中：`scipy.signal.get_window('hann', n_fft, fftbins=True)`。

- **步骤4：FFT和功率谱**

对每个窗口帧计算实值FFT，然后取平方幅度：

$$X[t, k] = \text{FFT}(\text{frame}_t \cdot w)[k], \quad k = 0, 1, \ldots, \text{n\\_fft}/2$$

$$S[k, t] = |X[t, k]|^2$$

这得到形状为`(n_fft/2 + 1, n_frames)`的功率谱矩阵。

- **步骤5：Mel滤波器组**

构建`n_mels`个三角滤波器的组，在HTK Mel尺度上均匀分布：

$$\text{mel}(f) = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)$$

$$f(\text{mel}) = 700 \cdot \left(10^{\text{mel}/2595} - 1\right)$$

构造：
1. 计算`n_mels + 2`个在`mel(f_min)`和`mel(f_max)`之间线性分布的Mel点
2. 转换回Hz得到滤波器中心频率
3. 对于每个滤波器 $m$ ，边缘为 $(f_l, f_c, f_r)$ ，在FFT箱频率 $f$ 处的响应为：

$$H_m(f) = \max\!\left(0, \min\!\left(\frac{f - f_l}{f_c - f_l}, \frac{f_r - f}{f_r - f_c}\right)\right)$$

4. 应用*Slaney面积归一化* —— 用带宽除每个滤波器：

$$H_m(f) \leftarrow H_m(f) \cdot \frac{2}{f_r - f_l}$$

滤波器组矩阵形状为`(n_mels, n_fft/2 + 1)`。

- **步骤6：应用滤波器组**

将滤波器组矩阵乘以功率谱：

$$M = H \cdot S$$

结果是形状为`(n_mels, n_frames)`的Mel谱图。

## 实现要求

我们的代码库结构如下：
```
.
├── data *
│   ├── audio.pcm
│   └── expected
│       ├── default.bin
│       ├── large_fft.bin
│       ├── narrow_band.bin
│       └── small_mels.bin
├── include
│   └── melspec.h
├── Makefile
├── melspec.py
├── README.md
├── slurm.job
├── src
│   ├── main.cpp *
│   └── melspec.cpp
└── validate.py *
```

* 标有星号的文件/文件夹是**不可更改**的。你必须在不修改这些文件的前提下进行优化，保持它们的接口和行为不变。
* 所需的输入文件存储在集群上，具体路径请查看仓库里的链接，禁止更改这些文件。
* 你的输出必须在数值上接近[librosa](https://librosa.org/)的`melspectrogram(y=y, sr=96000, center=False, htk=True)`。提供了一个验证脚本来检查正确性和测量性能。
* 在**数学等效**（例如将`a+b+c`改为`a+(b+c)`）且**结果正确**（误差在可接受范围）的情况下，可以任意修改**不带星号的**项目源文件及构建文件。
* 目录下还包含一个格式文件`.clang-format`，为保持良好编码风格，建议使用[clang-format](https://clang.llvm.org/docs/ClangFormat.html)格式化代码。


### 注意
* 本题目主要考察对该程序的优化，故**禁止**调用任何的音频处理库（librosa, essentia, torchaudio等）。
* **允许**使用科学计算库（如BLAS, LAPACK, FFTW, cuFFT, Eigen, numpy等）来实现算法的某些部分。
* **禁止**根据输入数据的某些特征对程序进行针对性优化。
* **禁止**更改标注为不可更改的地方的值。
* **允许**更改构建脚本，**允许**使用任何别的你认为能获得更高高性能的语言，**允许**使用多线程或GPU加速。

## 编译、运行与测试

### 构建

通过Make构建程序：

```bash
make test
```

### 参数

该程序接受以下参数:

| 参数         | 默认值 | 描述                                           |
|--------------|--------|------------------------------------------------|
| `n_fft`      | 2048   | FFT窗口大小（必须是2的幂）                       |
| `hop_length` | 512    | 连续帧之间的样本数                              |
| `n_mels`     | 128    | Mel频率带数量                                  |
| `sr`         | 44100  | 采样率（Hz）                                   |
| `f_min`      | 0      | Mel滤波器组的最低频率（Hz）                     |
| `f_max`      | 22050  | Mel滤波器组的最高频率（Hz）                     |

### 运行

通过以下CLI接口运行程序：

```
<program> <input_pcm> <output_bin> [--sr SR] [--n_fft N_FFT] [--hop_length HOP] [--n_mels N_MELS] [--f_min F_MIN] [--f_max F_MAX]
```

位置参数：
1. `input_pcm` — 输入PCM文件路径
2. `output_bin` — 输出二进制文件路径

所有选项参数都是可选的，应使用上述默认值。

**示例：**
```bash
# 默认参数
./melspec data/audio.pcm output.bin

# 自定义参数
./melspec data/audio.pcm output.bin --n_fft 4096 --n_mels 64 --f_min 300 --f_max 8000
```

### 二进制I/O格式

**输入** (`data/audio.pcm`)：
- 单声道，16位有符号整数，小端序，96000 Hz
- 作为原始字节读取（无头，无容器格式）

**输出**：
- `float32`，小端序，行优先
- 形状：`[n_mels, n_frames]` — Mel带是行，时间帧是列
- 无头 —— 只是按序写入原始float32值

### 验证

验证程序的运行需要`numpy`和`librosa`，请自行安装Python环境。

运行验证脚本来检查你的实现与真值输出的对比：

```bash
python validate.py ./melspec
```

将这用4个不同参数集运行你的程序并报告：
- **最大绝对误差**与预先计算的参考值对比
- **通过/失败**状态（容差：`rtol=1e-3, atol=1e-4`）
- **计算时间**（3次运行中的最小值）
- **加速比**对比librosa的加速比（如果安装了librosa）

### 参考实现

除了C++实现，我们还提供了一个Python实现`melspec.py`，它使用NumPy和SciPy来实现Mel谱图的计算，它的CLI接口与C++版本相同。

同样的，请确保你的环境中安装了`numpy`和`scipy`。

Python实现是为了帮助你理解算法细节，我们不建议你直接使用它来进行性能优化。C++实现则是一个相对朴素的版本，可能存在一些性能很低的地方，欢迎你在此基础上进行优化。

**注意，朴素实现速度非常慢，默认算例在集群上运行会很耗时，请使用较小的算例调试。**

## 提交

你需要 fork 题目仓库，在自己的仓库上进行开发，注意，良好的版本控制习惯也是考查的一部分。

请注意做题过程中将仓库设为`private`，在截止后再设为`public`。

与此同时，你需要制作一份 ppt，内容如下：

* 每一步优化的思路、过程和效果（举例：使用了 xxx 优化，相对于原代码，速度提升了 114.514 倍）
* 你对程序进行的 profile，分析出来的性能瓶颈
* 你在解题过程中所参考的资料，包括使用的AI工具
* 在解题过程中，遇到的有意思的事情，或者是让你印象深刻的 bug（可选）

另外，在笔试结束后，你需要使用这份PPT向我们展示，介绍你的优化成果。

## 关于AI

我们欢迎使用AI工具来辅助你完成这道题目，但请务必遵守以下原则：
* **透明性**：在展示中明确说明你使用了哪些AI工具，以及它们在解题过程中发挥了什么作用。
* **原创性**：AI工具可以帮助你理解概念、提供代码示例或者优化建议，但最终的代码实现必须是你自己的原创作品，请确保你理解了AI在做什么，以及为什么要这么做。
* **正确性**：无论是否使用AI工具，最终提交的代码都必须满足题目的正确性要求，并通过验证脚本的测试。
* **诚信**：请勿使用AI工具来生成虚假的优化结果或者性能数据。

同样的，我们依旧会针对你的优化提出问题，如果你无法解释你的优化，我们将会取消你的招新资格。

## 提示

* 通常来说，优化的第一步是**分析**，也就是找出性能瓶颈在哪里。之后，常见的方向包括算法优化、并行化（多进程、多线程、多机）、SIMD优化、内存访问优化、GPU加速等。
* 在优化过程中，你可以使用**整个**集群的资源。我们欢迎你尝试一些大胆的优化，充分利用集群的计算能力。
* 良好的版本控制习惯可以有效避免 “改了一下午代码后它跑不起来了，但是我又改不回去了” 之类的极端情况。请充分使用 [Git](https://git-scm.com/)
* 使用 htop/glances 或者资源管理器查看 CPU 负载是验证自己程序正在并行工作的好方法之一。
* Perf 是 Linux 下常用的性能分析工具：<https://www.brendangregg.com/perf.html>
* 火焰图可以直观地展示程序的性能热点：<https://www.brendangregg.com/flamegraphs.html>
* OpenMP 是常用的并行计算框架：<https://bisqwit.iki.fi/story/howto/openmp/>
* 如果需要做多机的并行，你可能会对 MPI 感兴趣：<https://mpitutorial.com/tutorials/>
* SIMD 是提升程序单核性能的有效手段：<https://zhuanlan.zhihu.com/p/583326378> <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#>
* 适量的 Cache 知识不仅对单核优化有用，也能避免一些多线程场景下的诡异问题：<https://zhuanlan.zhihu.com/p/136300660>
* 更加系统性的高性能计算知识，请参考七边形的 HPC 学习路线：<https://heptagonhust.github.io/HPC-roadmap/>
* Slurm 怎么用: <http://scc.ustc.edu.cn/zlsc/user_doc/html/slurm/index.html>

以及一些技术无关的提示：

* 如果遇到问题，请积极在 QQ 群内或现场提问。
* 部分优化可能相当难做/难调。所以即使没有写出程序/调试成功，也欢迎把自己的天才优化想法写进提交的文档里，通过文字来展示自己对高性能计算的理解和认知。
* 比起最终的结果，我们更关注你展现出来的能力和思考。
