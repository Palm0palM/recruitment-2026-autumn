# P2：合法 output 初始化的页面成本

独立选项 `-DMELSPEC_OUTPUT_HUGEPAGE=1`，默认关闭；不隐式启用 slab。
Linux 有 MADV_HUGEPAGE 时执行，其他平台/系统保持原 resize。

## Profile 依据与假设

作业335：大输出初始化约.20秒，进程约210k minor faults，明显不是只有 worker 算术。
登录节点 THP 策略是 `[madvise]`，PMD大小2MiB；计算节点策略必须在作业内另记录。
假设：output 先 reserve 标准分配，然后仅对完全包含于分配区域内的完整系统页给出
MADV_HUGEPAGE 提示，之后执行**同一个正常 resize**。可能减少缺页异常与页表成本。

## 安全性与可移植性

- 不改 allocator、vector 布局、size/capacity，不读未初始化内容，不延后初始化。
- reserve 仍由标准 vector 管理存储；resize 逐个构造/初始化 double，不通过 worker 抢写。
- 若有旧 output 内容，仍由 reserve 保留，再由原 resize 保持/补零；接口测试覆盖这种情况。
- madvise 是提示，不改变已有对象值；只覆盖 [向上对齐 data, 向下对齐 data+所需字节数)。
  不包含前后 allocator 元数据或其他分配。使用系统页大小，不硬编码4KiB/2MiB。
- 提示失败时不影响正确性；非Linux/无MADV_HUGEPAGE时直接原路径。不引入新依赖。
- 不设置全机THP策略、不修改NUMA策略、不使用hugetlb专用池、不申请额外权限。

内存大小和初始化流量不变；不能宣称消除了8*output_count写入。THP分配/清零、碎片整理、
内存压力可能使它更慢，所以只能作为可选系统调优，而不是无条件跨平台算法收益。
该方向理论上限是合法初始化约10–24% elapsed；是否改善必须看独立同节点A/B。

## 首轮（作业354，hepnode3）

计算节点确认 `[madvise]` 策略。8次交错median（冻结 → HP，秒）：
`.836368 → .701576`、`.485367 → .453320`、`1.038876 → .904008`、`.996366 → .867797`，
elapsed分别减少约16.1% / 6.6% / 13.0% / 12.9%。这是HP单独选项，slab关闭。
全量expected四组PASS，67组合成输入输出与冻结B16 bit-exact。

诊断显示：大输出resize由约.20降为.080秒，小输出由.050降为.021秒；
大输出AnonHugePages约382976KiB，minor faults从约210k降为115k，RSS没有大幅减少。
因此是页面fault/页表/清零路径改善，而不是跳过初始化或减少output元素数。
worker阶段没有对应幅度改善，符合成本模型。PMU `cycles`不可用，保留原始错误，
不宣称测得cache-miss或硬件周期。

本轮接口sanitizer仍在安全修复前，含已知n_fft=1空batch引用警告，**不算sanitizer通过**。
后续以修复后的严格sanitizer、独立第二轮、官方完整validate与组合实验证实可采用性。

修复后作业364/380：68组合成bit-exact与严格sanitizer的24组接口均PASS；
作业380官方完整validate为4/4 PASS。三次作业每组各8次AB/BA，HP独立改善约6.0–16.1%，
96对逐次比较全部更快。其他Linux的THP配置、碎片、内存压力或容器限制仍可能改变结果，
不把该测量当作无条件平台承诺。
