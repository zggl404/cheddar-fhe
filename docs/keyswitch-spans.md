# KS / Hybrid GPU 阶段与起止时间

NVTX 主口径 `nsys_keyswitch_segment_gpu_span_v1` 同时支持 `--split-hybrid on/off`，
两者均按显式连续阶段的 GPU 首尾跨度统计，包含段内间隔。
采集显式使用 `--cuda-event-trace=false`，关闭 Nsight 自身的 Event trace。
保留 `BSFusedKernel` 的现有实现。CPU 使用同步操作墙钟；Event 在两种模式下均
直接复用 `KsTraceSegment` 的进入/退出边界、分类和 ID，独立运行后与 NVTX 逐段配对。

```bash
python scripts/run_inference.py --model resnet20 --build-dir build \
  --cifar "$CIFAR_BIN" --encode-mode pre --key-mode load --key-dir /data/keys/resnet20 \
  --time-nvtx on --split-hybrid on --count 6 --warmup 1 \
  --timing-dir validation/resnet20-ks-spans
```

批量脚本保持基线和采集两次独立 load，同一密钥、权重、图片、编码与拆分参数。三开关全 OFF
不打印计时；`--timing-file` 可单独保存粗粒度数据。所有总量含 6 张，均值排除第 1 张、使用后 5 张。

## 边界

| kind | 记录的阶段 | 后续 NonKS |
|---|---|---|
| multkey | core: Decompose/ModUp、内积、修正、b/a ModDown | 独立置换、额外 Add |
| relinearize_rescale | core: Decompose/ModUp、内积、修正、b/a ModDown | b/a Rescale |
| baby | prepare: 共享 ModUp、PseudoModUp、BSFusedKernel；finish: 消费者的 b/a ModDown | 明文权重乘加、Rescale |
| giant | core: 前置 a ModDown、ModUp、内积、修正 | 独立置换、跨 Giant Add |
| sparse_to_dense | core: P 扩展/NTT、专用密钥乘法、修正、b/a ModDown | 共同短基前置、Q 恢复 |
| shared_finish | 有非零 Giant 时，整个 LinearTransform 最终一次 b/a ModDown | Rescale |
| shared_prepare | 仅 Giant 的退化路径中的共享 ModUp/PseudoModUp | 明文权重乘法、置换 |

后两项是共享阶段，不新增逻辑 KS 或执行调用数量。无非零 Giant 的 finish 关联原 Baby ID；
有非零 Giant 的 finish 归整个 `lt_id`，不分摊给每次旋转。ModDown/Rescale 拆分路径先完成两个
分量的 ModDown，再分别 Rescale；没有改变各分量内部的运算顺序。

`split-hybrid off` 保留原融合计算，仍使用相同的 KS 阶段边界；以下混合工作另列为 Hybrid：

| kind / segment | Hybrid 阶段 |
|---|---|
| relinearize_rescale / finish | b/a 两个分量的 ModDownAndRescale；前面的 KS core 在此之前结束 |
| baby 或 shared_finish / finish | LinearTransform 最终 b/a ModDownAndRescale，与前面的权重乘加分开 |
| shared_lift / prepare | Sparse→Dense 中 Q/P 共用的 lift |
| automorphism_accum / core | 一次融合累加调用中的全部 kernel |

相邻的 KS 与 Hybrid 使用不同 segment_id，互不嵌套。b/a 合并为一个 finish，
其中的间隔属于该 Hybrid；阶段之间的权重乘加及其余工作属于 NonKS。
融合累加超过 8 项时，递归子调用继承最外层 Hybrid 归属，不创建嵌套阶段。
本次 C++ 标记更新需要重新编译 `resnet_cifar`、`cnn_cifar`，并重新采集旧融合结果。
之后切换 on/off 不再需要编译。缺少完整阶段标签的旧 fused trace 会报错，不回退逐 kernel 统计。

## 文件

统一入口写在 `nvtx-analysis/`，批量脚本写在 `MODEL/profiled/analysis/`：

- `ks_segments.csv`：每段的 image、layer、subpath、ks_id、lt_id、baby_id、kind、segment、
  category（keyswitch / hybrid）、logical_ks、gpu_start_ns、gpu_end_ns、duration_ms、kernel_sum_ms。
- `ks_units.csv`：按 `(image, ks_id)` 合并，保留各消费者 lt_ids、segment_count、逻辑 KS 和执行调用数。
- `ks_layers.csv`：每图每层的调用数及各 kind 耗时，包括 shortcut、Pool、FC，ReLU 内细分
  boot_input/boot_sign、CtS/StC phase 和多项式子路径。
- `gpu_activities.csv.gz`、`breakdown.csv`：原始活动和旧 kernel duration 诊断。
- `summary.json`：方法、计时边界、覆盖检查、均值、预热和每图统计。

时间戳使用 Nsight trace 时间轴，单位 ns；`start_relative_us` / `end_relative_us` 相对该图片
首个 GPU kernel，单位 us。它们来自 GPU 活动，不是 CPU NVTX push/pop 时间或 UTC 时间。
每图的预热标志 `measured=0` 保留，便于查看全部轨迹。

`duration_ms` 是每个连续阶段的 GPU 首尾跨度。单个 unit 的 `duration_ms` 是各段之和；
其 `gpu_end_ns - gpu_start_ns` 可能包含两段之间的 NonKS，不得作为 KS 耗时。
unit 的 `keyswitch_ms` / `hybrid_ms` 分别累加对应类别的跨度，两者之和为 `duration_ms`；
层表也分别输出这两项，逻辑 KS 次数只按 unit 计算一次。
一个融合 Baby batch 覆盖多个逻辑旋转，无法从已有 trace 提取每个内部旋转的独立起止时间。

## 总量与校验

```text
nvtx_keyswitch_ms = 正式图片 KS 阶段跨度的均值
nvtx_hybrid_ms = 正式图片 Hybrid 阶段跨度的均值
profiled_inference_mean_ms = nvtx_keyswitch_ms + nvtx_hybrid_ms + non_keyswitch_residual_ms
```

`fused_baby_inclusive_ms` 是 Baby 准备及关联 KS finish 子项，已包含在 KS 总量内，不可再次相加。
融合路径的 Hybrid finish 单独计入 Hybrid。`nvtx_keyswitch_kernel_diagnostic_ms` /
`nvtx_hybrid_kernel_diagnostic_ms` 是对应分类 kernel 时长之和，与阶段跨度分别报告。
residual 包含独立置换、其他 kernel、host、剩余空闲、输入、校验及运行期编码等，不是纯 NonKS kernel 时间。
只能从同次采集的 Inference 相减，不能从基线相减。

分析器要求每个 KS kernel 有阶段归属，每个声明阶段有 GPU kernel，拒绝跨层、嵌套重复、
阶段重叠、多 kernel stream 或跨度中夹入其他工作。KS 与 Hybrid 均检查覆盖和重叠；
旧二进制缺少标签时明确失败，不能静默退回旧口径。两种拆分模式的计时均不标记为旧 kernel 诊断。

## 导出 KS / NonKS 时间线

`scripts/keyswitch_timeline.py` 读取 **NVTX 的 `ks_segments.csv`**，按 image 排序，
保留 KS / Hybrid 阶段，输出二者的补集 NonKS。使用 segment，不能使用 `ks_units.csv` 的首尾包络。
脚本仅使用 Python 标准库，不运行 GPU，也不修改输入文件。

```bash
python scripts/keyswitch_timeline.py validation/resnet20-ks-spans/nvtx-analysis/ks_segments.csv \
  --activities validation/resnet20-ks-spans/nvtx-analysis/gpu_activities.csv.gz \
  --output validation/resnet20-timeline
```

该命令使用同次采集的首个至最后一个 GPU 活动，摘要明确标记
`scope=gpu_activity_envelope`。它覆盖输入、FHE、校验等已记录活动，但不覆盖包络之外的
Inference 开头和结尾，因此不能直接等同于完整 Inference residual。
`gpu_activities.csv.gz` 由 NVTX 运行入口自动调用 `analyze_gpu_activity.py` 从 `nvtx.sqlite`
导出，包含 kernel、memcpy、memset 的绝对起止时间，脚本可直接读取压缩文件。

如已从同一 trace 时间轴导出完整 Inference 起止边界，改用 `--bounds inference_bounds.csv`：

```csv
image,start_ns,end_ns,measured
0,1000000,2000000,0
1,3000000,4000000,1
```

这是格式示例，实际时间戳必须来自自己的 trace。`--bounds` 和 `--activities` 二选一；
`--bounds` 标记 `scope=inference`，并要求所有 KS 位于对应边界内。`wall.json` 只有时长，
不能将其时长直接加到首个 kernel 时间上来伪造完整 Inference 边界。
当前采集流程尚未自动生成 `inference_bounds.csv`；这是调用者提供准确边界的输入接口。
现有 `cheddar:image:*` 外层范围还覆盖 Inference 计时之外的工作，与 `inference_ms` 边界
不完全一致，不能直接替代同口径的完整 Inference 边界。
两个选项均省略时，仅使用首个 KS 开始至最后一个 KS 结束，标记 `scope=ks_envelope`，
不包含该包络之外的 NonKS。所有模式均保留 warmup 行，均值只使用 `measured=1` 的图片。

输出目录包含：

- `timeline.csv`：按时间顺序列出 KS / Hybrid / NonKS，含 image、measured、scope、interval_id、
  start_ns、end_ns、duration_ns/ms、相对 T0 的 us，以及原始 segment_ids、ks_ids、layers、
  subpaths、kinds、segments。重叠 KS 合并并保留来源，连续但不重叠的阶段保留独立 ID。
- `timeline.txt`：`T0 → NonKS_0 → [KS_0] → NonKS_1 → … → T1` 的逐行展开，
  每行显示绝对起止时间戳、相对 T0 时间和时长；每图尾部及文件末尾给出汇总。
- `summary.json`：每图 T0/T1、窗口/KS/Hybrid/NonKS 时间、区间数量、合并重叠量，
  正式图片总量与均值，以及 `accounting_error_ns`。T0/T1 为左闭右开边界 `[T0,T1)`。

所有 `start_from_t0_us` / `end_from_t0_us` 都以本文件的 T0 为原点，
与原 `ks_segments.csv` 以首个 kernel 为原点的相对时间可能不同。绝对 ns 不变。
每图必须满足 `KS + Hybrid + NonKS = T1 - T0`，误差为 0 ns；NonKS 包含其区间内的计算、
内存操作和空档，不是纯 kernel 时间。KS / Hybrid 段内部的间隔分别属于该段，不能重复归入 NonKS。

融合路径使用完全相同的导出命令，输入其带 category 的阶段 CSV：

```bash
python3 scripts/keyswitch_timeline.py validation/nvtx-all-fused/resnet20/profiled/analysis/ks_segments.csv \
  --activities validation/nvtx-all-fused/resnet20/profiled/analysis/gpu_activities.csv.gz \
  --output validation/resnet20-fused-timeline
```

两种模式均输出 `method=ks_interval_complement_v1`，`timing_diagnostic_only=false`。
`segment_ids` 为 C++ 声明的阶段 ID，区间数量不是逻辑 KS 次数。
旧 split CSV 无 category 时按 keyswitch 兼容读取；缺少阶段 CSV 的旧融合结果无法仅由活动表恢复。
`summarize_nvtx.py` 自动使用相同时间线入口，详见 README。

## Event 与 NVTX 独立交叉验证

以相同权重、参数、预编码模式、图片及已保存的密钥分别运行。两次都关闭 CPU 操作同步计时：

```bash
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" \
  --encode-mode pre --key-mode load --key-dir /data/keys/resnet20 \
  --split-hybrid on --count 6 --warmup 1 --time-cuda-event on \
  --timing-dir validation/ks-event
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" \
  --encode-mode pre --key-mode load --key-dir /data/keys/resnet20 \
  --split-hybrid on --count 6 --warmup 1 --time-nvtx on --nsys /path/to/nsys \
  --timing-dir validation/ks-nvtx
python scripts/compare_ks_timings.py --event-dir validation/ks-event \
  --nvtx-dir validation/ks-nvtx --output validation/ks-comparison
```

Event 模式每图先记录一个 origin event，每个 KS 阶段再记录一对 event；op/source/phase
标签只跟踪归属，不额外插入旧的逐原语事件。事件预先创建，每图末尾统一读回，没有逐 KS 同步。
每图 `events = 2 × segments + 1`，用 `CUDA_EVENT_KS_IMAGE` 审计。
Hybrid 阶段同样每段一对 event。推理入口预创建 131,072 个事件，覆盖深层 ResNet
未拆分路径增加的混合阶段；溢出明确报错。上面的两次运行也可以同时替换为
`--split-hybrid off`，比较脚本要求两次的拆分模式一致。
旧逐原语 Event CSV 仍可离线分析，不能与新版阶段统计混为同一口径。

`events.csv` 保存原始事件编号、相对 origin 的 ms 及 JSON metadata；
`event-analysis/ks_segments.csv` 展开与 NVTX 相同的 image/layer/subpath/ks_id/segment_id/
kind/segment/logical_ks/lt_id/baby_id/includes_permutation，并保存 `duration_ms`。
Event 相对时间起点是 BeginImage 的 event，NVTX 相对时间起点是该图首个 GPU kernel；
两次独立运行不能按绝对/相对时间戳直接匹配，应按 `(image, segment_id)` 匹配并核对所有归属字段。

两者使用相同应用代码边界，但 GPU 端点的定义有差异：Event 包含开始 event 到首个 kernel、
末个 kernel 到结束 event 的边缘间隙，NVTX 主口径截在首末 kernel 上；独立运行还有 profiler/
event marker 开销与调度差异。FusedBaby 在两者中都保持现有融合实现。
新 Event 摘要与 NVTX 一样，以同次完整 Inference 减去 KS 和 Hybrid 得到 residual，不能与旧 Event 的 FHE total 混用。

比较输出 `segments.csv`、`layers.csv`、`kinds.csv` 和 `summary.json`，保留 category，分别汇总 KS 与 Hybrid，报告逐段差值、
分层/分类型均值、总体差值和逐段绝对差的中位数/P95。PASS 验证元数据一致、覆盖、密钥/
配置与数值门禁，以及汇总闭合；不自动宣称某个时间误差阈值已通过。
