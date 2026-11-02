# 三种独立计时开关

统一入口是 `scripts/run_inference.py`，ResNet/CNN 使用相同开关，默认全部 `off`：

| 开关 | 方法 | 输出 |
|---|---|---|
| `--time-cpu on` | `steady_clock` + 同步的独占操作区间 | 端到端、Prepare、Inference，以及 FHE 三分类 |
| `--time-cuda-event on --split-hybrid on` | 默认 CUDA stream 上的 KS 阶段 event 对 | 与 NVTX 共用阶段 ID、KS 时间及同次 Inference 剩余墙钟 |
| `--time-nvtx on --split-hybrid on` | NVTX + Nsight Systems 的 CUDA launch correlation | 逐层 KS GPU 阶段起止时间、跨度及同次 Inference 剩余墙钟 |

`--ks-profile` 保留为 `--time-cpu` 的兼容别名。命令行开关覆盖继承的 tracing 环境变量；
例如三个开关全 off 时，即使 shell 中存在 `CHEDDAR_GPU_TRACE=1` 也不会开启标记。
三个开关可以组合。比较方法差距时应分别运行；组合运行会相互引入采集和同步开销，
输出的 `enabled_methods` 明确记录组合，不将组合运行称为无插桩基线。

先编译一次所需功能，运行时再切换：

```bash
cmake -S . -B build -DBUILD_RESNET_EXAMPLE=ON -DBUILD_CNN_EXAMPLE=ON \
  -DCHEDDAR_NVTX_TRACE=ON -DCHEDDAR_CUDA_EVENT_TRACE=ON
cmake --build build --parallel 12 --target resnet_cifar cnn_cifar
```

CMake 的两个 GPU tracing 选项仍默认 OFF。请求未编译的功能会明确报错。
明文参考仍由独立的 `CHEDDAR_PLAINTEXT_REFERENCE=ON|OFF` 决定；三个计时开关不改变它。

```bash
# 全 OFF：只输出参数、Logits 和 Success。
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin

# CPU 方式：没有 NVTX 或 CUDA Event 插桩。
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --time-cpu on --count 3 --warmup 1

# CUDA Event 方式：CPU 分类和 NVTX 关闭。
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --time-cuda-event on --split-hybrid on --count 6 --warmup 1

# NVTX 方式：launcher 自动运行 nsys profile、导出 SQLite 并计算分类。
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --time-nvtx on --split-hybrid on --nsys /path/to/nsys --count 6 --warmup 1 \
  --timing-dir validation/resnet20-ks-spans
```

可追加 `--encode-mode pre --key-mode load --key-dir /data/keys/resnet20` 固定编码/密钥模式。
`--timing-dir /new/path` 保留 worker log、CPU 原始报告、Event CSV、Nsight SQLite、分类明细和摘要；
目录必须不存在。未指定时在临时目录分析，结束后清理。`--log-format json` 输出对应的
`CPU_TIMING`、`CUDA_EVENT_TIMING`、`NVTX_TIMING`，关闭的方法没有对应记录。

## 共同的分类与边界

CPU 及旧 Event（split-hybrid off）的三个分类只覆盖 FHE 求值，即 linear + bootstrap + activation + graph operations；
不含图片输入/加密、实时权重编码、网络明文参考、逐层解密校验及结果打印。
这两种 FHE 分类按同一组正式图片求均值，单位为 ms：

```text
该方法的 FHE total = KeyswitchTime + NonKeyswitchTime + Hybrid
```

| 分类 | 操作 |
|---|---|
| KeyswitchTime | ModUp、PseudoModUp、evaluation-key 乘加、必要 ciphertext correction、fused_baby、独立 ModDown、StD |
| Hybrid | 共享 lift、ModDown+Rescale、置换+累加等不能独占归属的区间 |
| NonKeyswitchTime | FHE 边界内剩余工作，包括独立置换、拆分后的共同短基前置 |

`fused_baby` 按完整准备阶段记录，在拆分模式中也保留现有融合。
giant-step 明文乘加仍归 NonKeyswitch，保留自身融合。
`eacaa59` 版本曾保守地把 fused_baby 放入 Hybrid，旧分类数值不能与当前分类直接比较。

`--split-hybrid on` 是改变执行路径的诊断开关，不是第四种计时方式：

- ModDown+Rescale 拆为独立 ModDown（KS）及 Rescale（NonKS）。
- HRotAdd/HConjAdd 拆为独立置换（NonKS）和额外加法（NonKS）。
- giant-step 先完成每次旋转的 b 项修正（KS）和独立置换（NonKS），再累加不同 giant-step（NonKS）。
- SharedLift 拆为共同短基 INTT/归一化（NonKS）、Q 扩展/NTT（NonKS）、P 扩展/NTT（KS）。

拆分后 Hybrid 必须为零。共同前置只执行、记录一次，统一归 NonKeyswitch；
保留 `shared_lift_common` 子项供审计。C++ 和两个 GPU 分析器都会拒绝残留的混合操作。
拆分增加临时数据读写/launch，联合降模与顺序降模还可能有舍入差异；
拆分时间用于诊断，不能反推原融合 kernel 内部的唯一耗时比例。

## 三种数值各自代表什么

CPU 使用 `steady_clock`，在 GPU 阶段和操作归属切换处同步，嵌套范围暂停父范围后恢复，
因此同一时间不会重复计入两个分类。`NonKeyswitchTime = FHE wall - KeyswitchTime - Hybrid`。
它包含主机提交、等待、分配和同步开销；CPU 开启本身会扰动执行，不等于无扰动 kernel 时间。
只开 CPU 时没有 NVTX/CUDA Event；三个开关全 OFF 时不记录 CPU 叶区间，也不执行这些计时同步。

CPU 还报告：

```text
end_to_end_ms = prepare_ms + inference_ms
prepare_ms = model_prepare_ms + boot_prepare_ms
```

这些总量覆盖全部图片（含预热），分类的 `total` 是正式图片的 FHE 均值。
模型准备项包含 Context、secrets/基础/模型 key、权重预编码及其余进程开销；
Boot 准备含 Boot key、EvalMod、CtS/StC 明文等。Linux 使用 launcher 进程出生时间，
非 Linux 从 Python main 起计，`boundary` 明确标注。最终固定摘要打印在边界之外。
同时开 NVTX 时外层总时间还包含 profiler 启停/导出，不能称为基线端到端。

CUDA Event 在 phase/source/part/op 所有者变化时向 legacy default stream 插入 event，
用相邻边界形成互斥区间；每图完成后统一 synchronize/readback，不逐段同步。
它包含区间内的 kernel、内存操作、设备空闲和 event 开销，不是纯 kernel busy time。
当前实现限单 host thread、legacy default stream；换成多流前须扩展测量边界。
关闭此功能只取消计时用 event，编码器及内存分配器用于生命周期/流同步的 disable-timing event 仍保留。

上述逐原语 Event 行为保留在 `--split-hybrid off`。开启 split 后，Event 复用 NVTX 的
`KsTraceSegment` 进入/退出点和 segment_id，每段一对 event，每图统一读回。此时公开分类的
父区间是同次完整 Inference，NonKS 是 Inference 减去 KS 的剩余墙钟，与旧 FHE total 不同。
边界端点差异和逐段独立比较命令见 [Event/NVTX 交叉验证](keyswitch-spans.md#event-与-nvtx-独立交叉验证)。

NVTX 本身只有标记，不提供 GPU 耗时。launcher 调用 Nsight Systems，把 CUDA runtime 的
correlation ID 与 GPU 活动关联，即使 kernel 在 CPU range 退出后执行也不会错归属。
新主口径要求 `--split-hybrid on`，按连续 KS 阶段的“最后 kernel 结束 − 首个 kernel 开始”统计，
包含内部内存操作和启动/等待间隙。FusedBaby 的 prepare 与后续 finish 分段求和，权重乘加不跨入跨度。
阶段跨度按实际执行区间归属，包含内部间隙，不是纯 GPU 指令忙碌时间。
`NVTX_TIMING.NonKeyswitchTime = 同次 Inference 均值 − KS 阶段跨度均值`，因此其 total 是
Inference 均值，包含当前 Inference 边界内输入、校验等工作，与 CPU/旧 Event 的 FHE total 不同。
旧 kernel duration sum 仍保留为诊断列。`--split-hybrid off` 仅输出旧融合路径的 kernel 诊断，
不发布新的 KS 跨度或 Inference 剩余值。单 stream、launch 覆盖、阶段闭合及重叠均经过检查。
逐层起止时间与共享关系见 [KS 阶段报告](keyswitch-spans.md)。

## 输出与兼容性

全 OFF 的 JSON 仅有 `RUN_PARAM`、`PARAM` 和每图 `RESULT`（含 `fhe_logits`、`success`）；
human 输出为参数、Logits、`Success: true`。失败仍输出错误并返回非零。
Success 表示当前构建启用的数值/结构检查通过，不表示预测一定等于数据标签。
明文参考关闭时，不声称与明文参考一致。

`--numerical-gate strict|warn` 默认 strict。warn 保留 `0.01` 数值门槛，但将有限数值超标及最终参考
top-1 不一致记录为 `NUMERICAL_WARNING`，继续运行；结果标记 `NUMERICAL_FAILED`、`success=false`。
计时摘要同时标记 `numerical_passed=false`、`diagnostic_only=true`，只供诊断。
NaN/Inf、CUDA、结构、密钥及其他非数值门槛错误仍中止。完整运行不代表正确性通过。

开启 CPU 并加 `--verbose 1` 时保留旧 schema v2 的 `IMAGE_TIMING`、`TIMING` 审计记录，
其中旧的扁平 `non_keyswitch_ms` 仍以整个 Inference 为父区间；新公开三分类位于 `CPU_TIMING`，
以 FHE 为父区间。两套范围不能混加。现有 GPU 验证脚本已经显式开启 CPU/verbose。
直接运行 C++ 可得到 CPU 内部进程边界和 Event 原始记录；NVTX 聚合需要进程退出后的 profiler 数据，
因此完整的三种报告请通过 Python 统一入口获取。
