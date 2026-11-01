# 三种独立计时开关

统一入口是 `scripts/run_inference.py`，ResNet/CNN 使用相同开关，默认全部 `off`：

| 开关 | 方法 | 输出 |
|---|---|---|
| `--time-cpu on` | `steady_clock` + 同步的独占操作区间 | 端到端、Prepare、Inference，以及 FHE 三分类 |
| `--time-cuda-event on` | 默认 CUDA stream 上的相邻 event 时间差 | FHE 三分类的 stream elapsed time |
| `--time-nvtx on` | NVTX + Nsight Systems 的 CUDA launch correlation | FHE 三分类的 kernel duration sum |

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
  --time-cuda-event on --count 3 --warmup 1

# NVTX 方式：launcher 自动运行 nsys profile、导出 SQLite 并计算分类。
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --time-nvtx on --nsys /path/to/nsys --count 3 --warmup 1
```

可追加 `--encode-mode pre --key-mode load --key-dir /data/keys/resnet20` 固定编码/密钥模式。
`--timing-dir /new/path` 保留 worker log、CPU 原始报告、Event CSV、Nsight SQLite、分类明细和摘要；
目录必须不存在。未指定时在临时目录分析，结束后清理。`--log-format json` 输出对应的
`CPU_TIMING`、`CUDA_EVENT_TIMING`、`NVTX_TIMING`，关闭的方法没有对应记录。

## 共同的分类与边界

公开的三个分类只覆盖 FHE 求值，即 linear + bootstrap + activation + graph operations；
不含图片输入/加密、实时权重编码、网络明文参考、逐层解密校验及结果打印。
每个方法都按同一组正式图片求均值，单位为 ms：

```text
该方法的 FHE total = KeyswitchTime + NonKeyswitchTime + Hybrid
```

| 分类 | 操作 |
|---|---|
| KeyswitchTime | ModUp、PseudoModUp、evaluation-key 乘加、必要 ciphertext correction、独立 ModDown、独立置换、StD |
| Hybrid | fused_baby、共享 lift、ModDown+Rescale、置换+累加等不能独立拆出的混合区间 |
| NonKeyswitchTime | FHE 边界内剩余工作 |

本次统一将 `fused_baby` 放入 Hybrid；旧 GPU activity 报告把它归入 Keyswitch，
不能直接用旧报告的分类数值与新结果比较。必要 ciphertext correction 现在也进入 CPU profile，
避免 CPU 与 GPU 分析器的范围不一致。共享的 ModUp 只按实际执行记录一次。
`--split-hybrid on` 仍是改变执行路径的诊断开关，不是第四种计时方式。

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

NVTX 本身只有标记，不提供 GPU 耗时。launcher 调用 Nsight Systems，把 CUDA runtime 的
correlation ID 与 GPU kernel 活动关联，按照 launch 时的 NVTX 所有者分类，
即使 kernel 在 CPU range 退出后执行也不会错归属。公开值累加 FHE kernel duration；
不含 memcpy/memset、CPU 提交间隙、设备 idle 或 profiler 后处理。原始分析保留内存活动和 overlap 诊断。
存在并发 kernel 时 duration sum 可能大于时间轴跨度，不能将它当成墙钟。

## 输出与兼容性

全 OFF 的 JSON 仅有 `RUN_PARAM`、`PARAM` 和每图 `RESULT`（含 `fhe_logits`、`success`）；
human 输出为参数、Logits、`Success: true`。失败仍输出错误并返回非零。
Success 表示当前构建启用的数值/结构检查通过，不表示预测一定等于数据标签。
明文参考关闭时，不声称与明文参考一致。

开启 CPU 并加 `--verbose 1` 时保留旧 schema v2 的 `IMAGE_TIMING`、`TIMING` 审计记录，
其中旧的扁平 `non_keyswitch_ms` 仍以整个 Inference 为父区间；新公开三分类位于 `CPU_TIMING`，
以 FHE 为父区间。两套范围不能混加。现有 GPU 验证脚本已经显式开启 CPU/verbose。
直接运行 C++ 可得到 CPU 内部进程边界和 Event 原始记录；NVTX 聚合需要进程退出后的 profiler 数据，
因此完整的三种报告请通过 Python 统一入口获取。
