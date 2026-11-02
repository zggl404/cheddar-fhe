# 执行计时与全模型验证

Docker 环境中直接从项目根目录执行。输入是原始 CIFAR-10 二进制测试文件，以及
`models/MODEL.bin`。模型名：`resnet20 resnet32 resnet44 resnet56 resnet110 vgg11 vgg16 mobilenet squeezenet alexnet`。

## 1. 编译

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native \
  -DBUILD_UNITTEST=OFF -DBUILD_RESNET_EXAMPLE=ON -DBUILD_CNN_EXAMPLE=ON \
  -DCHEDDAR_PLAINTEXT_REFERENCE=OFF \
  -DCHEDDAR_NVTX_TRACE=ON -DCHEDDAR_CUDA_EVENT_TRACE=ON
cmake --build build --parallel 12
export CIFAR_BIN="$PWD/data/cifar-10-batches-bin/test_batch.bin"
```

`CHEDDAR_PLAINTEXT_REFERENCE=OFF` 关闭明文网络参考计算；计时开关不改变这个编译选项。
需要逐层参考数值回归时另用 ON 构建。GPU 架构也可以显式指定，例如 RTX 5090 使用 120。
Nsight Systems CLI 的 `nsys` 需要位于 PATH；否则在后续命令中增加 `--nsys /absolute/path/to/nsys`。

## 2. 一条命令验证全部模型

```bash
python scripts/benchmark_timing.py \
  --build-dir build --models-dir models --cifar "$CIFAR_BIN" \
  --output validation/timing-all --count 6 --warmup 1 --encode-mode auto
```

默认顺序执行十个模型。每个模型先生成并保存临时密钥，然后使用同一份密钥独立运行：

1. 三个计时开关全部 OFF 的粗粒度墙钟基线。
2. Event-only。
3. NVTX-only。
4. CPU 同步诊断。

统一使用 `split-hybrid on` 和 strict 数值门禁，检查所有 load 运行的 `keys_generated=0`。
auto 先在基线解析为 pre 或 stream，再固定给其余运行。模型完成或失败后清理脚本创建的临时密钥；
单模型失败会记录日志并继续其他模型，最终只在全部模型通过时返回 0。
输出目录必须不存在，避免覆盖已有证据。

只跑指定模型：

```bash
python scripts/benchmark_timing.py \
  --build-dir build --models-dir models --cifar "$CIFAR_BIN" \
  --output validation/timing-resnets --models resnet20 resnet110 \
  --count 6 --warmup 1 --encode-mode auto
```

`count=6` 包含 1 张预热，正式均值取后 5 张。端到端和 Inference 总量仍覆盖全部 6 张。
strict 失败不能视为正确性通过；这里的少量样本验证不代表完整测试集准确率。

## 3. 单独执行一种计时

下面用 ResNet20、pre 编码举例。先保存密钥，密钥准备不作为正式 load 测量：

```bash
mkdir -p validation/keys
build/example/resnet_cifar \
  --weights models/resnet20.bin --cifar "$CIFAR_BIN" --count 1 \
  --encode-mode pre --split-hybrid on \
  --key-mode save --key-dir validation/keys/resnet20
```

CPU 同步诊断：

```bash
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" \
  --encode-mode pre --key-mode load --key-dir validation/keys/resnet20 \
  --split-hybrid on --count 6 --warmup 1 --time-cpu on \
  --timing-dir validation/timing-resnet20-cpu
```

CUDA Event：

```bash
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" \
  --encode-mode pre --key-mode load --key-dir validation/keys/resnet20 \
  --split-hybrid on --count 6 --warmup 1 --time-cuda-event on \
  --timing-dir validation/timing-resnet20-event
```

NVTX：

```bash
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" \
  --encode-mode pre --key-mode load --key-dir validation/keys/resnet20 \
  --split-hybrid on --count 6 --warmup 1 --time-nvtx on \
  --timing-dir validation/timing-resnet20-nvtx
```

未指定的计时开关均为 OFF。比较时应独立运行，不要同时开启多种计时。
其他模型替换 `--model` 和密钥目录；大模型使用 stream 或先由批量脚本决定编码模式。
可以用 `--weights` 显式指定权重；否则从 `models/MODEL.bin` 读取。

不计时时三个开关均省略，只输出参数、Logits 和 Success：

```bash
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" \
  --encode-mode pre --key-mode load --key-dir validation/keys/resnet20 --split-hybrid on
```

如需无细粒度插桩的端到端、key load 和预编码数据，使用第 2 节批量脚本的 baseline 输出。

## 4. Event/NVTX 逐阶段交叉验证

```bash
python scripts/compare_ks_timings.py \
  --event-dir validation/timing-resnet20-event \
  --nvtx-dir validation/timing-resnet20-nvtx \
  --output validation/timing-resnet20-comparison
```

要求同一密钥、权重、图片、编码模式及独立运行。按 `(image, segment_id)` 对齐并核对全部归属字段。
Event 在阶段进入/退出记录事件，NVTX 使用该阶段的 GPU 首末 kernel；边缘间隙与采集扰动会产生差值。
PASS 表示覆盖、归属、数值门禁和汇总一致，不表示两种测量的每个数值相等。

## 5. 输出重点

批量根目录 `summary.csv` / `summary.json` 包含每个模型的状态、编码模式、基线准备/推理时间、
Event/NVTX KS 均值和差值，以及 CPU 诊断。每模型子目录：

| 位置 | 用途 |
|---|---|
| `baseline/wall.json`、`time.txt` | 无细粒度插桩的阶段墙钟，以及 GNU time real/user/sys |
| `event/event-analysis/ks_segments.csv` | Event 的逐阶段 ID、相对起止时间和耗时 |
| `nvtx/nvtx-analysis/ks_segments.csv` | NVTX 的逐阶段 GPU 起止时间和耗时 |
| `nvtx/nvtx-analysis/ks_units.csv` | 按调用/共享批次合并，统计逻辑 KS 与执行调用数 |
| `nvtx/nvtx-analysis/ks_layers.csv` | 按图片、网络层汇总 |
| `comparison/segments.csv`、`layers.csv`、`kinds.csv` | 两种方法的逐段、逐层、逐类差值 |
| `cpu/summary.json` | 同步 CPU 诊断，分类父区间为 FHE |

计时均为 ms，起止时间列名另有标明。新 Event/NVTX 报告使用同次完整 Inference 为父区间：

```text
Inference_mean = KeyswitchTime + NonKeyswitchTime
end_to_end = key_load + weight_preencode + inference_total + other_prepare_and_teardown
```

NonKeyswitchTime 是剩余墙钟，包含输入、主机工作、校验、运行期编码及剩余等待。
CPU 的分类总量是 FHE，不与 Event/NVTX 的完整 Inference 分类直接混加。
共享调用包含多个离散阶段时，应累加阶段 `duration_ms`，不能用最早开始到最晚结束代替。
逻辑次数在 `ks_units.csv` 上按调用统计，不能在多个共享阶段上重复累加。

完整字段及边界见 [KS 阶段说明](keyswitch-spans.md) 和 [计时实现说明](timing-modes.md)。
十个模型的硬件、验证范围和实测结果见 [全模型验证报告](timing-validation.md)。
