# 推理配置、计时与验证细节

普通使用流程见[README](../README.md)。本页用于调整配置、解释计时和运行额外验证。

## 模型与数据

仅接受支持的 BN 已折叠 CIFAR-10 推理图。导出器验证形状、节点命名或算子清单，
不支持 MaxPool、ReLU6、未折叠 BN、非 2 的幂空间尺寸及多密文特征图。
这些 CNN 使用 AveragePool；AlexNet 为 CIFAR 版本，含 1024→4096→4096→10 分类头。

CIFAR 文件为原始 `test_batch.bin`，每条记录 1 字节标签加 3072 字节 CHW 像素。
完整测试集为 10000 条、30730000 字节。程序使用权重中保存的归一化参数，
ONNX 导出默认 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]。
更换 checkpoint 应核对预处理约定。

普通入口按模型选择范围：ResNet20/32/44 为 10，ResNet56 为 12，ResNet110 为 24，
其他 CNN 为 10。它不会运行时裁剪或自动扩大范围；任意支持的权重仍需独立校验。
范围会影响多项式近似误差。

## 构建

CMake >=3.24。README 示例使用 RMM branch-22.12 头文件和系统 spdlog，
需传入 `-DCHEDDAR_RMM_INCLUDE_DIR=/实际路径/include`。
CUDA 开发工具包必须支持目标 GPU；Blackwell 示例使用 CUDA 12.8 和架构 120。
其他常见架构：RTX 4090 为 89，RTX 3090 为 86，H100/H200 为 90，A100 为 80。

可执行文件与 `libcheddar.so` 必须来自同一构建目录。外层入口自动设置对应
`LD_LIBRARY_PATH`。自定义构建目录使用 `--build-dir /path/to/build`。
不要在构建期间修改源码；怀疑增量产物不一致时执行：

```bash
cmake --build build --clean-first --parallel 12
```

## 编码与调优

默认保留当前分支的 N17 优化内核和已验证配置：

| 配置 | 默认 |
|---|---|
| CNN 环维度 / 槽数 | N=131072 / 65536 |
| ResNet 环维度 / Boot 槽数 | N=65536 / 16384；特征槽数随层变化 |
| 后端 / normal、Boot scale 目标 | uint32 / 40、58 |
| CNN 卷积 / FC | 重复输入分组、共享收集掩码、相邻分组旋转累加 |
| CNN Pool / GAP | 空间 Trace 后收集；消耗一层 |
| CNN / ResNet Boot 可用输出 | level 9 / level 8；ReLU 分别返回 level 3 / level 2 |
| ResNet 卷积 | AE 式重复槽分组、通道 Trace、输出掩码；level 2→0 |
| ResNet baby step | 仅用于池化/FC，默认 16 |
| 编码方式 | auto |

`--encode-mode auto` 在模型旋转密钥和 Boot 资源备齐后，估算整网 CKKS 权重缓存，
包含 RMM bin 向上取整；预算取 CUDA 空闲容量加 RMM CUDA 池未使用容量，预留 4 GiB。
可复用的固定块空闲量没有全部计入，选择偏保守。实际分配失败仍会明确报错。
`pre` 强制整网驻留；`stream` 在每层/输出分组批次求值前实时构造、编码并释放。
预编码后同进程多图复用权重，所有整网模式复用提前准备的实际旋转密钥集合。
不生成所有可能旋转距离的 key；推理阶段缺少 key 会失败。

需要固定编码方式时：

```bash
python scripts/run_inference.py --model vgg11 --cifar /data/test_batch.bin --encode-mode stream
```

CNN 的空间分组由张量布局自动确定，实时编码按固定的权重条数预算划分输出分组。
旧 CNN 的 `--channel-tile`、`--diagonal-chunk`、`--baby-step` 已移除，避免接受无效配置。
CKKS 展开权重大小远大于原始 ONNX；不能由 ONNX 文件大小判断是否能整网预编码。
`resident_gpu_bytes` 含 RMM 保留空间，只是输出时快照，不是峰值或最低显存要求。
各模型预编码缓存、旋转 key 数量及实时编码工作集见[CNN 分解记录](cnn-decomposition.md)；
[此前静态统计](model-memory-static.md)中的其他 CNN 数字对应旧实现。

## 统一计时

计时 schema_version=2。C++ 每图测量、外层入口、两个验证脚本共用同一套定义。
完整运行 TIMING 仅由外层入口发布，C++ 不再另外发布 main 边界的总时间。
JSON 中 `RESULT` 只含正确性与资源数据；`IMAGE_TIMING` 为每张图，
`TIMING` 为整次运行。所有字段以 ms 表示，所有父项均由互斥子项汇总。
`PREPARED_MEMORY` 在全部准备完成、第一张输入处理前记录 CUDA 显存占用快照，
包含内存池保留空间；它是资源记录，不增加另一套计时。
不提供第二套算子 online 时间；重复采样由 `run_inference.py --count N --warmup W` 完成。

| 字段 | 定义 |
|---|---|
| model_prepare_ms | 加载、上下文及其固定表/常量、基础/模型密钥、权重规划与预编码，加其余进程启动/收尾开销 |
| boot_prepare_ms | Boot 切换/旋转 key、CtS/StC 明文、EvalMod、ReLU 常量准备 |
| inference_ms | 所有图片的输入开始到检查和结果输出完成，包括每图局部资源释放 |
| total_ms | 三段之和 |
| input_ms | 图片读取、归一化、打包、编码加密 |
| runtime_weight_prepare_ms | 推理中的权重矩阵构造与编码；整网预编码时为 0 |
| fhe_eval_ms | linear_ms + bootstrap_ms + activation_ms + graph_ops_ms |
| linear_ms | Conv、Depthwise Conv、Pool、FC 的完整密文求值，包括内部旋转、Rescale 等 |
| bootstrap_ms | 实际 Boot 执行，含内部 CtS、EvalMod、StC |
| activation_ms | Boot 之外的激活多项式及相应 Add/LevelDown/乘法 |
| graph_ops_ms | 其他密文操作，如残差 Copy/Add、分支对齐/Concat |
| reference_ms | 两条明文参考路径的计算 |
| validation_output_ms | 逐层/最终解密解码、正确性检查与结果打印 |
| dispatch_ms | inference_ms 减去全部已分类叶区间，含调度、管理和计时开销 |

必须满足：

```text
total_ms = model_prepare_ms + boot_prepare_ms + inference_ms
inference_ms = input_ms + runtime_weight_prepare_ms + fhe_eval_ms
             + reference_ms + validation_output_ms + dispatch_ms
fhe_eval_ms = linear_ms + bootstrap_ms + activation_ms + graph_ops_ms
```

C++ 使用 steady_clock 纳秒计时，累加后转为 ms。GPU 叶区段入口/出口同步，
等待时间包含在所属区段内，父项不重复计时；发现重叠或负余量立即报错。
这会引入同步及测量开销，数值不是纯 kernel 时间。

模型和 Boot 准备可以被权重预编码等步骤穿插；第一项是完整墙钟扣除第二、三项，
并非一个连续区间。最终时间摘要本身的打印属于外层收尾，无法包含自己的完整耗时。
Linux 外层使用进程出生时间，覆盖 Python 启动、子进程动态库加载及退出，
与外部 time 允许约一个系统 tick、显示舍入及最后打印/退出的微小差额。
非 Linux 使用 Python main 起点，`boundary` 会明确标注。
直接调用 C++ 仅输出结果和 JSON IMAGE_TIMING，不发布另一套总时间。
C++ 的内部计时文件用于外层汇总，不能作为独立进程端到端结果。

预热包含在 count 内，也计入三段总时间。statistics 中所有 mean_ms 使用同一组正式图片，
所以均值保持加法关系；仅 inference_ms、fhe_eval_ms 另外提供中位数，它们不用于拼接总量。
不同编码模式、硬件、日志级别、参数、图片集合应分别记录。

## 可选的操作分类与离线密钥

`--ks-profile on` 按相同的完整 `inference_ms` 父区间报告 Keyswitch、Non-keyswitch、Hybrid；
`--split-hybrid on` 切换到诊断执行，配合采集将 Hybrid 拆为两类。二者默认关闭，
开启采集会同步、拆分会增加执行与临时空间开销，详见[边界和实测](keyswitch-profiling.md)。
这些分类与原阶段子项是并列视角，不是额外相加的一段时间。

`--key-mode save|load --key-dir DIR` 支持预先保存、准备阶段加载全部实际密钥，
默认 `generate`。加载过程计入现有准备阶段，详见[KMS 约定](kms.md)。

## 日志和调试

```bash
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin --log-format json
python scripts/run_inference.py --model vgg11 --cifar /data/test_batch.bin --verbose 1
```

底层 C++ 参数会透传，普通入口已自动选择模型和现有默认范围。
专家可用 `--binary build/example/resnet_cifar` 或 `cnn_cifar` 调用探针/局部模式。
直接指定 binary 时不会根据权重自动调整范围。

```bash
python scripts/run_inference.py --binary build/example/resnet_cifar --mode boot
python scripts/run_inference.py --model vgg11 --cifar /data/test_batch.bin --stop-after 1
python scripts/run_inference.py --model vgg11 --cifar /data/test_batch.bin --mode calibrate --log-format json
```

reference/calibrate 为明文路径，不能作为密文性能结果。probe 的 IMAGE_TIMING 使用 image=-1，
不支持 warmup。ResNet 局部模式用于调试，整网预编码仅在 full 模式启用。

## 额外验证

CPU 可运行计时组件及入口协议测试，不需要 CUDA：

```bash
c++ -std=c++17 -DNDEBUG -Wall -Wextra tests/inference_run_test.cpp -o /tmp/inference_run_test
/tmp/inference_run_test
python tests/inference_launcher_test.py /tmp/inference_run_test
```

GPU 基础检查：

```bash
build/example/parameter_literal_test
build/example/resnet_conv_plan_test
build/example/resnet_ae_test
build/example/cnn_linear_plan_test
build/example/cnn_decomposition_test
build/example/cnn_decomposition_test 0  # 独立进程验证 Boot9 -> ReLU level 3
build/example/cnn_primitives_test 32 16 16384 linear
build/example/cnn_primitives_test 32 17 65536 relu
```

分阶段 ResNet 验证保留编码、Boot、ReLU、残差和整网检查：

```bash
python scripts/validate_resnet.py --binary build/example/resnet_cifar \
  --models models --cifar /data/test_batch.bin --output validation/resnet-check --count 3
```

CNN 的独立参考额外需要 PyTorch。用自己的 ONNX/CIFAR 逐模型生成参考，例如：

```bash
python scripts/cnn_float_reference.py --source /data/onnx/vgg11_cifar10.onnx \
  --cifar /data/test_batch.bin --output validation/reference/float-vgg11.json --count 3
python scripts/validate_cnn.py --binary build/example/cnn_cifar --models models \
  --cifar /data/test_batch.bin --reference-dir validation/reference \
  --output validation/cnn-check --families vgg11 --count 3
```

扩展到其他 family 前，为它们分别生成 float-FAMILY.json。参考源模型和数据哈希必须一致。
验证脚本保留源码、模型、二进制/共享库、参数和数据证据，并在失败时退出。
数值门槛仍为逐层所有配置槽、原始实数单位下实部/虚部误差 <=1e-2，
明文/FHE 的最终 logits 和 top-1 单独输出；解密结果不反馈到密文图。

新计时已通过 CPU 计时及跨进程协议测试；AE 式卷积改动后的 ResNet20 已在
RTX 5090 上按此口径完成 GPU 验证，见[验证记录](resnet-ae-validation.md)。
全部十个模型随后已在 96 GB GPU 上以本地真实 ONNX 权重完成整网预编码测评，
每模型 11 张图、1 张预热，见[96 GB 实测报告](gpu96-benchmark.md)。

网络实现细节：[ResNet](../example/README.resnet.md)、[其他 CNN](../example/README.cnn.md)。
