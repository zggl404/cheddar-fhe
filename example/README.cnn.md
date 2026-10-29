# 手写 CIFAR-10 CNN 与 N=2^17

`cnn_cifar` 使用 Cheddar C++ 算子执行已检查的 BN 折叠 CIFAR-10 网络。
Python 只导出权重和有限算子清单；C++ 运行时不依赖 Python、ONNX 或 FHE 编译器。

当前导出配置对应：

| 配置 | Conv | ReLU | Pool | 分支 |
|---|---:|---:|---|---|
| vgg11 | 8 | 8 | 5 AveragePool | 无 |
| vgg16 | 13 | 13 | 5 AveragePool | 无 |
| mobilenet | 27，含 13 depthwise | 27 | 1 AveragePool | 无 |
| squeezenet | 26 | 18 | 4 AveragePool + 1 GAP | 8 Concat |
| alexnet | 5 | 7 | 3 AveragePool | 无，另含 3 个 FC |

这些是固定的 CIFAR 模型，不能用同名 torchvision ImageNet 模型直接替换。
导出器检查算子清单和属性，拒绝未折叠的 BN 及未支持的算子。
MaxPool、ReLU6、非 2 的幂空间尺寸和多密文特征图属于后续扩展范围。
这里的 AlexNet 是 3×3 卷积、平均池化和 1024→4096→4096→10 分类头的 CIFAR 版本。

## 构建

```sh
cmake -S . -B build -DBUILD_UNITTEST=OFF \
  -DBUILD_RESNET_EXAMPLE=ON -DBUILD_CNN_EXAMPLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel 12
```

架构 `120` 对应本次 RTX 5090。使用已有 RMM 22.12 头文件和系统 spdlog 时可增加：

```sh
-DCHEDDAR_RMM_INCLUDE_DIR=/path/to/rmm/include
```

跨机器增量同步应在构建结束后进行。若使用 rsync 保留源时间戳，并曾在构建期间
更改或同步源文件，应完整重建，避免编译器已读取旧源文件而目标文件时间戳更新的情况：

```sh
cmake --build build --clean-first --parallel 12
```

## 基础算术与刷新检查

```sh
build/example/parameter_literal_test
build/example/cnn_primitives_test 32 17 65536 basic
build/example/cnn_primitives_test 32 17 65536 linear
build/example/cnn_primitives_test 32 16 16384 linear
build/example/cnn_primitives_test 32 17 65536 boot
build/example/cnn_primitives_test 32 17 65536 relu
build/example/cnn_primitives_test 32 17 32768 relu
build/example/cnn_primitives_test 32 16 16384 relu
build/example/cnn_primitives_test 64 17 65536 basic
```

`log_degree=17` 表示 N=131072，最多 65536 个 CKKS 槽。参数解析器支持 N=65536
和 N=131072，并针对对应的 2N 生成 NTT 素数；NTT/INTT 增加了 log_degree=17 实例。
基础检查包含全部活跃 Q/P 系数的精确 NTT 往返、全部配置槽的编码加密、旋转、
乘法与 Rescale、LevelDown、Boot 及复合多项式 ReLU。
`linear` 覆盖恒等对角线、包含零 giant-step 的 BSGS，以及不含零 giant-step 的 BSGS。
分块权重会产生后者；共享 Hoist 路径按零步是否实际存在来初始化累加器。

## 模型准备与独立参考

导出需要 `numpy` 和 `onnx`，独立参考另外需要 PyTorch。示例：

```sh
python3 scripts/export_cnn_weights.py /path/to/vgg11_cifar10.onnx \
  models/vgg11.bin --family vgg11

python3 scripts/cnn_float_reference.py --source /path/to/vgg11_cifar10.onnx \
  --cifar /path/to/test_batch.bin --output validation/cnn/float-vgg11.json --count 11

build/example/cnn_cifar --weights models/vgg11.bin --cifar /path/to/test_batch.bin \
  --mode calibrate --count 11
```

其余配置分别使用 `vgg16`、`mobilenet`、`squeezenet`、`alexnet`。
二进制嵌入 ACE CIFAR 预处理的 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]。
同名 JSON manifest 保存源模型和导出文件 SHA-256、张量形状及算子清单。
独立参考直接用 PyTorch CPU float64 重放源 ONNX，保留每张图的 logits、预测和逐层激活范围。

`calibrate` 仅运行精确 ReLU 明文路径；`reference` 另外执行相同多项式的明文路径。
两者均不构造同态计算上下文，也不编码或执行密文运算。校准样本的最大值不构成全数据集范围保证。

## 分阶段密文推理

```sh
# 第一层卷积：输出 64×32×32，恰好占用 65536 个槽。
build/example/cnn_cifar --weights models/vgg11.bin --cifar /path/to/test_batch.bin \
  --stop-after 1

# VGG11 的 Conv-ReLU-Pool-Conv-ReLU。
build/example/cnn_cifar --weights models/vgg11.bin --cifar /path/to/test_batch.bin \
  --stop-after 5

# 完整网络；默认 N=131072、65536 槽、32 位后端、normal/Boot scale 目标 40/58。
build/example/cnn_cifar --weights models/vgg11.bin --cifar /path/to/test_batch.bin \
  --count 3 --relu-range 10
```

激活使用固定的实数归一化范围，默认 10。多项式参考的任意 ReLU 输入超界都会失败，
没有运行时裁剪或自动扩大范围。`--scale-bits 48` 是显式的另一组密码参数配置，
不会自动切换；实际素数、level 配置和 scale 写入 `PARAM` JSON。

空间布局沿用 32×32 画布的通道交错方式，支持宽度 32/16/8/4/2/1；所有算子保留配置槽数。
Flatten 是逻辑视图，FC 直接读取原 CHW 的物理槽映射。Concat 使用两个布局兼容、
通道组对齐的分支，通过旋转后相加完成；两支的 level 在相加前对齐。

密文输入从 level 3 开始。普通/grouped 卷积和 FC 使用重复输入、通道归约、
共享收集掩码消耗两层；stride=1 的 depthwise 卷积只需要空间权重，消耗一层。
stride=2 的 depthwise 另有布局收集，消耗两层。Pool/GAP 先做空间 Trace，再收集，消耗一层。
每个 ReLU 使用两次普通 Boot；CNN Boot 的可用输出为 level 9，ReLU 返回 level 3。
因此支持卷积后直接接池化的 SqueezeNet 分支，没有额外插入 Boot。
每个输出位置的偏置只加一次；分批求值时仅第一批加偏置。

输出仍保持原有单份特征加零填充的布局，Concat 和独立明文参考不变。
只有卷积/FC 的临时输入被重复；掩码在输出分组间复用，使用相邻分组位移累加，
避免为每个分组分别保留旋转 key。AlexNet 的大 FC 同样使用分组和共享掩码。
代码与验证结果见 [CNN 分解记录](../docs/cnn-decomposition.md)。

## 运行、内存和计时

普通运行见[主 README](../README.md)。统一的三段及推理细分计时、
预编码/实时编码和额外验证，
见[详细文档](../docs/inference.md)。所有模型共用 schema_version=2。
