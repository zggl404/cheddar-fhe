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

密文输入从 level 2 开始。每个线性层消耗一个 level，每个 ReLU 使用两次普通 Boot
并返回 level 2。池化之后的卷积支持从 level 1 到 level 0，再由 ReLU 刷新。
每个输出位置的偏置只加一次。

## 权重内存与计时

较宽卷积按 `--channel-tile`（默认 16 个输出通道）构造 CPU 稀疏映射，
再按 `--diagonal-chunk`（默认 64 条对角线）逐块编码和求值。
Depthwise 和池化直接利用稀疏结构。部分结果在相同输出 level/scale 下累加。
这限制 GPU 同时驻留的权重明文数量，但会增加分块计算和编码次数。
N17 的 RMM bin 按 8 块预分配和增长，避免每个大 bin 默认预留 128 块所造成的显存浪费；
N16 保留原来的分配策略。
`--baby-step` 默认为 128，可显式改变 BSGS 划分。
CPU 内存充足时可用 `--channel-tile 512` 减少不同输出通道块重复编码相同旋转对角线的开销；
GPU 对角线驻留数量仍由 `--diagonal-chunk` 控制。CPU 稀疏映射的内存会相应增加。

每个算子都检查全部配置槽，包括无效槽的零值和虚部残留。
与相同多项式参考的实部误差和虚部残留均必须在原始实数单位下 <=1e-2。
完整网络要求 FHE、多项式参考和精确 ReLU 参考的 top-1 一致。
中间解密仅作校验，不反馈给密文计算。

`weights_ms`、`keys_ms`、`input_ms`、`online_ms`、`checks_ms` 分别记录。
`online_ms` 是分块密文求值的同步墙钟之和，要求在线编码为零；它不是纯 GPU kernel 时间。
`image_wall_ms` 包含该图的权重准备和明文/解密校验，但不包含进程级初始化及 Boot 准备。
`resident_gpu_bytes` 是图结束时的显存占用快照，包含 RMM 预留，不是峰值活跃张量大小或内存流量。
这些是带逐层检查的验证计时，不能直接作为稳定性能排名。

## 批量验证及证据

先为所选模型准备 `models/FAMILY.bin` 和 `validation/cnn/float-FAMILY.json`，然后运行：

```sh
python3 scripts/validate_cnn.py --binary build/example/cnn_cifar --models models \
  --cifar /path/to/test_batch.bin --reference-dir validation/cnn \
  --output validation/cnn/ciphertext --count 3 --channel-tile 512
```

可用 `--families vgg11` 缩小范围。验证器按首层、模块、完整模型顺序执行，首个失败即停止。
报告保留每项命令、日志、返回码、硬件、源文件/模型/数据/二进制/共享库哈希和参数哈希。
参数哈希使用 `json.dumps(sort_keys=True,separators=(',',':'))` 的 SHA-256。
与独立 PyTorch 参考的 C++ 精确 ReLU logits 差异门槛为 1e-7。
当 count>1 时，第一张作为预热，其余在线时间单独求平均；三张图的检查不代表全 CIFAR-10 准确率。

## 2026-10-28 验证记录

RTX 5090（32607 MiB），驱动 580.105.08，CUDA 12.8.93，sm_120。
五个模型统一采用 N=131072、65536 槽、32 位后端、40/58 位 scale 目标、ReLU 范围 10，
通道块 512、GPU 对角线块 64。首层、模块以及每模型前三张图的整网检查均通过。

| 模型 | 整网图数 | 每图 Boot | 与同多项式参考的最大逐层误差 | 图 1、2 的 online_ms 均值 |
|---|---:|---:|---:|---:|
| VGG11 | 3 | 16 | 2.297e-4 | 4022.792 |
| VGG16 | 3 | 26 | 1.417e-4 | 6466.050 |
| MobileNet | 3 | 54 | 6.503e-4 | 5980.972 |
| SqueezeNet | 3 | 36 | 1.032e-3 | 3811.941 |
| AlexNet | 3 | 14 | 3.985e-4 | 5963.789 |

全部 15 次整网推理均满足 FHE、多项式参考、精确 ReLU 参考及独立 PyTorch 的 top-1 一致，
在线编码和 CPU 编码尝试均为零。当前 C++ 精确 ReLU logits 与独立 float64 参考的最大差异为 3.75e-11。
以上时间沿用逐层验证计时边界，权重流式准备和密钥生成另报。

多项式近似误差单独报告：FHE 与精确 ReLU 最终 logits 的最大差异分别为
VGG11 0.1054、VGG16 0.1700、MobileNet 0.1624、SqueezeNet 0.0866、AlexNet 0.0336。
这些值包含原网络非线性被多项式近似产生的差异；上述 1e-2 门槛约束的是与相同多项式参考的密文误差。
本轮样本的 top-1 一致，完整 CIFAR-10 准确率及更小近似误差仍需后续评估。

N17 的精确 NTT 往返、32 位满槽/半槽 Boot/ReLU、64 位基础算术，以及 N16 回归均通过。
原 ResNet20 还完成了一张图的全网回归：38 次 Boot，最大密文误差 8.674e-5，预测与参考一致。

本次接受的报告为 `validation/cnn/ciphertext-attempt04/report.json`；
参数 SHA-256 为 `2b37e7a38cb9d27d782c4fe4d9241b105f56ed78c84fc443f28d34185a5f4c1e`。
该次报告的源码对应初始实现提交 `aaa2b9b`。早期构建、缺失零 giant-step 分块和
显存不足的失败日志一并保留，未计入通过结果。

## N17 NTT online 优化与复测

`NTTLaunchConfig::LogWarpBatching()` 对 N17 的 NTT 第一阶段和 INTT 第二阶段
从 0 改为 4，每个线程块处理 16 组。对应线程数从 32 增至 512，每个素数的
grid.x 从 512 降至 32；32 位后端的每块共享内存从 1 KiB 增至 16 KiB。
蝶形运算、stage merging、素数和缩放配置保持相同，N16 的配置也保持相同。

`NODE_TIMING` 在原有计时区间外输出每个节点的时间差，所有节点的 online 之和
必须与 `RESULT.online_ms` 相符。新增复测脚本使用相同的同步墙钟边界，
单列准备、加密及校验时间，并检查在线编码为零、固定 Boot 次数、全部逐层误差、
FHE/多项式/精确 ReLU/独立 PyTorch 的最终 top-1 和精确 ReLU logits 一致性。

```sh
python3 scripts/benchmark_cnn_online.py --binary build/example/cnn_cifar \
  --models models --cifar /path/to/test_batch.bin --reference-dir validation/cnn \
  --output validation/online/run-01 --label warp4 --count 6 --warmup 1 \
  --channel-tile 512 --diagonal-chunk 64 --baby-step 128
```

首图为预热，图 1–5 为正式采样；输出保留原始样本、中位数、均值、范围、算子拆分、
GPU 温度/频率快照和源文件（含 `.cuh`）、二进制、共享库、模型、数据及参数哈希。
脚本从指定二进制同一构建目录加载 `libcheddar.so`，拒绝覆盖已有报告。
比较历史二进制时，应从与该构建对应的源码快照运行脚本，使源码哈希准确对应被测版本。

基础程序可额外输出规范化 RNS 变换结果，并测量 NTT+INTT 的同步墙钟时间：

```sh
build/example/cnn_primitives_test 32 17 65536 basic /tmp/ntt32-n17.rns
```

提供输出路径时执行 3 次预热和 10 次正式 NTT 往返采样。
本轮基线和优化版的 32/64 位 N17、32 位 N16 变换结果均逐系数完全一致；
32 位 N17 往返中位数从 0.103129 ms 降至 0.036486 ms（2.83 倍），
64 位 N17 从 0.086128 ms 降至 0.059743 ms（1.44 倍）。这些是特定 Q/P 配置的
NTT 往返收益，使用 level 2 的全部活跃 Q/P 素数；N16 不据此宣称性能提升。

2026-10-28–29 在同一 RTX 5090 上分别完整重建基线和优化版，五个模型按顺序执行，
每个模型的两版交替先后顺序。固定 N=131072、65536 槽、32 位后端、40/58 位 scale
目标、ReLU 范围 10、channel tile 512、diagonal chunk 64、baby step 128。
下表为各版本正式 5 张图的 online 中位数及 [最小值, 最大值]，单位 ms。

| 模型 | 基线 online | 优化后 online | 中位数降幅 | 优化版最大逐层密文误差 |
|---|---:|---:|---:|---:|
| VGG11 | 4026.661 [4025.718, 4029.857] | 3480.267 [3477.949, 3480.639] | 13.57% | 3.194e-4 |
| VGG16 | 6475.815 [6475.704, 6477.711] | 5592.689 [5589.731, 5594.884] | 13.64% | 1.956e-4 |
| MobileNet | 5994.022 [5993.042, 5996.980] | 4448.262 [4447.254, 4452.047] | 25.79% | 1.149e-3 |
| SqueezeNet | 3816.211 [3815.426, 3816.928] | 2672.788 [2671.268, 2673.744] | 29.96% | 1.489e-3 |
| AlexNet | 5973.186 [5972.267, 5974.565] | 5411.373 [5408.528, 5412.056] | 9.41% | 7.358e-4 |

含预热共 60 次整网推理全部通过。Boot 次数仍分别为 16/26/54/36/14，在线编码为零，
同多项式逐层误差门槛仍为 1e-2，四条执行路径的 top-1 均一致。
C++ 精确 ReLU logits 与独立 PyTorch 的最大差异为 4.79e-11。
本轮 FHE 与精确 ReLU 最终 logits 的最大差异为 0.2576（VGG16），包含固定多项式近似误差；
表中误差列比较的是相同多项式参考。六张图的检查不代表完整 CIFAR-10 准确率。

VGG11 的算子均值中，ReLU 从 1011.5 ms 降至 607.1 ms，卷积从 2930.3 ms 降至
2804.0 ms。不同网络的收益取决于实际算子构成。全部节点时间之和与原 online 时间一致，
原 N16 ResNet20 整网回归也通过：38 次 Boot、最大逐层误差 8.885e-5、预测一致。

本轮证据位于 `validation/online_20261028/`：

- `paired/FAMILY/{baseline,warp4}/report.json`：逐图、逐节点结果和原始日志。
- `comparison.json`、`compare.py`：跨版本参数、源码、权重、数据、明文参考及采样范围核对。
- `{baseline,warp4}-primitives/`：NTT 原始 RNS 输出、微测及算术检查。
- `build-audit.json`、`resnet20-regression.json`：相同构建选项和 N16 整网回归。

两版记录的计算源码仅 `include/core/NTTUtils.cuh` 不同，优化版报告的 81 个源码哈希
已与工作区核对一致。参数 SHA-256 仍为
`2b37e7a38cb9d27d782c4fe4d9241b105f56ed78c84fc443f28d34185a5f4c1e`。
