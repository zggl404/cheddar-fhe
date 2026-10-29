# 当前模型显存与旋转 key 静态统计

> 历史快照：本页的其他 CNN 数字对应通用对角线实现。
> 它们随后已接入卷积分解，当前结果见 [CNN 分解记录](cnn-decomposition.md)。
> ResNet 的预编码缓存及参数保持本页口径。
> 后续全部真实模型的整网预编码显存与时间见[96 GB 实测报告](gpu96-benchmark.md)。

统计日期：2026-10-30。基础提交为 `9fff247d81139edd9665cecd484b283fe8a19e93`，
包含 AE 式 ResNet 卷积补丁。本页对应 CPU 静态分析，后续 GPU 实测结果见上方链接。
下列单位均为 **GiB（2^30 字节）**，已计入当前 RMM bin 对每个对象的分配取整。

## 范围与口径

- 默认 uint32，normal/Boot scale 目标 40/58 bit，P=11，完整 Q 池为 37 main + 6 terminal。
- ResNet 使用新 AE 式卷积，N=65536，Boot 槽数 16384；其他 CNN 保留当前通用对角线实现，N=131072，槽数 65536。
- CNN 默认 channel tile=512、diagonal chunk=64、baby step=128；ResNet 卷积按输入重复分组，Pool/FC 的 baby step=16。
- ResNet 根据当前 ConvPlan 和网络循环统计；VGG11、MobileNet、SqueezeNet、AlexNet 根据本地 ONNX 的节点、张量形状和当前布局统计。
- 四个 CNN 的 FP32 张量已检查，没有零值，形状产生的对角线支持集与当前跳过零权重的实现一致。
- **本地没有 VGG16 ONNX**。该行按标准 CIFAR VGG16 的五组卷积数 `[2,2,3,3,3]`、通道 `[64,128,256,512,512]`、五次平均池化、512→10 分类头估算；不能用于任意同名模型。

“预编码主项合计”包含整网权重/掩码/偏置、去重后的旋转 key、基础及 Boot 切换 key、
CtS/StC 矩阵明文和两份 secret。**它不是完整 GPU 峰值**：上下文固定表、EvalMod/ReLU 小常数、
运行临时密文、密钥生成临时量、内存池空闲保留块及 CUDA 驱动开销尚未完整累计。
RMM 对象分配取整与内存池未使用的预留块是两项，不能混为一谈。

## 整网预编码

“对角线/全部明文”中，全部明文额外计入偏置；CNN 还计入每个 chunk 内的零偏置明文。
Boot 的矩阵明文单独计入公共资源，不包含在模型明文条数内。

| 模型 | 对角线 / 全部模型明文 | RNS 系数本体 | 权重缓存分配 | 全部旋转 key | 预编码主项合计 |
|---|---:|---:|---:|---:|---:|
| ResNet20 | 2354 / 2377 | 8.71 | 12.55 | 4.86 | 19.38 |
| ResNet32 | 3666 / 3701 | 13.62 | 19.60 | 4.86 | 26.44 |
| ResNet44 | 4978 / 5025 | 18.53 | 26.65 | 4.86 | 33.49 |
| ResNet56 | 6290 / 6349 | 23.44 | 33.70 | 4.86 | 40.54 |
| ResNet110 | 12194 / 12307 | 45.55 | 65.44 | 4.86 | 72.28 |
| VGG11 | 49711 / 50502 | 367.22 | 535.20 | 14.33 | 554.85 |
| VGG16，结构估算 | 78859 / 80112 | 595.61 | 849.21 | 14.33 | 868.86 |
| MobileNet | 41917 / 42619 | 327.49 | 451.63 | 13.58 | 470.53 |
| SqueezeNet | 33811 / 34374 | 263.01 | 364.27 | 16.46 | 386.05 |
| AlexNet | 80363 / 81632 | 606.72 | 865.39 | 11.99 | 882.70 |

其他 CNN 尚未采用本次 ResNet 的重复槽卷积分解，因此仍需大量通用矩阵对角线；
不能将 ResNet 的权重缩减比例直接用于这些网络。这里的权重缓存是展开、编码后的 GPU 明文，
不是 FP32 checkpoint 大小，也不是 CPU 内存需求。

## 旋转 key 数量与去重

按旋转距离合并模型与 Boot 的请求，同一距离仅保留能覆盖最高请求 level 的一份 key。
key 数量不是旋转操作执行次数，也不随图片数或 Boot 次数重复增长。

| 模型 | 模型请求距离数 | Boot 请求距离数 | 重叠距离数 | 实际合并 rot key 数 | Boot 之外的独立模型 rot key |
|---|---:|---:|---:|---:|---:|
| ResNet20/32/44/56/110 | 208 | 30 | 23 | 215 | 185 |
| VGG11 | 376 | 33 | 33 | 376 | 343 |
| VGG16，结构估算 | 376 | 33 | 33 | 376 | 343 |
| MobileNet | 341 | 33 | 33 | 341 | 308 |
| SqueezeNet | 475 | 33 | 33 | 475 | 442 |
| AlexNet | 264 | 33 | 30 | 267 | 234 |

例如 ResNet 为 `208+30-23=215`。Boot 重叠距离要使用较高 level 的 key；
不能简单将“208 个低 level 模型 key 的大小”加到“30 个 Boot key 的大小”上。
上述 rot key 数以外，每个进程还准备 4 个特殊 evaluation key：
乘法、共轭、dense→sparse 和 sparse→dense。

| 模型组 | Boot 外独立模型 rot key | Boot rot key | 基础乘法/共轭 key | Boot 两个切换 key | CtS/StC 明文 | 两份 secret |
|---|---:|---:|---:|---:|---:|---:|
| ResNet 全系列 | 1.9873 | 2.8711 | 0.2109 | 0.1074 | 1.6355 | 0.0264 |
| VGG11/16 | 7.3691 | 6.9609 | 0.4219 | 0.2148 | 4.6304 | 0.0527 |
| MobileNet | 6.6172 | 6.9609 | 0.4219 | 0.2148 | 4.6304 | 0.0527 |
| SqueezeNet | 9.4961 | 6.9609 | 0.4219 | 0.2148 | 4.6304 | 0.0527 |
| AlexNet | 5.0273 | 6.9609 | 0.4219 | 0.2148 | 4.6304 | 0.0527 |

该表各列互斥：重叠旋转距离归入 Boot rot key，避免重复累计。
Boot 矩阵在 ResNet 配置下有 154 条，在其他 CNN 配置下有 218 条。
这些资源在同一进程内复用，ResNet20 每图 38 次 Boot、ResNet110 每图 218 次 Boot
不会分别创建 38 或 218 套 Boot key。

## 单个 key 的大小如何计算

明文在 level l 的 Q 模数数为 `main[l]+terminal[l]`。
但 UserInterface::GetNPForEvk(l) 为了覆盖低层求值，会使用：

```text
q_key(l) = max(main[0], ..., main[l]) + 全部 terminal 数量
beta = ceil(q_key(l) / P)
key_bytes = 2 × beta × AllocationBytes(N × 4 × (q_key(l) + P))
```

`2` 对应每个分解分量的 ax、bx 两个多项式；当前 key 没有使用只存 seed 的压缩形式。
DtS 是例外：短基 Q=2、P=2、beta=1。
StD 虽将各分量相加到第一个分量，源码仍保留原有 beta 组存储，按全部分量计入。

| key 类型 | Q / P | beta | N16 单个 key | N17 单个 key |
|---|---:|---:|---:|---:|
| 请求 level 1/2 的模型旋转 | 9 / 11 | 1 | 11 MiB | 22 MiB |
| 请求 level 21–24 的 Boot 旋转 | 37–43 / 11 | 4 | 108 MiB | 216 MiB |
| 最大 level 的乘法、共轭、StD | 43 / 11 | 4 | 每个 108 MiB | 每个 216 MiB |
| DtS | 2 / 2 | 1 | 2 MiB | 4 MiB |

ResNet 的 30 个 Boot 旋转中，26 个位于 level 21–24，4 个位于 level 10–12；
后者每个 33 MiB，所以 Boot rot 合计为 `(26×108+4×33)/1024=2.87109375 GiB`。
N17 的 33 个 Boot 旋转全部位于 level 21–24，合计 `33×216/1024=6.9609375 GiB`。

## 实时编码时

stream 保留全部所需密钥和 Boot 资源，按层或 chunk 编码、求值并释放权重对象。

| 模型 | 权重之外的常驻主项 | 最大活跃编码权重 | 两项合计，仍非完整峰值 |
|---|---:|---:|---:|
| ResNet 全系列 | 6.839 | 1.493 | 8.332 |
| VGG11/16 | 19.650 | 0.689 | 20.339 |
| MobileNet | 18.898 | 0.689 | 19.587 |
| SqueezeNet | 21.777 | 0.689 | 22.466 |
| AlexNet | 17.308 | 0.689 | 17.998 |

ResNet 的 1.493 GiB 来自第三阶段投影块：shortcut 权重 384.5 MiB
在整个块中存活，与 conv1 的 1144.5 MiB 相加，最大同时活跃为 1529 MiB。
CNN 的单个 64 对角线 chunk 为 `64×11+2=706 MiB`。
实际 node bias、临时密文及编码工作区仍需另外计入；RMM 也可能保留已释放的权重块。

默认 auto 在模型及 Boot 密钥准备后计算权重预算，另外预留 4 GiB。
这 4 GiB 是预算策略，不能视为全部临时空间的已验证上限。
在本次 RTX 5090 32 GB 的准备状态下，ResNet20 的权重预算为 15.806 GiB，
可选择 pre；相同条件下其他 ResNet 的权重估算超过该预算，预计选择 stream。
五个其他 CNN 的整网权重缓存均超过 32 GB，使用该卡时需要 stream。
上述 stream 主项不足以单独证明所有模型在 32 GB 上均不会 OOM。

## 与已验证结果的核对

ResNet20 的 CPU 静态结果与上次 GPU 完整验证日志一致：

```text
模型明文缓存：13471055872 字节 = 12.5458984375 GiB
模型旋转请求：208
Boot 旋转请求：30
第一张图 GPU 向量编码次数：2377 模型明文 + 154 Boot 矩阵 + 1 输入 = 2532
```

预编码常驻主项为 19.3845 GiB，上次完整运行的 100 ms 显存采样峰值为 23.9365 GiB。
两者约 4.55 GiB 的差额包含未累计的上下文、临时量、空闲保留块及驱动等，
不能归因给某一种资源，也不能作为其他 CNN 的固定修正值。
GPU 运行与误差范围见 [ResNet20 验证记录](resnet-ae-validation.md)。

本次证据存于本地被 git 忽略的 `validation/model-memory-static-20261030/`：
`host_plan.cpp` 使用当前参数解析器和 ConvPlan 生成参数与 ResNet 旋转集合，
`host_plan.json`、`boot_plan.json` 保存计划，`audit.py`、`audit.log`、`results.json`
保存计算过程及结果。results.json 包含每个旋转距离、请求 level、Q/P、beta、分配字节数，
以及输入 ONNX 和相关源码的 SHA256。

主要源码依据：
[ResNet 明文与旋转规划](../src/extension/ResNetOps.cpp)、
[CNN 分块与密钥请求](../example/CnnCifar.cpp)、
[key 参数和生成](../src/UserInterface.cu)、
[key 存储结构](../src/core/Container.cpp)、
[Boot FFT 规划](../src/extension/EvalSpecialFFT.cpp)、
[Hoist 编码](../src/extension/Hoist.cu)、
[内存池](../src/core/MemoryPool.cpp)。
