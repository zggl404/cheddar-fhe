# CNN 卷积分解与验证

2026-10-30，VGG11/16、MobileNet、SqueezeNet、AlexNet 的运行入口已接入分解后的
卷积、FC 和池化。原来数百 GiB 的通用对角线缓存不再用于这些网络的默认推理。
本分支包含实现、测试和测评摘要；原始模型、数据及运行日志保留在本地忽略目录中。

## 求值方式

- 普通/grouped 卷积：临时重复输入，每份输入计算一个输出通道；共享空间 baby step，
  再归约输入通道并收集输出。1×1 权重只编码 Q，3×3 空间权重及收集掩码编码 QP。
- 输出分组复用收集掩码，逆序累加时只旋转相邻分组的位移差；避免每个输出分组各用一份旋转 key。
- stride=1 的 depthwise：直接执行空间卷积，省去输入重复、通道归约和输出收集；
  stride=2 另做布局收集。
- FC：按原 CHW 物理槽读取权重，临时重复输入、归约、共享收集；AlexNet 的
  1024→4096→4096→10 分类头也走此路径。
- Pool/GAP：先用 Trace 求空间和，再以包含平均因子的掩码收集，避免展开池化稀疏矩阵。

输出仍是原有的单份特征加零填充，Concat 和明文参考保持原语义。
解密检查的结果不会反馈到密文图。

CNN 的 Boot 可用输出由 level 8 调整为 level 9，ReLU 返回 level 3。
普通卷积及 FC 消耗两层，Pool 消耗一层，支持 SqueezeNet 的 Conv→Pool→Concat→ReLU 分支。
每个 ReLU 仍执行两次普通 Boot，没有额外插入 Boot；normal/Boot scale 目标仍为 40/58，
激活范围仍为 10，逐层实部/虚部误差门槛仍为原来的 1e-2。
ResNet 保留原来的 Boot level 8 / ReLU level 2。

默认 CNN 参数为 N=131072、65536 槽、uint32、Q=40 main+4 terminal、P=11，
最大 level 25，StC 从 level 13 开始。代码自动规划分组，不再接受旧的
`--channel-tile`、`--diagonal-chunk`、`--baby-step` 参数。

## 静态显存结果

单位为 GiB，包含当前 RMM bin 对每个对象的分配取整。旧值对应本次分解前的实现。
VGG11、MobileNet、SqueezeNet、AlexNet 根据本地 ONNX 形状；VGG16 初次按标准 CIFAR
结构统计，随后在 Downloads 找到实际 ONNX，并在 96 GB GPU 上核对形状和资源数完全一致。

| 模型 | 旧缓存 | 新缓存 | 模型明文条数，含偏置 | 最大实时编码批次 |
|---|---:|---:|---:|---:|
| VGG11 | 535.20 | 29.64 | 2768 | 1.022 |
| VGG16，标准结构 | 849.21 | 56.46 | 5268 | 1.022 |
| MobileNet | 451.63 | 9.20 | 1306 | 0.786 |
| SqueezeNet | 364.27 | 26.78 | 2946 | 1.375 |
| AlexNet | 865.39 | 12.72 | 1395 | 0.938 |

整网 pre 缓存共享掩码；stream 每批释放权重，重新准备下一批可能重复编码共享掩码，
所以运行期间的累计编码次数会高于整网缓存的明文条数。
偏置仅在第一批加入，输出批次之间只做密文相加。

| 模型 | 模型 / Boot 请求数 | 重叠数 | 合并 rot key 数 | rot key 大小 | pre 常驻主项 | stream 常驻主项及活跃权重 |
|---|---:|---:|---:|---:|---:|---:|
| VGG11 | 363 / 33 | 26 | 370 | 14.33 | 49.36 | 20.74 |
| VGG16，标准结构 | 371 / 33 | 26 | 378 | 14.50 | 76.34 | 20.91 |
| MobileNet | 388 / 33 | 27 | 394 | 14.85 | 29.43 | 21.02 |
| SqueezeNet | 360 / 33 | 19 | 374 | 14.42 | 46.58 | 21.18 |
| AlexNet | 252 / 33 | 20 | 265 | 12.07 | 30.18 | 18.40 |

主项还包含 CtS/StC 明文 4.6836 GiB、乘法/共轭 key 0.4297 GiB、
Dense↔Sparse key 0.2188 GiB、两份 secret 0.0537 GiB。
它仍未完整累计上下文表、运行临时量、内存池空闲保留块和 CUDA 驱动开销，不能当作实际峰值。
同一个旋转距离保留最高请求 level；不能将模型和 Boot key 数量直接相加。

auto 在全部 key 和 Boot 准备完成后决定 pre/stream；本次 32 GB GPU 上五个结构均选择 stream。
即使模型缓存只有 9.20 GiB，其他资源和保留余量也可能使整网 pre 不满足预算。

## 验证范围

CPU 独立 CHW/FC/Pool 参考覆盖 22 种几何，包括 grouped、depthwise、192/384 通道、
展平 FC、4096→4096 FC、普通及全局平均池化、分批相邻位移累加。
真实 GPU 密文单层回归覆盖同样的 22 种几何，并额外比较卷积和大 FC 的分批累加；
Boot9→ReLU level 3 在独立进程验证。统一计时 CPU 测试及入口协议测试也通过。
共享 ReLU 改动后，原公开权重的 ResNet20 整网回归通过。

五个 CNN 结构已在 RTX 5090 32607 MiB、CUDA 12.8.93 上完成合成权重整网密文回归。
输入为公开 CIFAR-10 训练集第一条记录。合成权重由固定种子生成，标准差为
`1/sqrt(fan_in)`，不是训练后的 checkpoint，因此这些检查验证求值流程，不能作为模型准确率。
VGG16 的检查对象是上面明确列出的标准结构。

以下为最终源码下一图合成整网回归结果，全部使用默认 auto→stream。
显存为 nvidia-smi 每 100 ms 的采样峰值，可能遗漏更短暂的峰值。

| 结构 | 采样峰值 GiB | 逐层最大实部/虚部误差 | 每图 Boot 次数 |
|---|---:|---:|---:|
| VGG11 | 25.0303 | 1.04910e-4 | 16 |
| VGG16，标准结构 | 25.1865 | 1.22761e-4 | 26 |
| MobileNet | 25.9365 | 1.15793e-4 | 54 |
| SqueezeNet | 27.5615 | 1.11101e-4 | 36 |
| AlexNet | 22.9365 | 1.97373e-4 | 14 |

单层 GPU 回归及额外分批比较的最大复数误差为 `1.90398e-4`，
独立 Boot9/ReLU 误差为 `2.1742e-6`。VGG11 前五个节点另以 pre 和 stream
分别通过；pre 的 `runtime_weight_prepare_ms=0`，两者密文求值内编码次数均为 0。
五个整网的静态缓存字节数、模型/Boot 旋转请求数已与实际 RESOURCES 核对一致，
三段及推理细分时间保持 schema_version=2 的加法关系。

随后已按用户授权，将全部实际 ONNX 导出的权重同步到指定 96 GB GPU，
包括 Downloads 下的真实 VGG16，十个模型的整网预编码测评均已完成。
真实权重的误差显著大于上面的合成权重示例，详见[96 GB 实测报告](gpu96-benchmark.md)。

## 运行与复现

主 README 的导出和运行命令保持相同：更新源码后重新编译，再运行统一入口。

```bash
python scripts/run_inference.py --model mobilenet --cifar /data/test_batch.bin
python scripts/run_inference.py --model alexnet --cifar /data/test_batch.bin --encode-mode stream
```

额外回归：

```bash
build/example/cnn_linear_plan_test
build/example/cnn_decomposition_test
build/example/cnn_decomposition_test 0
```

Boot9 测试独立运行，避免测试进程同时保留所有不相关几何的密钥及其内存池高水位。
历史上将两类测试放在同一进程时，22 个单层均通过，但之后额外准备 Boot 发生 OOM；
该日志保留，不将它计为通过结果。

代码入口：
[CPU 布局规划](../include/extension/CnnLinearPlan.h)、
[密文求值](../src/extension/CnnDecomposed.cpp)、
[CNN 运行器](../example/CnnCifar.cpp)。
本地证据保存在被 git 忽略的 `validation/cnn-decomposition-20261030/`，
包含布局及静态资源计划、单层及整网日志、100 ms 显存采样、导出清单和二进制哈希。
其中 `verify_results.py` 会核验源码哈希、五个结构的静态/运行资源、计时加法关系、
单层/分批/Boot9/ResNet20 和 pre/stream 前缀回归。
实际验证的共享库 SHA256 为
`6cada6717cf507c3a61ab1bbda95a0cf9c18ba34b3a14075a6b62cf65bb04f07`，
CNN 可执行文件 SHA256 为
`4df5499c48c2b85b14c9b184aee2dff5de0c85349f2d18f36a91dd0a0bfb7de5`。
