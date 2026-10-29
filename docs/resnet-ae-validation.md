# ResNet20 AE 式卷积验证记录

本文记录公开 AE 权重的 ResNet20 回归。后续本地 ONNX 的所有 ResNet 深度和其他 CNN
均已完成真实权重测评，见[96 GB 实测报告](gpu96-benchmark.md)；两轮权重与硬件不同。

2026-10-30，优化后的手写 ResNet20 已在 RTX 5090 上完成真实密文整网验证。
默认 `auto` 选择整网预编码，权重缓存估算 **12.55 GiB**，GPU 显存采样峰值
**23.94 GiB**；正式样本的平均 FHE 求值 **0.777 秒/图**。
本次按用户要求仅运行优化后的 ResNet，没有运行官方 cheddar-ae 基线。

## 修正内容

原实现先将整个卷积展开成稀疏矩阵，再为其对角线逐条预编码。
之前约 75.93 GiB 的数字是这种实现的权重缓存估算，并非实测 GPU 峰值。
本次改为根据 AE 实现分析得到的分解方式：

1. 将输入特征重复到全部 `N/2` 物理槽，每份输入并行计算一个输出通道。
2. 3×3 卷积只构造空间位移的权重，各输出分组共享 baby-step 结果。
3. 使用 Trace 归约输入通道，再用输出掩码收集各输出通道。
4. 合并分组，并恢复输出特征的重复布局。

普通卷积权重在 level 2 的 QP 编码，输出掩码在 level 1 的 QP 编码；
1×1 投影权重只在 level 2 的 Q 编码，偏置在 level 0 的 Q 编码。
因此显存下降主要来自卷积分解和对角线数量减少，并非把所有明文改为 Q。
卷积消耗两层，输出 level 0；残差相应对齐到 level 0，ReLU 输出仍为 level 2。
池化和分类也相应采用重复槽布局。没有修改内存池分配策略。

实现位于 [ConvPlan](../include/extension/ResNetConvPlan.h)、
[ConvBN/PackedLinear](../src/extension/ResNetOps.cpp) 和
[整网入口](../example/ResNetCifar.cpp)。这是使用开源 Cheddar 原语实现的
AE 式求值流程；它不链接 AE 的预编译网络实现，也不声称与其二进制逐位一致。
本次没有修改其他 CNN 的通用卷积路径。

## 环境与输入

| 项目 | 配置 |
|---|---|
| GPU | RTX 5090，32607 MiB |
| 驱动 / CUDA | 580.105.08 / 12.8.93，架构 120 |
| 宿主内存限制 | 96636764160 字节，约 90 GiB |
| 基础提交 | `9fff247d81139edd9665cecd484b283fe8a19e93`，另加本地补丁 |
| 密码参数 | uint32，N=65536，normal/Boot scale 目标 40/58 bit |
| Boot 槽数 / 次数 | 16384 / 每图 38 次 |
| 特征槽数 | 输入 4096；三阶段 16384/8192/4096；池化及 FC 后 64 |
| ReLU 范围 | 10 |
| 编码 | 默认 auto，实际选择 pre |

权重来自公开的 cheddar-ae 提交
`17dd2509fa69efe6fd512eca85604e94414e20ba` 的 ResNet20 NPY，
通过仓库导出器生成 FP32 二进制。归一化 mean 为
`[0.4914,0.4822,0.4465]`，std 为 `[0.2023,0.1994,0.2010]`。
输入为原始 CIFAR-10 公开二进制 `data_batch_1.bin` 的前四条训练记录，
标签为 `[6,9,9,4]`。仅用于少量样本的数值回归，不能作为测试集准确率。

| 文件 | SHA256 |
|---|---|
| resnet20-ae.bin | `9703b134080cbfb0cb886a545709452f4484c4fb660ec9f70678b332b16d5d79` |
| cifar10-train-first4.bin | `1d157899f621e501a9271254d5cfe92510f12d34b525844b0126d350e84d3398` |
| src/extension/ResNetOps.cpp | `d25e5bef2761e40b00bbf2a17e534e5a7aaa0c9cfe1bf9d4f4677ef92d55153b` |
| include/extension/ResNetConvPlan.h | `143bca6aaa491c606660b6eade4e2d2362475fe7c2e2c884095acd49c9eab235` |
| libcheddar.so | `db0b0cd54acccf9deaadac2d2099b2cf44eee819e6fe40e2d998679b87cec041` |
| resnet_cifar | `4fc3c44525249ea7f705deb852a07a38b99e219160198acfab85a2b8d6e0020c` |

## 运行方式

依照主 README 构建后，使用统一入口：

```bash
python scripts/run_inference.py --model resnet20 \
  --weights /data/resnet20-ae.bin \
  --cifar /data/cifar10-train-first4.bin \
  --count 4 --warmup 1 --log-format json
```

本次 GPU 上使用上述命令，路径分别为
`/root/autodl-tmp/ae-validation/resnet20-ae.bin` 和
`/root/autodl-tmp/ae-validation/data/cifar10-train-first4.bin`，
源码及构建位于 `/root/autodl-tmp/cheddar-memory`。
预热包含在四张图片及整次墙钟时间中；统计均值仅使用后面三张。
另一次强制 `--encode-mode pre` 的运行也已通过。

## 显存

| 项目 | 结果 | 口径 |
|---|---:|---|
| 权重及掩码明文条数 | 2354 | 含 Pool/FC，不含偏置 |
| 整网权重缓存 | 13471055872 字节，12.5459 GiB | 含偏置及 RMM 分配取整的估算 |
| 默认 auto 的缓存预算 | 16971792384 字节，15.806 GiB | 密钥及 Boot 准备后计算 |
| GPU 显存采样峰值 | 24511 MiB，23.9365 GiB | nvidia-smi 每 100 ms 采样 |
| 单图输出时 resident_gpu_bytes | 25700794368 字节 | RMM 保留量快照，不能当作峰值 |
| 模型 / Boot 旋转请求数 | 208 / 30 | 请求集合计数，不能相加视为独立分配密钥数 |

采样峰值包含模型密钥、Boot 资源、权重、上下文和临时分配；可能遗漏短于采样间隔的峰值。
权重缓存估算不是整网最低显存要求，也不能用其减去采样峰值来精确分解其他资源。
推理过程中没有运行时权重准备，`runtime_weight_prepare_ms=0`；
全部样本 `cpu_encode_attempts=0`、`fhe_encode_calls=0`。
每图输入加密需要的 GPU 编码在 input_ms 中，已与密文图求值分开。

## 统一计时结果

三个一级时间的范围保持 schema_version=2，不增加新的 online 计时口径：

| 整次运行 | 时间 ms |
|---|---:|
| model_prepare_ms：模型资源、密钥、权重预编码及外层启动/收尾 | 3440.226560 |
| boot_prepare_ms：Boot 资源准备 | 2785.307942 |
| inference_ms：全部四张图 | 4018.955168 |
| total_ms：三段之和 | 10244.489670 |

以下各项为同一组三张正式样本的均值：

| 单图项目 | 时间 ms |
|---|---:|
| 输入处理与加密 | 5.082582 |
| 运行时权重准备 | 0 |
| 线性层 | 149.821482 |
| Boot 执行 | 541.504358 |
| Boot 之外的激活 | 84.863264 |
| 其他图操作 | 0.928543 |
| **FHE 求值小计** | **777.117648** |
| 明文参考 | 190.002679 |
| 解密检查与输出 | 32.723743 |
| 调度及其余推理开销 | 0.593664 |
| **完整单图推理** | **1005.520316** |

FHE 小计等于线性层、Boot、激活和其他图操作之和；完整推理另含输入、参考、校验等。
所有时间为同步 CPU 墙钟，含同步及计时开销，不是纯 GPU kernel 时间。
不能将此 FHE 小计与完整单图时间混用，也没有运行官方 AE 来建立同条件配对比较。

## 正确性与回归

四张图全部通过原有逐层实部/虚部 `1e-2` 门槛，门槛及参数没有放宽。
密文图不会读取解密检查的结果。最终输出与两条明文参考的 top-1 都一致。

| 比较 | 最大绝对误差 | 平均绝对误差 |
|---|---:|---:|
| FHE 与相同激活多项式的明文 logits | 1.05159e-4 | 1.95113e-5 |
| FHE 与普通 ReLU 的明文 logits | 3.40978e-3 | 1.21248e-3 |

表中误差在四张图全部 40 个 logits 上计算，均为原始实数单位。
所有逐层检查中的最大实部误差为 `3.56277e-4`。
第一行主要反映同态计算误差，第二行还包含激活多项式近似误差。

| 检查 | 结果 |
|---|---|
| CPU 卷积打包、Trace、掩码与独立 CHW 卷积对比 | 8 种几何通过，最大误差约 1.7e-16 |
| 非法步长及超容量布局 | 正确拒绝 |
| GPU 实际密文卷积，含两种投影 | 8 种几何通过原有 1e-2 门槛，最大误差 1.88405e-5 |
| 额外严格 1e-5 单层诊断 | **未达到**，日志保留 `strict_1e-5=NOT_MET` |
| N16 通用 PackedLinear GPU 回归 | 通过，末项误差 1.58642e-7 |
| N17 通用 PackedLinear GPU 回归 | 通过，末项误差 6.17802e-7 |
| CPU 统一计时组件与跨进程入口协议 | 通过 |
| ResNet20 full，默认 auto，四张图 | 全部 PASS，每图 Boot 38 次 |

ResNet32/44/56/110 共享该层实现，但本次尚未完成它们的整网 GPU 验证。
其他 CNN 保留原有实现；本次结果不代表其整网显存或正确性已重新验证。

原始日志保存在本地被 git 忽略的
`validation/resnet-ae-20261030/`：`full-verified.log`、`memory-verified.csv`、
`primitive-verified.log`、`verified.sha256`、`summary.json`，以及打包和线性层回归日志。
源码哈希已与 GPU 实际构建的文件核对。实现及本页摘要随分支保存，原始模型和日志保留在本地忽略目录中。
