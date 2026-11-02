# AE 式 ResNet 与 CNN

本次参考固定为 `scale-snu/cheddar-ae` 提交
`17dd2509fa69efe6fd512eca85604e94414e20ba`。
AE 的 `ConvBN`、`DownSample`、`EvalReLU` 只有预编译实现；
这里使用开源 Cheddar 原语重建网络求值，并以发布库的旋转索引构造及公开驱动为依据。

## 默认旋转组织

3×3 卷积将 `pack × {−33,−32,−31,−1,0,1,31,32,33}` 全部放入 Baby，
Giant 只有零偏移，并显式禁止 Hoist 自动交换 Baby/Giant。
同一输入的 Baby 结果由多个输出通道分组共享；明文权重直接编码在其输出位置，
不再为非零 Giant 预先平移。
旋转 key 请求同步覆盖八个非零空间偏移，含对角方向。

以 `layer1.0.conv1` 为例，输入重复两份、输出分为八组：

| 逻辑 KS | 之前的水平/垂直拆分 | 当前 AE 布局 |
|---|---:|---:|
| 权重 Baby | 2 | 8 |
| 权重 Giant | 16 | 0 |
| 通道归约 | 32 | 32 |
| Gather | 15 | 15 |
| 重复归约 | 1 | 1 |
| 合计 | 66 | 56 |

默认 Pooling 保留零 Giant 的完整收集映射，共 30 次逻辑 KS；
默认 FC 使用 8×8 BSGS，共 14 次逻辑 KS。
显式 `--baby-step` 仍可覆盖 Pooling/FC，卷积保持 AE 布局。
逻辑 KS 数量不是独立完整 KS 调用数，也不能直接换算为加速比。
这些默认值改变了实际旋转 key 集合；已有旧版 KMS 包应重新生成并保存。

## 模型范围与实现边界

ResNet20/32/44/56/110 共用同一实现，每个 stage 分别为 3/5/7/9/18 个残差块。
VGG11/16、MobileNet、SqueezeNet、AlexNet 使用已有的固定 CIFAR-10 图及真实 ONNX 权重，
普通、分组、深度卷积复用相同的空间 Baby 组织；Concat、FC、Pool 保持各模型的语义。
后五个模型是沿用 AE 思路扩展的模型，官方 AE 没有发布它们的对应网络实现。
深度卷积只有一个输出分组，采用全空间 Baby 不保证比之前的二维拆分更快；
本次验证求值组织和正确性，没有据逻辑次数推断各模型的性能收益。

本次对齐的是网络算子路径和上述旋转组织。
开源实现仍使用 `BootConfig` 生成参数、稳定化的 Chebyshev 激活求值以及现有内存/编码管理，
不声称它们与官方预编译库的素数表、激活求值过程或输出逐位一致。
每个 ReLU 仍调用两次普通 Boot。
AE ResNet20 NPY 与其归一化单独导出为 `resnet20-ae.bin`；
其余模型使用各自的 ONNX 导出文件，不混用两种权重的预处理。

## 服务器使用

```bash
ssh -p 25218 root@connect.bjb1.seetacloud.com
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/cheddar-ae-models
export CIFAR_BIN="$PWD/data/cifar-10-batches-bin/test_batch.bin"

python scripts/run_inference.py --model resnet20 \
  --weights models/resnet20-ae.bin --cifar "$CIFAR_BIN"

python scripts/run_inference.py --model resnet110 --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model vgg16 --cifar "$CIFAR_BIN"
```

其他模型名为 `resnet32`、`resnet44`、`resnet56`、`vgg11`、`mobilenet`、
`squeezenet`、`alexnet`。32 GB GPU 上由 `auto` 决定预编码或流式编码。
三个计时开关及数值门槛沿用主 README，默认不开启计时。

验证时构建 `CHEDDAR_PLAINTEXT_REFERENCE=ON`，保持原有 `1e-2` 数值门槛：

```bash
python scripts/validate_resnet.py --binary build/example/resnet_cifar \
  --models models --cifar "$CIFAR_BIN" \
  --output validation/ae-models/resnet --count 3

python scripts/validate_cnn.py --binary build/example/cnn_cifar \
  --models models --cifar "$CIFAR_BIN" \
  --reference-dir validation/ae-models/reference \
  --output validation/ae-models/cnn --count 3
```

CNN 的独立参考通过 `scripts/cnn_float_reference.py` 直接重放源 ONNX，
使用 PyTorch CPU float64，与导出文件和 CIFAR 数据做 SHA-256 绑定。
公开 CIFAR-10 测试分片无损转换为 10,000 条原始 RGB 二进制记录；
来源与哈希保存在服务器 `data/test-data-manifest.json`。
完整日志、命令、源码/权重/程序哈希写入 `validation/ae-models/`。
少量样本回归不代表完整测试集准确率。

## 2026-11-02 实测

服务器为 RTX 5090（32607 MiB）、驱动 570.124.04、CUDA 12.8.93。
基础代码为 `16c7d53`，加本页所述的 AE 布局与入口修复。
下列每个模型均运行测试集图片 0、1、2，全部通过 strict 数值门槛，
每张图 CPU 编码尝试及密文求值内部编码次数均为零。

| 模型/权重 | 编码模式 | 最大逐层实部误差 | 每图 Boot |
|---|---|---:|---:|
| ResNet20，官方 AE 权重 | pre | 0.000224074 | 38 |
| ResNet20，ONNX | pre | 0.000325609 | 38 |
| ResNet32 | stream | 0.000630969 | 62 |
| ResNet44 | stream | 0.000779480 | 86 |
| ResNet56 | stream | 0.000728316 | 110 |
| ResNet110 | stream | 0.007256193 | 218 |
| VGG11 | stream | 0.001050124 | 16 |
| VGG16 | stream | 0.000382590 | 26 |
| MobileNet | stream | 0.001293220 | 54 |
| SqueezeNet | stream | 0.001754984 | 36 |
| AlexNet | stream | 0.001573945 | 14 |

五个 ONNX ResNet 的 C++ 精确 ReLU logits 与独立 PyTorch 回放最大差为
`2.40e-11`，五个 CNN 为 `2.79e-11`；所有测试图片的分类结果一致。
此比较检验模型导出和明文语义，密文误差由上表及逐层门槛另外检验。

另外通过了八种 ResNet 几何的真实密文卷积、六种普通 CNN 几何的真实密文回归、
全部 CNN 的第一层和网络前缀检查，以及 CPU 的八种 ResNet/二十二种 CNN 打包回放。
`--mode` 已禁止被 argparse 缩写匹配为 `--model`，
入口/计时协议及 NVTX 批处理的 CPU 回归通过。

这些运行打开了明文参考和 CPU scope 计时，属于正确性验证记录，
不能当作关闭参考、关闭 instrumentation 的性能基线。

## 官方原始库对照

服务器另有直接链接官方 `libcheddar_parfuse.so` 的程序：

```bash
cd /root/autodl-tmp/cheddar-ae-models/validation/ae-models/official-run
../../../build/example/official_ae_resnet \
  --gtest_filter=Cheddar/Testbed.ResNet20/resnetparam_40_json
```

官方库保持原样，使用独立目录内的 CUDA 11.8 runtime；系统 CUDA 12.8 未替换。
驱动只改为读取测试集、运行三张图片并打印 logits，网络算子和参数不变。
该程序保持官方默认 CPU 权重编码，GoogleTest 通过，三张图片分类均为 3/8/8，
与开源 AE 式实现一致。原版测试提供分类结果，不包含开源实现的逐层 `1e-2` 门槛。

外部每 200 ms 观察进程及日志，首次推理前的总准备约 `348.618 s`，
进程端到端约 `351.222 s`；观察边界与采样误差保存在
`validation/ae-models/official-wall-observation.json`。
准备包含上下文、密钥、Boot、权重和测试数据，不等于单独的 Encode 时间。
官方自身打印的三个同步 CPU 推理区间为 `734.231/734.259/734.151 ms`，
不含其计时器之前的输入加密，不是纯 GPU kernel 时间。

相同权重、相同三张图片的两个密文实现，最大 logits 差为 `0.1205269`。
开源实现自身对精确 ReLU 参考的最大 logits 差约 `0.00810`；
这些数据说明分类一致，但不能宣称与官方库数值完全复现。
参数、激活多项式求值和编码路径仍有实现差异，差值来源尚未逐项隔离。
对照输出保存在 `ae-library-comparison.json` 和 `official-resnet20.log`。
