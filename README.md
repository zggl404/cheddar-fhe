# Cheddar：从 ONNX 运行 CIFAR-10 密文推理

本分支 `feature/cnn-n17` 提供手写 C++/CUDA 推理入口，支持 ResNet20/32/44/56/110、
VGG11/16、MobileNet、SqueezeNet 和 AlexNet，包含 N17 NTT 线程分组优化。

本指南只假定你已经准备好对应的 ONNX 文件和 CIFAR-10 二进制测试数据。
不依赖旧服务器上的 build、导出权重、验证报告或明文参考文件。
完整流程为：

**安装依赖 → 设置文件路径 → 编译 → 逐个导出权重 → 运行密文模型 → 查看明文/密文输出与误差。**

直接运行全部十个模型按第 1–6 节操作；独立明文参考、分阶段验证和重复性能采样见第 8 节。
导出和生成独立明文参考在 CPU 上执行；密文推理需要 Linux NVIDIA CUDA 环境。

## 1. 获取代码并准备环境

### 1.1 获取当前分支

```bash
git clone --branch feature/cnn-n17 https://github.com/zggl404/cheddar-fhe.git
cd cheddar-fhe
```

已有仓库时，在工作区无未处理修改的前提下切换到 `feature/cnn-n17`，再使用 `git pull --ff-only`
更新。后续命令均从仓库根目录执行，使用 Bash。不要在构建期间修改或同步源码。

### 1.2 GPU、CUDA 和系统依赖

已实测环境是 Ubuntu 22.04、GCC 11、CUDA 12.8.93、CMake 3.29.6、RTX 5090 32 GB。
五个 N17 CNN 的整网验证在这台 32 GB GPU 上完成；其他 GPU/内存组合需要实际验证。
Mac 可以导出权重，密文推理请在上述 Linux CUDA 环境执行。

先检查驱动与 CUDA **开发工具包**：

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` 能运行不代表 `nvcc` 已安装。RTX 5090 使用支持其架构的 CUDA 12.8 开发环境。
如果 CUDA 安装在 `/usr/local/cuda`，可设置：

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

尚未安装 CUDA 时，先按 [NVIDIA CUDA 12.8 Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)
准备驱动和开发工具包，再继续本指南。

Ubuntu 系统依赖如下；以 root 登录时去掉 `sudo`：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libtommath-dev libspdlog-dev python3 python3-pip python3-venv
```

### 1.3 Python 环境

下面使用 Python 3.10/3.11；这些固定依赖不面向 Python 3.12 及以上版本。
虚拟环境放在仓库旁边，避免与源码混放：

```bash
python3 --version
python3 -m venv ../cheddar-cifar-venv
source ../cheddar-cifar-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.24.4 onnx==1.14.1 cmake==3.29.6
cmake --version
```

CMake 最低要求是 3.24，Ubuntu 默认 apt 版本可能不够。
NumPy 和 ONNX 用于导出权重；本节不需要安装 PyTorch。
完成编译和导出后，C++ 推理本身不调用 Python/ONNX。

重新登录或打开终端后，需要重新激活虚拟环境，并重新设置下一节的路径变量。

## 2. 设置 ONNX、数据和输出路径

### 2.1 输入文件约定

下面按一个 ONNX 目录举例。文件名不同时，请修改对应导出命令；文件分布在多个目录时，
同样可以直接给每条命令指定实际 ONNX 路径。

| 模型 | 本指南使用的 ONNX 文件名 | 导出后的权重 |
|---|---|---|
| ResNet20 | `resnet20_cifar10_pre.onnx` | `resnet20.bin` |
| ResNet32 | `resnet32_cifar10_pre.onnx` | `resnet32.bin` |
| ResNet44 | `resnet44_cifar10_pre.onnx` | `resnet44.bin` |
| ResNet56 | `resnet56_cifar10_pre.onnx` | `resnet56.bin` |
| ResNet110 | `resnet110_cifar10_train.onnx` | `resnet110.bin` |
| VGG11 | `vgg11_cifar10.onnx` | `vgg11.bin` |
| VGG16 | `vgg16_cifar10.onnx` | `vgg16.bin` |
| MobileNet | `mobilenet_cifar10.onnx` | `mobilenet.bin` |
| SqueezeNet | `squeezenet_cifar10.onnx` | `squeezenet.bin` |
| AlexNet | `alexnet_cifar10.onnx` | `alexnet.bin` |

模型必须是本项目支持的、BN 已折叠的 CIFAR-10 推理图，输入为 `[1,3,32,32]`，
输出为 `[1,10]`。不要混用 CIFAR-100 权重。ResNet 导出器还检查固定节点命名和残差结构；
CNN 导出器检查各 family 的算子清单与属性。同名的 torchvision ImageNet 网络不能直接代入。

这里的 CNN 使用 AveragePool；当前不支持 MaxPool、ReLU6、未折叠的 BatchNormalization、
非 2 的幂空间布局或多密文特征图。AlexNet 是本项目的 CIFAR 版本，
包含 3×3 卷积、平均池化和 1024→4096→4096→10 分类头。

CIFAR 输入必须指向原始二进制文件 `test_batch.bin`，而不是数据目录、图片目录或 pickle 文件。
每条记录为 1 字节标签 + 3072 字节 CHW 像素；标准 10000 张测试集大小为 30730000 字节。
程序读取原始像素并按权重中保存的 mean/std 归一化，不需要手动预处理。
从 ONNX 导出时，当前脚本内嵌 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]，
对应现有 ACE CIFAR 模型的预处理；使用其他训练来源的 checkpoint 时，先核对其归一化约定。

### 2.2 修改这两个输入路径

以下路径只是示例，换成你的真实路径。导出权重、构建产物和日志目录由后续步骤生成：

```bash
export FHE_ONNX_DIR="/data/onnx"
export FHE_CIFAR_BIN="/data/cifar-10-batches-bin/test_batch.bin"
export FHE_WEIGHTS_DIR="$PWD/models"
export FHE_BUILD_DIR="$PWD/build"
export FHE_RUN_DIR="$PWD/validation/manual-01"
export FHE_CNN_TILE=512
export FHE_LOG_FORMAT=human

mkdir -p "$FHE_WEIGHTS_DIR" "$FHE_RUN_DIR"
```

默认 FHE_CNN_TILE=512、diagonal chunk=64、baby step=128，采用当前已验证的优化分块；\n编码默认 auto，显存预算足够时整网预编码，不足时实时编码。\nCPU 内存不足时将 FHE_CNN_TILE 降为 16 或 8，并记录新配置后重测。\n\n检查输入文件是否齐全：

```bash
python - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["FHE_ONNX_DIR"])
names = [
    "resnet20_cifar10_pre.onnx", "resnet32_cifar10_pre.onnx",
    "resnet44_cifar10_pre.onnx", "resnet56_cifar10_pre.onnx",
    "resnet110_cifar10_train.onnx", "vgg11_cifar10.onnx",
    "vgg16_cifar10.onnx", "mobilenet_cifar10.onnx",
    "squeezenet_cifar10.onnx", "alexnet_cifar10.onnx",
]
missing = [str(source / name) for name in names if not (source / name).is_file()]
if missing:
    raise SystemExit("Missing ONNX files:\n" + "\n".join(missing))
data = Path(os.environ["FHE_CIFAR_BIN"])
if not data.is_file():
    raise SystemExit(f"Missing CIFAR binary: {data}")
size = data.stat().st_size
if not size or size % 3073:
    raise SystemExit(f"Invalid CIFAR binary size: {size}")
print("ONNX files: 10")
print("CIFAR records:", size // 3073)
PY
```

如果你使用了不同的文件名或分散的目录，同步调整这段检查及后面的导出路径。

## 3. 编译两个推理入口

### 3.1 准备 RMM 头文件

下面显式使用 RMM 22.12 头文件和已安装的系统 spdlog，不要求你预先安装 RMM 库：

```bash
git clone --depth 1 --branch branch-22.12 https://github.com/rapidsai/rmm.git ../cheddar-rmm
export FHE_RMM_INCLUDE_DIR="$(cd ../cheddar-rmm/include && pwd)"
```

如果已经有 RMM 22.12 源码，直接把 `FHE_RMM_INCLUDE_DIR` 设置为其 `include` 的绝对路径，
不用重复 clone。路径下应存在 `rmm/mr/device/binning_memory_resource.hpp`。
GitHub 访问失败时，可在能访问的机器下载该版本源码后传入服务器，保持相同目录结构。

### 3.2 选择 GPU 架构并构建

RTX 5090 使用 `120`。其他 GPU 按实际型号调整，不能把库的通用支持视为该模型已验证：

| GPU | `CMAKE_CUDA_ARCHITECTURES` |
|---|---:|
| RTX 5090 / RTX PRO 6000 Blackwell | 120 |
| RTX 4090 | 89 |
| RTX 3090 | 86 |
| H100 / H200 | 90 |
| A100 | 80 |

架构编号依据 [NVIDIA Compute Capability 表](https://developer.nvidia.com/cuda/gpus)，写成不带小数点的数字。

```bash
cmake -S . -B "$FHE_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCHEDDAR_RMM_INCLUDE_DIR="$FHE_RMM_INCLUDE_DIR" \
  -DENABLE_EXTENSION=ON \
  -DBUILD_UNITTEST=OFF \
  -DBUILD_RESNET_EXAMPLE=ON \
  -DBUILD_CNN_EXAMPLE=ON

cmake --build "$FHE_BUILD_DIR" --parallel 12
```

检查产物并设置动态库路径：

```bash
test -x "$FHE_BUILD_DIR/example/resnet_cifar"
test -x "$FHE_BUILD_DIR/example/cnn_cifar"
test -f "$FHE_BUILD_DIR/libcheddar.so"

export LD_LIBRARY_PATH="$FHE_BUILD_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

两个可执行文件和 `libcheddar.so` 必须来自同一个构建目录。
CPU 核心或内存较少时可将 `--parallel 12` 改小。

若曾在构建中途修改/同步源码，或怀疑增量产物不一致，应停止同步后完整重建：

```bash
cmake --build "$FHE_BUILD_DIR" --clean-first --parallel 12
```

可选：先做基础算术、线性层及 N17 Boot/ReLU 检查，再运行整网：

```bash
"$FHE_BUILD_DIR/example/parameter_literal_test"
"$FHE_BUILD_DIR/example/cnn_primitives_test" 32 16 16384 linear
"$FHE_BUILD_DIR/example/cnn_primitives_test" 32 17 65536 relu
```

## 4. 逐个导出全部十个模型的权重

ONNX 不能直接传给 C++ 的 `--weights`。先在 CPU 上导出本项目的 `.bin`；
下面每条命令独立执行，可以按需要只导出其中一个模型。

### 4.1 ResNet 系列

```bash
python scripts/export_resnet_weights.py "$FHE_ONNX_DIR/resnet20_cifar10_pre.onnx" "$FHE_WEIGHTS_DIR/resnet20.bin" --depth 20

python scripts/export_resnet_weights.py "$FHE_ONNX_DIR/resnet32_cifar10_pre.onnx" "$FHE_WEIGHTS_DIR/resnet32.bin" --depth 32

python scripts/export_resnet_weights.py "$FHE_ONNX_DIR/resnet44_cifar10_pre.onnx" "$FHE_WEIGHTS_DIR/resnet44.bin" --depth 44

python scripts/export_resnet_weights.py "$FHE_ONNX_DIR/resnet56_cifar10_pre.onnx" "$FHE_WEIGHTS_DIR/resnet56.bin" --depth 56

python scripts/export_resnet_weights.py "$FHE_ONNX_DIR/resnet110_cifar10_train.onnx" "$FHE_WEIGHTS_DIR/resnet110.bin" --depth 110
```

### 4.2 其他 CNN

```bash
python scripts/export_cnn_weights.py "$FHE_ONNX_DIR/vgg11_cifar10.onnx" "$FHE_WEIGHTS_DIR/vgg11.bin" --family vgg11

python scripts/export_cnn_weights.py "$FHE_ONNX_DIR/vgg16_cifar10.onnx" "$FHE_WEIGHTS_DIR/vgg16.bin" --family vgg16

python scripts/export_cnn_weights.py "$FHE_ONNX_DIR/mobilenet_cifar10.onnx" "$FHE_WEIGHTS_DIR/mobilenet.bin" --family mobilenet

python scripts/export_cnn_weights.py "$FHE_ONNX_DIR/squeezenet_cifar10.onnx" "$FHE_WEIGHTS_DIR/squeezenet.bin" --family squeezenet

python scripts/export_cnn_weights.py "$FHE_ONNX_DIR/alexnet_cifar10.onnx" "$FHE_WEIGHTS_DIR/alexnet.bin" --family alexnet
```

每次导出都会生成 `NAME.bin` 和 `NAME.json`；JSON 保存权重形状、校验和以及模型信息。
CNN manifest 另外包含源 ONNX SHA-256 和算子清单。保留两个文件，验证脚本会使用 manifest。

`.bin` 内保存普通权重和模型描述，不是已做 CKKS 编码的 GPU 明文。
GPU 明文编码在运行阶段进行。已有 ONNX 不需要重新训练模型，也不需要 FHE 编译器。

## 5. 运行 ResNet20/32/44/56/110

下面每个模型运行测试集第 0 张图，保存完整日志。设置 pipefail 后，管道状态可反映推理失败：

```bash
set -o pipefail
```

依次执行需要的模型。下面整组命令在子 shell 中遇到首个失败即停止，不结束当前交互终端；
只运行某个模型时，复制对应的一条完整命令即可：

```bash
(
set -euo pipefail

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --weights "$FHE_WEIGHTS_DIR/resnet20.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  2>&1 | tee "$FHE_RUN_DIR/resnet20.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --weights "$FHE_WEIGHTS_DIR/resnet32.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  2>&1 | tee "$FHE_RUN_DIR/resnet32.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --weights "$FHE_WEIGHTS_DIR/resnet44.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  2>&1 | tee "$FHE_RUN_DIR/resnet44.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --weights "$FHE_WEIGHTS_DIR/resnet56.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 12 \
  2>&1 | tee "$FHE_RUN_DIR/resnet56.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --weights "$FHE_WEIGHTS_DIR/resnet110.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 24 \
  2>&1 | tee "$FHE_RUN_DIR/resnet110.log"
)
```

遇到非零返回码或 `FAILED` 时先检查该模型日志，不要将失败结果计入通过或性能统计。

ResNet 固定使用 N=65536（2^16）、16384 槽、32 位后端，normal/Boot scale 目标为 40/58。
每个 ReLU 执行两次普通 Boot；预期每图 Boot 次数为 38/62/86/110/218。
`--baby-step 0` 是默认的逐层 BSGS 策略，不需要手工指定。

`--relu-range` 是实数激活归一化范围。20/32/44 使用 10，56/110 的示例使用 12/24，
与现有验证脚本和已有检查一致；这些范围不是任意 checkpoint 或完整数据集的保证。
更换权重后可先检查明文路径：

```bash
python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --weights "$FHE_WEIGHTS_DIR/resnet110.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode reference --start 0 --count 6 --relu-range 24
```

`reference` 执行明文多项式与精确 ReLU 参考，不做密文推理。
程序遇到多项式路径激活超界会失败，不会裁剪输入或自动增大范围。
调整范围会改变激活近似误差，应重新检查最终明文/密文输出。

## 6. 运行 VGG11/16、MobileNet、SqueezeNet、AlexNet

本节不需要 PyTorch 或额外参考 JSON，只需要第 4 节导出的权重和 CIFAR 二进制数据。
仍从仓库根目录、同一个已设置路径的 Bash 终端执行：

```bash
(
set -euo pipefail

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --weights "$FHE_WEIGHTS_DIR/vgg11.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  --log-degree 17 --slots 65536 --channel-tile "$FHE_CNN_TILE" \
  --diagonal-chunk 64 --baby-step 128 \
  2>&1 | tee "$FHE_RUN_DIR/vgg11.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --weights "$FHE_WEIGHTS_DIR/vgg16.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  --log-degree 17 --slots 65536 --channel-tile "$FHE_CNN_TILE" \
  --diagonal-chunk 64 --baby-step 128 \
  2>&1 | tee "$FHE_RUN_DIR/vgg16.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --weights "$FHE_WEIGHTS_DIR/mobilenet.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  --log-degree 17 --slots 65536 --channel-tile "$FHE_CNN_TILE" \
  --diagonal-chunk 64 --baby-step 128 \
  2>&1 | tee "$FHE_RUN_DIR/mobilenet.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --weights "$FHE_WEIGHTS_DIR/squeezenet.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  --log-degree 17 --slots 65536 --channel-tile "$FHE_CNN_TILE" \
  --diagonal-chunk 64 --baby-step 128 \
  2>&1 | tee "$FHE_RUN_DIR/squeezenet.log"

python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --weights "$FHE_WEIGHTS_DIR/alexnet.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --log-format "$FHE_LOG_FORMAT" --encode-mode auto --start 0 --count 1 --scale-bits 40 --relu-range 10 \
  --log-degree 17 --slots 65536 --channel-tile "$FHE_CNN_TILE" \
  --diagonal-chunk 64 --baby-step 128 \
  2>&1 | tee "$FHE_RUN_DIR/alexnet.log"
)
```

五个模型使用 N=131072（2^17）、65536 槽、32 位后端，normal/Boot scale 目标为 40/58。
`N` 是环维度；CKKS 最大槽数是 `N/2`，这里不是 131072 个槽。
预期每图 Boot 次数依次为 VGG11 16、VGG16 26、MobileNet 54、SqueezeNet 36、AlexNet 14。

修改 `--start` 可选择起始图片，索引从 0 开始；`--count` 表示连续图片数，
要求 `start + count <= 10000`，且数据文件必须含有对应记录。
首次建议各跑一张，通过后再扩大样本数量。

可选：先用 VGG11 的第一层检查布局与密文线性层：

```bash
python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --weights "$FHE_WEIGHTS_DIR/vgg11.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --stop-after 1 --count 1 --channel-tile "$FHE_CNN_TILE" \
  --diagonal-chunk 64 --baby-step 128
```

检查更大数据范围的精确 ReLU 明文激活时可以运行：

```bash
python scripts/run_inference.py --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --weights "$FHE_WEIGHTS_DIR/vgg11.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode calibrate --log-format json --count 11 --relu-range 10
```

`calibrate` 只运行精确 ReLU 明文路径；`reference` 则还运行相同多项式的明文路径。
两者不构造密文计算上下文。校准样本的范围不代表整个测试集的范围。

## 7. 查看密文与原始明文输出的误差

### 7.1 默认输出与编码模式

第 5–6 节通过 `scripts/run_inference.py` 调用 C++，默认只显示运行参数、资源选择、
每个类别一行的明文与 FHE 解密结果、误差以及三段时间和派生合计。
逐层 CHECK/ACTIVATION/LINEAR/NODE_TIMING 和内部准备日志默认收起，原有校验仍然执行。

| 参数 | 作用 |
|---|---|
| `--log-format human` | 默认：参数、资源、输出和时间分行显示 |
| `--log-format json` | PARAM、RESOURCES、RESULT、TIMING 机器记录 |
| `--verbose 1` | 打开逐层及内部诊断日志，默认 0 |
| `--encode-mode auto` | 默认：保守显存预算足够时整网预编码，不够时选择 stream |
| `--encode-mode pre` | 强制预编码全部用到的权重并驻留；不足时明确失败 |
| `--encode-mode stream` | 先备齐所有实际用到的密钥，推理期间逐层/逐块编码权重并释放 |

以下资源前置/缓存约定面向 `--mode full`；ResNet 的 conv/stem/blocks/projection 等局部模式仍是逐层调试路径。

模型旋转请求先按最高所需 level 合并生成，再准备 Boot 专用切换密钥、旋转密钥和常量，
复用已有兼容 key。这里的“全部旋转 key”指模型实际用到的集合，不枚举所有可能距离。
推理阶段如果发现未提前准备的 key 会失败。同进程的后续图片复用密钥，
pre 模式也复用整网权重明文。

权重估算包含 RMM 固定块分配的向上取整。auto 预算使用 CUDA 空闲和 RMM CUDA 池
尚未使用的容量，再预留 4 GiB 工作空间；固定块池中的可复用空块未全部计入预算，
因此选择偏保守，实际分配失败仍会报告失败。不能用 ONNX 文件大小推断展开后 CKKS 权重的驻留大小。

JSON 日志中每张图有一条 RESULT，成功退出有 GPU_ONLY_PASS，最后有整次运行的 TIMING。
下面是 JSON RESULT 的主要字段；默认 human 已直接显示对应结果：

`RESULT` 中的主要字段：

| 字段 | 含义 |
|---|---|
| `fhe_logits` | 密文推理结束后解密得到的 10 维输出 |
| `relu_logits` | 相同输入、权重的精确 ReLU 明文网络输出 |
| `polynomial_logits` | 与密文使用相同激活多项式的明文输出 |
| `fhe_vs_relu_logits` | **你关心的端到端最大绝对误差：FHE 解密输出对精确 ReLU 明文输出** |
| `fhe_vs_polynomial_logits` | 最终 FHE 输出对同多项式明文参考的最大绝对误差 |
| `max_abs_error` | 所有已检查层/槽的同多项式实部误差和虚部残留的最大值 |
| `prediction` / `relu_top1` | 密文 / 精确 ReLU 明文预测类别 |
| `label` | CIFAR-10 的真实标签，准确率应与它比较 |
| `boot_count` | 当前图片的实际 Boot 调用次数 |
| `online_ms` | 分段密文算子求值的同步墙钟之和 |
| `weights_ms` / `keys_ms` | 当前图片的权重准备 / 补充密钥时间 |
| `input_ms` / `checks_ms` | 输入编码加密 / 逐层解密校验时间 |
| `image_wall_ms` | 包含准备、明文参考和校验的单图墙钟 |

这些输出是 logits（类别得分），不是百分比。最大得分对应预测类别。
输出误差和分类准确率是不同指标；top-1 相同不代表全部数值误差足够小。

### 7.2 直接打印明文、密文和两种误差统计

默认 human 已显示逐类输出、最大与平均绝对误差。使用下面的自动解析时，先选择 JSON
并换一个日志目录，再重跑第 5–6 节中需要的命令，无需重新导出权重：

```bash
export FHE_LOG_FORMAT=json
export FHE_RUN_DIR="$PWD/validation/json-01"
mkdir -p "$FHE_RUN_DIR"
```

以下代码读取该 JSON 日志目录，每张图输出两个向量、
最大绝对误差和平均绝对误差。统计基于未舍入的十个 logits：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

logs = sorted(Path(os.environ["FHE_RUN_DIR"]).glob("*.log"))
if not logs:
    raise SystemExit("No logs found; run sections 5 and 6 first.")
for path in logs:
    text = path.read_text()
    if not any(line.startswith("GPU_ONLY_PASS ") for line in text.splitlines()):
        raise SystemExit(f"Incomplete/failed run; inspect {path}")
    for line in text.splitlines():
        if not line.startswith("RESULT "):
            continue
        row = json.loads(line[7:])
        plain = row.get("relu_logits", [])
        cipher = row.get("fhe_logits", [])
        if len(plain) != 10 or len(cipher) != 10:
            continue
        errors = [abs(a - b) for a, b in zip(cipher, plain)]
        print(f"{path.stem} image={row['image']}")
        print("  plaintext:", plain)
        print("  decrypted:", cipher)
        print(f"  max_abs_error={max(errors):.9g}, mean_abs_error={sum(errors)/10:.9g}")
        print(f"  top1: cipher={row['prediction']}, plaintext={row['relu_top1']}, label={row['label']}")
PY
```

最大绝对误差是 `max_i |cipher_i - plain_i|`，平均绝对误差是十个差值绝对值之和除以 10。
如果统计多张图的总体指标，应对所有图片的输出一起计算，并说明样本数量。

当前逐层 `1e-2` 门槛约束的是 **FHE 对同多项式参考**，不是 FHE 对精确 ReLU 的端到端总误差。
程序输出 `PASS` 不等于 `fhe_vs_relu_logits <= 0.01`。若你的验收目标是后者，
需要额外检查这个字段；多项式近似造成的差异也计入总误差。

### 7.3 三段时间如何与 time 端到端对应

最终三项互不重叠，覆盖整次进程运行；最后一行是三项相加的派生合计：

| 控制台名称 / TIMING 字段 | 包含内容 |
|---|---|
| Model resources / process overhead / `model_prepare_ms` | 文件加载、参数/上下文、基础乘法与共轭密钥、模型旋转密钥、权重规划、pre 权重预编码，以及其余进程启动、收尾、资源析构和外层入口开销 |
| Boot resources / `boot_prepare_ms` | Boot 专用稠密/稀疏切换密钥、Boot 旋转密钥、CtS/StC/EvalMod 常量与 ReLU 多项式准备 |
| Inference stage, all images / `inference_ms` | 输入读取/编码/加密、完整 FHE 图（含残差 Copy）、明文参考、逐层解密校验及结果输出；stream 权重编码也计入此项 |
| Sum / end-to-end wall / `total_ms` | 三项之和，是派生合计，不是第四种独立计时 |

`--count 1` 时第三项对应一张图；多张图时为所有图片的阶段耗时之和。
第三项包含校验，不声称是纯 GPU kernel 或剔除验证后的净计算时间。
第一项采用总墙钟减去第二、三项，确保所有进程开销有归属且没有重复计时：

`model_prepare_ms + boot_prepare_ms + inference_ms = total_ms`

可以这样核对同一次运行：

```bash
time -p python scripts/run_inference.py \
  --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --weights "$FHE_WEIGHTS_DIR/resnet20.bin" --cifar "$FHE_CIFAR_BIN" \
  --mode full --count 1 --encode-mode auto
```

将合计毫秒除以 1000，与外部 time 的 real 对照，而不是 user/sys CPU 时间。
Linux 外层入口从进程出生时间计时，包含 Python 自身和子进程 CUDA 库加载/退出；
两者允许约一个系统时钟 tick、外部显示舍入及最后打印的微小差额。

原生 C++ 入口也能直接运行，它的三项覆盖 main 到资源析构；
main 前动态库加载、main 后运行时退出不在原生合计内。
需要完整端到端口径时使用第 5–6 节的外层入口。

`online_ms` 作为 JSON 诊断字段保留旧分段求值边界；它没有覆盖全部输入、校验、
实时编码或旧区间外图操作，不能与新的第三项混称。

## 8. 可选：独立明文参考、分阶段验证与性能复测

直接运行第 5–6 节不需要本节。下面使用的参考 JSON 都从你自己的 ONNX 和 CIFAR 数据生成，
不要求已有 `validation/` 文件。

### 8.1 为五个 CNN 生成独立明文参考

参考脚本直接用 PyTorch CPU float64 重放源 ONNX 的精确 ReLU 网络。
Ubuntu x86_64、Python 3.10/3.11 可安装 CPU 版本；轮子由 [PyTorch 官方 CPU 索引](https://download.pytorch.org/whl/cpu/torch/)提供：

```bash
python -m pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
mkdir -p "$FHE_RUN_DIR/reference"

python scripts/cnn_float_reference.py --source "$FHE_ONNX_DIR/vgg11_cifar10.onnx" --cifar "$FHE_CIFAR_BIN" --output "$FHE_RUN_DIR/reference/float-vgg11.json" --count 11

python scripts/cnn_float_reference.py --source "$FHE_ONNX_DIR/vgg16_cifar10.onnx" --cifar "$FHE_CIFAR_BIN" --output "$FHE_RUN_DIR/reference/float-vgg16.json" --count 11

python scripts/cnn_float_reference.py --source "$FHE_ONNX_DIR/mobilenet_cifar10.onnx" --cifar "$FHE_CIFAR_BIN" --output "$FHE_RUN_DIR/reference/float-mobilenet.json" --count 11

python scripts/cnn_float_reference.py --source "$FHE_ONNX_DIR/squeezenet_cifar10.onnx" --cifar "$FHE_CIFAR_BIN" --output "$FHE_RUN_DIR/reference/float-squeezenet.json" --count 11

python scripts/cnn_float_reference.py --source "$FHE_ONNX_DIR/alexnet_cifar10.onnx" --cifar "$FHE_CIFAR_BIN" --output "$FHE_RUN_DIR/reference/float-alexnet.json" --count 11
```

ONNX 或 CIFAR 文件变化后应重新导出/生成对应参考，脚本会核对哈希。
这里的 11 张图覆盖索引 0–10；验证数量不能超过已经生成的参考数量。
本节验证器从图 0 开始，参考也保持 `--start 0`（默认值）。

### 8.2 一条命令分阶段验证全部五个 CNN

先完成 8.1，再运行：

```bash
python scripts/validate_cnn.py \
  --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --models "$FHE_WEIGHTS_DIR" --cifar "$FHE_CIFAR_BIN" \
  --reference-dir "$FHE_RUN_DIR/reference" \
  --output "$FHE_RUN_DIR/cnn-staged" \
  --count 3 --relu-range 10 --channel-tile "$FHE_CNN_TILE"
```

它按每个模型的首层 → 模块 → 整网顺序验证，首个失败即停止，保留日志和退出码。
批量脚本使用 JSON、verbose 和外层计时入口，默认 auto 选择预编码或实时编码；可用 --encode-mode 显式覆盖。
默认覆盖五个 family；只检查一个时增加 `--families alexnet`。
整网结果还与独立 PyTorch 参考比较，C++ 精确 ReLU logits 的一致性门槛为 `1e-7`。

### 8.3 一条命令分阶段验证整个 ResNet 系列

先导出全部五个 ResNet 权重，再运行：

```bash
python scripts/validate_resnet.py \
  --binary "$FHE_BUILD_DIR/example/resnet_cifar" \
  --models "$FHE_WEIGHTS_DIR" --cifar "$FHE_CIFAR_BIN" \
  --output "$FHE_RUN_DIR/resnet-staged" --count 3
```

该脚本先检查编码、Boot、ReLU、首层/残差块/投影，再跑 ResNet20/32/44/56/110；
自动为 56/110 选择范围 12/24。此命令使用 C++ 内部明文参考，不依赖 CNN 的参考 JSON，
也不意味着新增了 ResNet ONNX 独立重放结果。

### 8.4 五个 CNN 的重复 online 采样

先完成 8.1，使用新输出目录：

```bash
python scripts/benchmark_cnn_online.py \
  --binary "$FHE_BUILD_DIR/example/cnn_cifar" \
  --models "$FHE_WEIGHTS_DIR" --cifar "$FHE_CIFAR_BIN" \
  --reference-dir "$FHE_RUN_DIR/reference" \
  --output "$FHE_RUN_DIR/cnn-online-01" --label n17-optimized \
  --count 6 --warmup 1 --channel-tile "$FHE_CNN_TILE" \
  --diagonal-chunk 64 --baby-step 128
```

每个模型运行图 0–5：图 0 预热，图 1–5 统计均值、中位数、最小值、最大值和原始样本。
脚本记录源码（包括 .cuh）、二进制、共享库、模型、数据和参数哈希，以及节点拆分。
已有 `report.json` 时会拒绝覆盖；复跑请换目录。只测一个模型可增加 `--families vgg11`。
该脚本默认采用 auto，保留旧 online_ms 及新增三段 TIMING；固定旧实时编码口径时显式使用 --encode-mode stream。
固定 channel tile、diagonal chunk、baby step 和密码参数再作性能对比，不并行启动多个 GPU 推理任务。

ResNet 可将第 5 节某条命令改为 `--count 6`，自行排除图 0 后统计图 1–5。
`validate_resnet.py` 的现有报告汇总为均值，不提供上述 CNN 脚本的全部中位数/节点统计。
每次验证和采样都建议使用新的 `FHE_RUN_DIR`，避免混淆不同运行的日志。

## 9. 已验证范围与结果口径

本分支的 N17 优化曾在 RTX 5090 上完成五个 CNN 各版本 1 次预热 + 5 次正式采样，
基线和优化版共 60 次整网推理通过当时的逐层/参考校验。正式样本的 online 中位数如下；
使用 channel tile 512、diagonal chunk 64、baby step 128：

| 模型 | 基线 ms | 优化版 ms | 降幅 |
|---|---:|---:|---:|
| VGG11 | 4026.661 | 3480.267 | 13.57% |
| VGG16 | 6475.815 | 5592.689 | 13.64% |
| MobileNet | 5994.022 | 4448.262 | 25.79% |
| SqueezeNet | 3816.211 | 2672.788 | 29.96% |
| AlexNet | 5973.186 | 5411.373 | 9.41% |

前述 N17 优化记录中的 ResNet 回归只覆盖 ResNet20：旧 online 为 732.444 ms、38 次 Boot。
当前日志/资源流程的额外验证覆盖全部十个模型，见下文；两轮硬件、权重及计时口径应分别记录。

以上是历史固定参数/计时边界下的证据，不是任意 ONNX、硬件、分块或全测试集准确率保证。
新机器按本指南生成自己的报告即可，不依赖这些历史文件。

优化版五个 CNN 各 6 张图、ResNet20 一张图的最终输出总误差为：

| 模型 | FHE 对精确 ReLU 明文 logits 的最大绝对误差 |
|---|---:|
| ResNet20 | 0.002806 |
| VGG11 | 0.105271 |
| VGG16 | 0.257558 |
| MobileNet | 0.163078 |
| SqueezeNet | 0.087306 |
| AlexNet | 0.050112 |

这些误差包含激活多项式近似；不能用同多项式逐层 `1e-2` 校验通过替代这一总误差要求。
已测样本 top-1 一致，也不构成完整 CIFAR-10 准确率无损的结论。

### 三段计时与默认优化路径的额外验证

日志/资源流程在 RTX PRO 6000 Blackwell Server Edition（97887 MiB 显存）、
CUDA 12.8、GCC 11 环境实测。默认 auto 在 ResNet20 上选择整网预编码，
估算权重缓存约 75.93 GiB；推理阶段两张图的 weights_ms 和 keys_ms 均为 0。
强制 stream 的两张图也没有新增 key，权重编码时间完整计入第三项。

ResNet20 一张图、默认简洁输出的示例（ms）：

| 项目 | 实测 |
|---|---:|
| 模型资源及进程开销 | 9063.622 |
| Boot 资源 | 3032.101 |
| 单图推理阶段（含输入、校验与输出） | 1054.829 |
| 三项合计 | 13150.551 |

该次与 GNU time 的差额为 0.551 ms。
十模型矩阵各次对照的最大绝对差额约 11.1 ms，符合系统时钟粒度及外部显示舍入范围。

验证还包括 ResNet20–110 整网、全部五个 CNN 整网、N16 线性层与 N17 Boot/ReLU、
CNN 预编码前缀的两图复用、原 ResNet 分阶段脚本以及 CNN 性能脚本兼容性。
这些是有限样本的功能/计时校验，不是全测试集准确率或完整参数空间的最优性证明。
默认使用当前已验证的优化配置：auto、CNN channel tile 512、diagonal chunk 64、
baby step 128；ResNet 保留默认逐层 BSGS 策略和本分支优化内核。
需要固定对比可显式选择 pre/stream 和分块参数。

原始记录在本次工作区的 `validation/runtime-20261029/`，
其中 `attempt02/report.json`、`final-resnet/report.json`、
`resnet-staged-final/report.json` 与 `default-optimal/report.json` 分别保存矩阵、
最终缓存/计时、分阶段和脚本默认值校验。新环境无需预先拥有这些历史文件。

## 10. 常见问题与文件位置

| 现象 | 检查方法 |
|---|---|
| `nvcc: command not found` | 安装 CUDA 开发工具包或设置实际 CUDA bin 路径；仅有驱动不够 |
| CMake 版本小于 3.24 | 激活第 1 节虚拟环境，确认使用其中的 cmake |
| `Unsupported gpu architecture 'compute_120'` | RTX 5090 需要支持该目标的 CUDA；其他 GPU 使用实际架构编号 |
| RMM clone/FetchContent 下载失败 | 按第 3 节准备 RMM 22.12 头文件并显式传入路径 |
| 找不到 tommath 或 spdlog | 安装 `libtommath-dev`、`libspdlog-dev`；本指南使用默认 `USE_GMP=OFF` |
| 不认识 --encode-mode / --log-format / --timing-file | 拉取代码后重编译两个入口，确认 --binary 指向当前构建 |
| 找不到 `libcheddar.so` | 设置同一构建目录的 LD_LIBRARY_PATH，避免加载旧构建库 |
| 找不到 ONNX、.bin 或 CIFAR 文件 | 检查绝对路径；--weights 接收导出的 .bin，--cifar 接收实际二进制文件 |
| `operator list does not match...` / BN 未折叠 | 使用匹配 family 的受支持 CIFAR-10 推理图；不能仅改 family 强行适配其他图 |
| `fixed ReLU range exceeded` / `exceeds fixed ReLU range` | 核对权重、预处理和明文激活范围，按实际 checkpoint 校准并重新验证输出误差 |
| 显存或 CPU 内存不足 | 降低 --channel-tile / --diagonal-chunk，记录新配置后重测；不要把参数变化混入原性能对照 |
| 验证器缺少 `float-FAMILY.json` | 先完成 8.1，C++ 直接推理不要求这些文件 |
| 超过参考图数或源数据哈希不匹配 | 使用同一 ONNX/CIFAR，从图 0 重新生成足够数量的参考 |
| JSON 日志提前退出且没有 `GPU_ONLY_PASS` | 检查日志中的首个 FAILED/异常及真实退出码，不将已有部分 RESULT 当整网成功 |

`models/`、`validation/` 和 `build/` 默认不提交到 GitHub。
克隆分支得到的是源码与脚本，ONNX、CIFAR、导出权重、参考和运行产物由你在本机准备。
推理过程中解密只用于校验，不把明文结果反馈给密文图。

更多实现和历史细节见 [ResNet 说明](example/README.resnet.md)、
[CNN/N17 说明](example/README.cnn.md)。项目许可证见 [LICENSE](LICENSE)。
