# Cheddar ResNet 编译与运行

本分支通过 Cheddar C++/CUDA API 在 NVIDIA GPU 上执行 CIFAR-10 ResNet 的 CKKS 密文推理，支持 ResNet20、32、44、56 和 110，无需 FHE 编译器。

## 1. 环境准备

在配有 NVIDIA GPU 的 Linux 机器上编译运行，需要：

- 可用的 NVIDIA 驱动和 CUDA Toolkit 11.8 或更新版本；CUDA 版本必须支持实际 GPU 的架构。
- CMake 3.24 或更新版本、支持 C++17 的 C++ 编译器、Git 和 GMP。
- Python 3：仅导出权重和运行验证脚本时需要；导出依赖 NumPy，ONNX 输入另需 ONNX。
- 足够的 GPU 显存；显存占用取决于模型和参数，本指南不保证特定显存容量能运行所有模型。

Ubuntu 系统依赖：

```bash
sudo apt update
sudo apt install -y build-essential git libgmp-dev python3 python3-venv
nvcc --version
cmake --version
nvidia-smi
```

CUDA Toolkit 和 CMake 如未满足版本要求，需先自行安装。

## 2. 获取代码并编译

```bash
git clone --branch cheddar_resnet https://github.com/zggl404/cheddar-fhe.git
cd cheddar-fhe

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GMP=ON \
  -DENABLE_EXTENSION=ON \
  -DBUILD_UNITTEST=OFF \
  -DBUILD_RESNET_EXAMPLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target resnet_cifar --parallel 8
```

可执行文件为 `build/example/resnet_cifar`。`native` 用于在目标 GPU 机器上编译；交叉编译或自动探测失败时，将其替换为目标 GPU 的 CUDA 架构编号，并使用支持该架构的 CUDA Toolkit。

首次配置会通过 CMake FetchContent 下载 RMM 22.12 及其依赖，需要联网。如果已有 RMM 22.12 头文件和系统 spdlog，可在配置时增加：

```bash
-DCHEDDAR_RMM_INCLUDE_DIR=/path/to/rmm/include
```

此选项要求系统提供可被 `find_package(spdlog)` 找到的 spdlog。

## 3. 准备权重和 CIFAR-10 数据

仓库不附带模型权重或 CIFAR-10 数据，运行程序也不会自动下载。请准备 CIFAR-10 **二进制格式**的测试文件 `test_batch.bin`，不是 Python pickle 版本。

创建权重导出环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy onnx
```

以下路径均需替换为本机实际路径。可选择一种受支持的权重来源：

```bash
# AE ResNet20 已融合权重目录（包含 conv1_reparam.weight 等 NumPy 文件）
python3 scripts/export_resnet_weights.py \
  /path/to/resnet20_fused models/resnet20.bin

# 或：固定结构、已折叠 BatchNorm 的 CIFAR ResNet ONNX
python3 scripts/export_resnet_weights.py \
  /path/to/resnet20_cifar10_train.onnx models/resnet20.bin --depth 20
```

ONNX 导出器匹配固定的层命名及结构，不支持任意 ResNet ONNX，也不会自动折叠 BatchNorm。其他深度使用对应的 ONNX 文件并设置 `--depth 32`、`44`、`56` 或 `110`，输出为 `models/resnet32.bin` 等。导出同时生成包含归一化参数、张量清单及 SHA-256 的同名 `.json` 文件。C++ 运行程序无需安装 NumPy 或 ONNX。

## 4. 运行 ResNet

以下命令均在仓库根目录执行。先用单张图像检查卷积，再运行完整 ResNet20：

```bash
build/example/resnet_cifar --mode conv \
  --weights models/resnet20.bin --cifar /path/to/test_batch.bin

build/example/resnet_cifar --mode full \
  --weights models/resnet20.bin --cifar /path/to/test_batch.bin \
  --start 0 --count 1 --relu-range 10
```

模型深度从权重文件读取，无需向运行程序传入 `--depth`。完整模型的运行参数建议与现有验证脚本一致：

| 模型 | 权重文件 | `--relu-range` |
| --- | --- | ---: |
| ResNet20 | `models/resnet20.bin` | 10 |
| ResNet32 | `models/resnet32.bin` | 10 |
| ResNet44 | `models/resnet44.bin` | 10 |
| ResNet56 | `models/resnet56.bin` | 12 |
| ResNet110 | `models/resnet110.bin` | 24 |

例如：

```bash
build/example/resnet_cifar --mode full \
  --weights models/resnet110.bin --cifar /path/to/test_batch.bin \
  --start 0 --count 1 --relu-range 24
```

这些激活范围来自有限样本检查，不构成全数据集范围保证。程序会检查激活范围，超出时退出，不会自动裁剪或放宽范围。

常用参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--mode` | `full` | `reference`、`encode`、`boot`、`relu`、`conv`、`stem`、`blocks`、`projection` 或 `full` |
| `--start` | `0` | 起始图像索引 |
| `--count` | `1` | 连续处理的图像数量 |
| `--scale-bits` | `40` | 支持 `40`、`48`；会同时选择配套密码参数 |
| `--relu-range` | `10` | 激活归一化范围 |
| `--baby-step` | `0` | 自动选择逐层 BSGS 策略；可用 1–128 内的 2 的幂覆盖 |
| `--max-blocks` | `1` | `blocks` 模式的残差块数量上限 |
| `--probe-stage` | `2` | `projection` 模式选择 stage 2 或 3 |

无需权重和数据即可运行基础检查：

```bash
build/example/resnet_cifar --mode encode
build/example/resnet_cifar --mode boot
build/example/resnet_cifar --mode relu
```

默认密码配置为 32 位后端、环维度 65536、固定 16384 slots、普通/Boot scale 目标 40/58 bits、Boot 输出 level 8。程序输出 `PARAM` 记录实际解析的参数。实现细节见 [ResNet 技术说明](example/README.resnet.md)。

## 5. 验证结果与计时

每张图像输出 `RESULT` JSON，包含预测、数值误差及分项计时；同时输出逐层检查与 GPU 编码审计。范围、数值或编码检查失败会以非零退出码终止。

程序将密文结果与相同多项式的明文参考逐层比较，实部和虚部误差阈值均为原始实数单位下的 `1e-2`，并额外报告精确 ReLU 参考的预测与近似误差。

准备全部五种深度的权重后，可执行分阶段验证：

```bash
python3 scripts/validate_resnet.py \
  --binary build/example/resnet_cifar \
  --models models \
  --cifar /path/to/test_batch.bin \
  --output validation/resnet \
  --count 10
```

脚本先检查编码、Boot、ReLU、卷积和残差块，再运行五个完整模型；遇到第一个失败即停止，保存日志、退出码、哈希和 `report.json`。脚本还要求完整模型的密文预测与多项式、精确 ReLU 两种参考预测一致。

如需观察预热后的耗时，可在同一进程处理 11 张图像：

```bash
build/example/resnet_cifar --mode full \
  --weights models/resnet20.bin --cifar /path/to/test_batch.bin \
  --start 0 --count 11 --relu-range 10
```

程序不会自动剔除预热结果；统计时需自行排除图像 0。权重准备、密钥、输入、在线计算和验证分别计时。当前计时包含同步和验证插桩，属于正确性验证运行，不能直接视为纯 GPU kernel 或稳态端到端推理性能。

## License

本项目使用 [MIT License](LICENSE)。
