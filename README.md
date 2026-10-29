# Cheddar：CIFAR-10 密文推理

支持 ResNet20/32/44/56/110、VGG11/16、MobileNet、SqueezeNet 和 AlexNet。
你只需准备受支持的 CIFAR-10 ONNX 模型，以及 `test_batch.bin`。
流程是：**编译 → 逐个导出权重 → 运行模型**。

## 1. 准备环境并编译

密文推理需要 Linux、NVIDIA GPU 和 CUDA 开发工具包。下面以 Ubuntu、
Python 3.10/3.11、CUDA 12.8、Blackwell GPU 为例；其他 GPU 的配置见[详细文档](docs/inference.md)。

```bash
git clone --branch feature/cnn-n17 https://github.com/zggl404/cheddar-fhe.git
cd cheddar-fhe

# root 用户可以去掉 sudo。
sudo apt-get update
sudo apt-get install -y build-essential git libtommath-dev libspdlog-dev python3-venv

python3 -m venv ../cheddar-env
source ../cheddar-env/bin/activate
python -m pip install numpy==1.24.4 onnx==1.14.1 cmake==3.29.6

export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
nvidia-smi

git clone --depth 1 --branch branch-22.12 https://github.com/rapidsai/rmm.git ../cheddar-rmm
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCHEDDAR_RMM_INCLUDE_DIR="$(cd ../cheddar-rmm/include && pwd)" \
  -DBUILD_UNITTEST=OFF \
  -DBUILD_RESNET_EXAMPLE=ON \
  -DBUILD_CNN_EXAMPLE=ON
cmake --build build --parallel 12
```

重新登录后，在仓库根目录重新激活 `../cheddar-env`。更新 C++ 源码后需要重新编译。

## 2. 逐个导出权重

设置 ONNX 目录和 CIFAR 测试文件路径，改成你的实际位置：

```bash
export ONNX_DIR="/data/onnx"
export CIFAR_BIN="/data/cifar-10-batches-bin/test_batch.bin"
mkdir -p models
```

每条命令独立执行，只需导出你要运行的模型。文件名不同时，修改对应输入路径即可：

```bash
python scripts/export_resnet_weights.py "$ONNX_DIR/resnet20_cifar10_pre.onnx" models/resnet20.bin --depth 20
python scripts/export_resnet_weights.py "$ONNX_DIR/resnet32_cifar10_pre.onnx" models/resnet32.bin --depth 32
python scripts/export_resnet_weights.py "$ONNX_DIR/resnet44_cifar10_pre.onnx" models/resnet44.bin --depth 44
python scripts/export_resnet_weights.py "$ONNX_DIR/resnet56_cifar10_pre.onnx" models/resnet56.bin --depth 56
python scripts/export_resnet_weights.py "$ONNX_DIR/resnet110_cifar10_train.onnx" models/resnet110.bin --depth 110

python scripts/export_cnn_weights.py "$ONNX_DIR/vgg11_cifar10.onnx" models/vgg11.bin --family vgg11
python scripts/export_cnn_weights.py "$ONNX_DIR/vgg16_cifar10.onnx" models/vgg16.bin --family vgg16
python scripts/export_cnn_weights.py "$ONNX_DIR/mobilenet_cifar10.onnx" models/mobilenet.bin --family mobilenet
python scripts/export_cnn_weights.py "$ONNX_DIR/squeezenet_cifar10.onnx" models/squeezenet.bin --family squeezenet
python scripts/export_cnn_weights.py "$ONNX_DIR/alexnet_cifar10.onnx" models/alexnet.bin --family alexnet
```

导出会生成 `.bin` 和同名 `.json`；运行时使用 `.bin`。
模型须为本项目支持的 BN 已折叠 CIFAR-10 图，输入 `[1,3,32,32]`、输出 `[1,10]`。
同名 ImageNet 模型不能直接使用，具体结构限制见[模型与数据约定](docs/inference.md#模型与数据)。

## 3. 运行模型

所有模型使用同一个入口。默认读取 `build/` 下的程序和 `models/模型名.bin`，
采用当前优化默认配置，自动准备实际需要的密钥；显存足够时整网预编码，不足时实时编码。
普通运行不需要设置底层调优参数。

下面每条命令运行第 0 张图片，按需选择：

```bash
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model resnet32 --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model resnet44 --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model resnet56 --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model resnet110 --cifar "$CIFAR_BIN"

python scripts/run_inference.py --model vgg11 --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model vgg16 --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model mobilenet --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model squeezenet --cifar "$CIFAR_BIN"
python scripts/run_inference.py --model alexnet --cifar "$CIFAR_BIN"
```

选择图片或自定义权重路径：

```bash
python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" --start 5 --count 3
python scripts/run_inference.py --model vgg11 --cifar "$CIFAR_BIN" --weights /data/weights/vgg11.bin
```

重复采样仍使用同一入口。例如共运行 6 张，图 0 预热，后 5 张计算统计：

```bash
python scripts/run_inference.py --model vgg11 --cifar "$CIFAR_BIN" --count 6 --warmup 1
```

## 4. 看结果和时间

默认分行显示参数、资源模式、每个类别的精确 ReLU 明文输出、FHE 解密输出，
以及最大/平均绝对误差。逐层诊断默认收起，数值校验仍然执行；失败会返回非零状态。

计时只有一套口径，所有模型一致：

```text
端到端 = 模型资源准备及进程开销 + Boot资源准备 + 推理阶段

推理阶段 = 输入准备 + 运行期权重准备 + 密文求值
         + 明文参考 + 校验与输出 + 调度及其他开销

密文求值 = 线性层 + Boot执行 + 激活计算（不含Boot）+ 其他密文操作
```

预编码计入模型准备；实时编码计入推理阶段。Boot 准备与 Boot 执行分开统计。
三段总计覆盖所有图片，包括预热；每图均值只统计同一组正式图片。
所有时间为同步墙钟，单位 ms，包含相应区段的主机开销。

核对端到端耗时：

```bash
time -p python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN"
```

合计除以 1000 后与 `real` 对照，允许系统时钟粒度和最后打印/退出的小额差异。
机器可读日志加 `--log-format json`，字段定义见[统一计时说明](docs/inference.md#统一计时)。

FHE 对精确 ReLU 的误差包含激活多项式近似；逐层 `1e-2` 校验约束的是 FHE 对同多项式参考，
不能替代原始网络的输出误差或全测试集准确率。

底层参数、显存策略、分阶段验证和独立参考见[详细文档](docs/inference.md)。
`models/`、`build/`、`validation/` 不提交到 GitHub。许可证见 [LICENSE](LICENSE)。
