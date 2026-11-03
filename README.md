# Cheddar：CIFAR-10 密文推理

支持 ResNet20/32/44/56/110、VGG11/16、MobileNet、SqueezeNet 和 AlexNet。
你只需准备受支持的 CIFAR-10 ONNX 模型，以及 `test_batch.bin`。
流程是：**编译 → 逐个导出权重 → 运行模型**。

ResNet 卷积、Pooling 和 FC 的默认旋转组织已对齐到官方 AE 的已核实路径；
其他 CNN 复用其共享空间 Baby 的卷积分解。实现边界、服务器路径和验证记录见
[AE 式模型实现](docs/ae-models.md)。

## 1. 在已有 Docker 环境中编译

以下假设容器中的 CUDA、编译器和依赖已准备好，在仓库根目录执行。
非 Docker 环境的依赖安装放在本文末尾。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DBUILD_UNITTEST=OFF \
  -DBUILD_RESNET_EXAMPLE=ON \
  -DBUILD_CNN_EXAMPLE=ON \
  -DCHEDDAR_NVTX_TRACE=ON \
  -DCHEDDAR_CUDA_EVENT_TRACE=ON \
  -DCHEDDAR_PLAINTEXT_REFERENCE=OFF
cmake --build build --parallel 12 --target resnet_cifar cnn_cifar
```

`native` 针对容器可见的 GPU 编译；也可以指定实际架构。更新 C++ 源码或 CMake 选项后重新编译。
容器已有 RMM 22.12 头文件时，追加 `-DCHEDDAR_RMM_INCLUDE_DIR=/实际路径/include`，避免重复下载。

| CMake 选项 | 默认 | 作用 |
|---|---|---|
| `BUILD_RESNET_EXAMPLE` | `OFF` | 编译 ResNet 推理入口及对应检查程序 |
| `BUILD_CNN_EXAMPLE` | `OFF` | 编译 VGG/MobileNet/SqueezeNet/AlexNet 入口及对应检查程序 |
| `CHEDDAR_NVTX_TRACE` | `OFF` | 编译 NVTX 标记支持；运行时仍需打开采集，另外需要 `nsys` |
| `CHEDDAR_CUDA_EVENT_TRACE` | `OFF` | 编译 CUDA Event 计时支持；运行时仍默认关闭 |
| `CHEDDAR_PLAINTEXT_REFERENCE` | `ON` | 计算同多项式和精确 ReLU 明文参考、执行数值对照；性能测量可设为 `OFF` |
| `CHEDDAR_RMM_INCLUDE_DIR` | 空 | 指定已有 RMM 22.12 include；留空时通过 FetchContent 下载 |
| `ENABLE_EXTENSION` | `ON` | 编译 Boot、神经网络等扩展，两个推理入口需要它 |
| `BUILD_UNITTEST` | `ON` | 编译 `unittest/` 下的单元测试；示例目录还有自己的检查 target |
| `USE_GMP` | `OFF` | 大整数后端由 libtommath 切换为 GMP，需要相应开发库 |
| `CMAKE_BUILD_TYPE` | `Release` | 构建类型；基准测试使用 Release |
| `CMAKE_CUDA_ARCHITECTURES` | `60;61;70;75;80;86;89;90` | 编译目标 GPU 架构，建议按实际设备覆盖 |
| `CMAKE_CUDA_COMPILER` | 自动查找 | CUDA 编译器路径，必要时设 `/usr/local/cuda/bin/nvcc` |

顶层还声明了 `BUILD_TESTS`，但本项目自身测试 target 由 `BUILD_UNITTEST` 和示例开关控制，
不要把它当成统一测试开关。CUDA Toolkit 最低查找要求为 11.8，实际编译器还须支持目标 GPU。

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
python scripts/run_inference.py --model vgg11 --cifar "$CIFAR_BIN" --count 6 --warmup 1 --time-cpu on
```

## 4. 看结果和时间

默认三个计时开关全部关闭，只输出参数、Logits 和 Success。
按需独立添加 `--time-cpu on`、`--time-cuda-event on`、`--time-nvtx on`；
每种都报告 KeyswitchTime、NonKeyswitchTime、Hybrid，只有 CPU 方式额外报告端到端/准备/推理墙钟。
NVTX 方式需要 Nsight Systems，GPU 方式需要先启用对应 CMake 功能。
复制执行命令见[计时与全模型验证](docs/run-timing.md)，实现与边界见[三种计时方式](docs/timing-modes.md)。

密钥可通过 `--key-mode save|load --key-dir /data/keys/模型名` 预生成保存并在后续运行加载，
默认仍为 `generate`，详见[KMS 使用说明](docs/kms.md)。

需要拆开跨分类的融合操作时，组合使用
`--split-hybrid on --time-cpu on`。拆分后共同短基前置归 NonKeyswitch，Hybrid 为零，
其余步骤按语义归属。默认不拆分；拆分会增加运行时间，详见[诊断模式](docs/keyswitch-profiling.md#hybrid-拆分诊断模式)。
十个模型的开关、精度、时间和显存验证见[48 GB 实测报告](docs/keyswitch-kms-gpu48.md)。

默认显示参数、FHE Logits 和 Success。开启计时后输出相应方法的报告；
开启计时并加 `--verbose 1` 可查看详细诊断。默认 strict 模式下数值校验失败返回非零状态。

数值门槛默认 `--numerical-gate strict`，逐层实部误差或虚部残差超过 `0.01` 时停止。
需要先跑完整个网络以收集诊断数据时，使用 `--numerical-gate warn`：保留同一门槛，
超标打印 `NUMERICAL_WARNING`，包含图片、层名及实际误差，继续后续层和图片。
最终 top-1 对照不一致也记录警告；对应结果为 `status=NUMERICAL_FAILED`、`success=false`，不会标记为通过。
NaN/Inf、CUDA 错误、结构/密钥错误和明文激活校准范围越界等其他错误仍然终止。
两种模式都适用于 ResNet/CNN、直接运行 C++ 及 `run_inference.py`；修改后需重新编译两个推理入口。

明文参考可通过 CMake 关闭，适用于 ResNet 和 CNN 入口，默认 `ON`：

```bash
cmake -S . -B build -DCHEDDAR_PLAINTEXT_REFERENCE=OFF
cmake --build build --parallel 12 --target resnet_cifar cnn_cifar
```

关闭后不执行网络明文前向，`reference_ms=0`；日志标记 `plaintext_reference=false`，
参考输出、top-1 和比较误差为 `null`。保留密文结构检查、逐层有限值/虚部检查及最终解密预测。
参考对比及基于明文的激活范围检查不执行；需要正确性验证时改为 `ON` 并重新编译。
详细边界见[明文参考开关](docs/inference.md#明文参考开关)。

CPU 墙钟计时的阶段边界在所有模型中一致：

```text
端到端 = 模型资源准备及进程开销 + Boot资源准备 + 推理阶段

推理阶段 = 输入准备 + 运行期权重准备 + 密文求值
         + 明文参考 + 校验与输出 + 调度及其他开销

密文求值 = 线性层 + Boot执行 + 激活计算（不含Boot）+ 其他密文操作
```

预编码计入模型准备；实时编码计入推理阶段。Boot 准备与 Boot 执行分开统计。
三段总计覆盖所有图片，包括预热；每图均值只统计同一组正式图片。
上述 CPU 时间为同步墙钟，单位 ms，包含相应区段的主机开销。CPU 分类覆盖 FHE。
split 模式下 Event/NVTX 共用 KS 阶段 ID，分别报告 Event elapsed / GPU 阶段跨度及同次完整 Inference 剩余墙钟。
逐段独立比较命令见 [交叉验证](docs/keyswitch-spans.md#event-与-nvtx-独立交叉验证)；旧 Event 的 FHE 口径保留在 split off。

核对端到端耗时：

```bash
time -p python scripts/run_inference.py --model resnet20 --cifar "$CIFAR_BIN" --time-cpu on
```

合计除以 1000 后与 `real` 对照，允许系统时钟粒度和最后打印/退出的小额差异。
机器可读日志加 `--log-format json`，字段定义见[统一计时说明](docs/inference.md#统一计时)。

FHE 对精确 ReLU 的误差包含激活多项式近似；逐层 `1e-2` 校验约束的是 FHE 对同多项式参考，
不能替代原始网络的输出误差或全测试集准确率。

底层参数、显存策略、分阶段验证和独立参考见[详细文档](docs/inference.md)。
十个真实模型的全预编码时间、显存和输出误差见[96 GB GPU 实测报告](docs/gpu96-benchmark.md)。
`models/`、`build/`、`validation/` 不提交到 GitHub。许可证见 [LICENSE](LICENSE)。

## 5. 自动统计全部模型的时间

同时验证基线、Event、NVTX 和 CPU 四次独立运行：

```bash
python scripts/benchmark_timing.py \
  --build-dir build --models-dir models --cifar "$CIFAR_BIN" \
  --output validation/timing-all --count 6 --warmup 1 --encode-mode auto
```

默认处理全部十个模型，固定 `split-hybrid on`，逐模型生成临时密钥并使用 load 测量，结束后清理。
每个模型会核对四次运行的 Logits，以及 Event/NVTX 每个阶段的 ID 和归属；
`summary.csv` 汇总准备、推理和 KS 时间，详细命令与字段见[计时执行说明](docs/run-timing.md)。
`--models resnet20 resnet110` 可选择部分模型；`nsys` 不在 PATH 时追加 `--nsys /path/to/nsys`。

只需要基线与 NVTX 两次运行时，仍可使用下面的专用脚本。

准备好上面十个模型的 `.bin` 权重、CIFAR 文件、GNU `time` 和 Nsight Systems CLI。
脚本默认逐个处理全部十个模型，每个模型运行两次：无 NVTX/Event 的粗粒度墙钟基线，
以及独立的 NVTX 采集。两次都关闭逐操作 CPU 同步计时。
本批处理脚本默认开启 Hybrid 拆分，两次均采用相同的拆分路径，基线用于对照采集开销。

```bash
python scripts/benchmark_nvtx.py \
  --build-dir build --models-dir models --cifar "$CIFAR_BIN" \
  --output validation/nvtx-all-split \
  --count 6 --warmup 1 \
  --split-hybrid on
```

`--output` 必须是新目录。`nsys` 不在 PATH 时添加 `--nsys /实际路径/nsys`。
使用 `--models resnet20 vgg11` 可先检查少数模型；`--encode-mode auto|pre|stream` 控制编码方式。
默认 `auto`，采集运行固定采用基线实际选中的编码模式，确保配对一致。

密钥流程：添加 `--key-root /data/keys` 可复用 `/data/keys/模型名`。
没有相应包时，脚本先在自己创建的私有临时目录生成、保存并执行一次验证推理，
随后两次测量都使用 `load`；该模型结束或失败后清理临时密钥。
原本存在的密钥包不会删除。预生成阶段不计入这两次端到端；文件缓存可能已被预热，结果不代表冷盘读取。
默认临时密钥位于输出目录下，磁盘须能容纳一个模型的包，无需同时保存十套密钥。

输出 `summary.csv`、`summary.json` 和每模型的命令、日志、墙钟报告、Nsight SQLite 与分类明细。
单个模型失败会记录错误并继续后续模型，最终有失败则返回非零。

批处理也支持 `--numerical-gate warn`，例如在上面的命令末尾追加该选项，并使用新的输出目录。
即使密钥准备后的验证推理或基线发生有限数值超标，也会继续后续图片和 NVTX 采集，保存完整时间及 Logits。
汇总增加 `numerical_gate`、`numerical_passed`、`numerical_failures` 和 `diagnostic_only`；
`numerical_failures` 为检查失败次数，包含密钥验证、预热图片及配对不一致事件，并非失败图片数。
未通过逐层校验、top-1 对照或两次运行数值一致性的模型标记 `NUMERICAL_FAILED`、`diagnostic_only=true`。
这些时间仅供诊断。批处理在全部模型处理完成后仍返回非零，表示有数值未通过；
单独运行的 C++/`run_inference.py` 在 warn 模式完成时返回 0，是否数值通过须看结果字段。
原有的密钥加载、分类完整性和 kernel 重叠检查保持严格。

| 报告字段 | 口径 |
|---|---|
| `baseline_end_to_end_ms` / `profiled_end_to_end_ms` | C++ main 起点到模型/上下文释放完成，覆盖全部图片 |
| `*_time_real_ms` | `/usr/bin/time -p` 的完整模型进程 real，含加载/退出；采集时不含 nsys 导出与后处理 |
| `*_time_user_ms` / `*_time_sys_ms` | GNU time 的 CPU user/sys；不是 GPU 耗时 |
| `*_key_load_ms` | 全部 secrets/evaluation keys 的 manifest 校验、文件读取、checksum、GPU 分配及上传，汇总模型准备与 Boot 准备两个阶段 |
| `*_weight_preencode_ms` | 网络权重构建和预编码；stream 为 0，其运行期编码包含在 Inference |
| `*_inference_total_ms` | 全部图片的输入开始至校验/输出及末尾同步完成，含预热 |
| `*_inference_mean_ms` | 排除 warmup 后的单图 Inference 均值 |
| `*_other_prepare_and_teardown_ms` | 端到端扣除 key load、预编码、全部 Inference 的其余部分 |
| `nvtx_keyswitch_ms` | 同一组正式图片的 KS GPU 阶段跨度均值 |
| `fused_baby_inclusive_ms` | Baby 准备及关联 finish 子项，已包含在 KS 总量内 |
| `nvtx_keyswitch_kernel_diagnostic_ms` | 另存 KS kernel duration sum，供对照诊断 |
| `non_keyswitch_residual_ms` | **profiled Inference 均值 − 同次采集的 NVTX Keyswitch 均值** |
| `nvtx_other_kernel_ms` / `nvtx_hybrid_ms` | 另外保留纯 GPU Other 与 Hybrid，避免把剩余墙钟解释为纯 GPU NonKeyswitch |

这套相减口径观察整次推理扣除 KS GPU 阶段后的剩余时间，逐层起止时间见 [KS 阶段报告](docs/keyswitch-spans.md)。
剩余时间包含主机提交/等待、设备空闲、输入、明文参考（若开启）、解密校验、运行期编码及尚未拆开的 Hybrid。
它和纯 GPU Other 有不同含义，也不能用基线 Inference 减去另一次 NVTX 的 Keyswitch。
本脚本的 `--split-hybrid` 默认 `on`，Hybrid 必须为零；共同短基前置计入 NonKeyswitch。
需要测原融合路径时显式加 `--split-hybrid off`，只保留 kernel 诊断，主 KS 跨度和 residual 为 null。
拆分会改变执行路径并可能增加耗时，报告中保留所选模式。
普通 `run_inference.py` 入口的拆分默认仍为 `off`。
当前相减限定为无 kernel 重叠的执行，脚本发现重叠会拒绝发布这个差值。

两次各自满足：

```text
C++端到端 = key load + 权重预编码 + 全部Inference + 其余准备/收尾
```

key load 和预编码本来就是 Prepare 的子项，不要再与完整 Prepare 重复相加。
粗粒度 `wall.json` 标记 `timing_detail=coarse`，原有叶阶段字段未采集，不能把其中的 0 解释为没有对应工作。
`wrapper_wall_ms` 另存于 JSON，包含采集工具启动/导出；批处理总耗时还包含密钥预生成和分析。

### 导出 KS / NonKS 起止时间线

在仓库根目录执行，使用上面 NVTX 批处理生成的同一次采集结果：

```bash
ANALYSIS=validation/nvtx-all-split/resnet20/profiled/analysis

python3 scripts/keyswitch_timeline.py "$ANALYSIS/ks_segments.csv" \
  --activities "$ANALYSIS/gpu_activities.csv.gz" \
  --output validation/resnet20-timeline
```

`gpu_activities.csv.gz` 由 `run_inference.py` 的 NVTX 模式或 `benchmark_nvtx.py`
自动调用 `analyze_gpu_activity.py`，从 Nsight 导出的 `nvtx.sqlite` 中生成。
每行记录一个 kernel、memcpy 或 memset 的 GPU 起止时间，脚本直接读取压缩文件。

| 参数 | 是否必填 | 作用 |
|---|---|---|
| 第一个位置参数 `ks_segments.csv` | 是 | 每个 KS 阶段的 GPU 起止时间戳；不要使用 `ks_units.csv` |
| `--output DIR` | 是 | 自动创建输出目录；重新运行会覆盖其中的三个报告文件 |
| `--activities FILE` | 否 | 使用同次采集的首尾 GPU 活动确定每图 T0/T1，标记 `scope=gpu_activity_envelope` |
| `--bounds FILE` | 否 | 使用自行提供的完整 Inference 边界 CSV，标记 `scope=inference`；与 `--activities` 互斥 |

输出文件：

- `timeline.txt`：按 `T0 → NonKS_0 → [KS_0] → NonKS_1 → … → T1` 逐行展开，
  显示每段绝对起止时间戳、相对 T0 的时间、时长及 KS 的层名和类型。
- `timeline.csv`：可进一步处理的全部 KS / NonKS 区间，保留原始 KS 阶段 ID 和归属。
- `summary.json`：每图窗口、KS、NonKS 时间，以及正式图片的总量与均值。

时间戳单位为 ns，使用左闭右开区间 `[start_ns,end_ns)`；相对 T0 的时间单位为 us。
脚本合并重叠 KS 后求补集，每图校验 `KS + NonKS = T1 - T0`，误差必须为 0 ns。
保留 warmup 的区间，但汇总均值只使用 `measured=1` 的图片。
NonKS 包含其区间内的计算、内存操作和空档；KS 内部间隔仍计入 KS。

**上面的命令得到 GPU 活动窗口内的时间线，不能等同于完整 Inference residual。**
`--activities` 与 `--bounds` 都省略时，仅覆盖首个 KS 开始至最后一个 KS 结束，
标记 `scope=ks_envelope`，无法确定该窗口之外的 NonKS。

当前采集流程尚未自动生成 `inference_bounds.csv`。如果已经获得同一 trace 时钟下、
与 Inference 计时一致的起止边界，可将 `--activities` 替换为 `--bounds inference_bounds.csv`，
CSV 列为 `image,start_ns,end_ns`，可另加 `measured`。
`wall.json` 只有 Inference 时长，现有 `cheddar:image:*` 范围与该计时边界也不完全一致，
不能据此直接补出完整 Inference 的起止时间戳。详细格式见 [KS 时间线说明](docs/keyswitch-spans.md#导出-ks--nonks-时间线)。

## 6. 环境准备（非 Docker 或容器缺少依赖时）

密文推理需要 Linux、NVIDIA GPU 和 CUDA 开发工具包。下面以 Ubuntu、
Python 3.10/3.11、CUDA 12.8、Blackwell GPU 为例；其他 GPU 的配置见[详细文档](docs/inference.md)。

```bash
git clone --branch feature/keyswitch-profile-kms https://github.com/zggl404/cheddar-fhe.git
cd cheddar-fhe

# root 用户可以去掉 sudo。
sudo apt-get update
sudo apt-get install -y build-essential git libtommath-dev libspdlog-dev python3-venv time wget ca-certificates

python3 -m venv ../cheddar-env
source ../cheddar-env/bin/activate
python -m pip install numpy==1.24.4 onnx==1.14.1 cmake==3.29.6

export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
nvidia-smi

git clone --depth 1 --branch branch-22.12 https://github.com/rapidsai/rmm.git ../cheddar-rmm
```

然后使用第 1 节的 CMake 命令编译；重新登录后激活 `../cheddar-env`。
