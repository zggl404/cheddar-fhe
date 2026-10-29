# KMS：预生成、保存与准备阶段加载密钥

本次扩展了客户端构造函数；更新时库与调用程序需一起重新编译，避免混用旧二进制。

三个模式均通过统一入口运行，默认 `generate` 保留原生成流程：

```bash
# 离线预先执行一次：生成、保存，并运行正常推理验证密钥包。
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --key-mode save --key-dir /data/keys/resnet20

# 后续进程：准备阶段全部加载，不重新生成。
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --key-mode load --key-dir /data/keys/resnet20 --ks-profile on
```

`save` 的目标目录必须不存在，避免用另一套秘密密钥覆盖已有密钥包。
它完成准备和保存后继续正常推理；可以把这次运行作为独立的离线预生成操作。
`load` 缺文件、参数不匹配、密钥身份不一致、level 覆盖不足或校验失败时返回非零，
不回退到生成。不同进程加载同一包使用匹配的 main/sparse secrets。

这是 Cheddar 专用的密钥格式，参考 ACE 的保存/加载流程，不与 ACE/Phantom 的二进制文件互通。

## 文件与兼容性

目录包含 `manifest.txt`、`0.key`（main secret）、`-1.key`（使用 SSE 时的 sparse secret），
以及以 evaluation-key index 命名的 `.key` 文件。涵盖基础乘法/共轭、模型与 Boot 的实际
旋转集合和 dense/sparse 切换密钥。正数 index 是旋转距离，特殊负数/正数 index
沿用 `EvkMap` 的约定。相同旋转只保存一份最终覆盖最高请求 level 的密钥。

manifest 记录版本、唯一密钥身份、全部真实 Q/P 素数、N、word 位宽、分解参数、
level 配置、scale 与秘密密钥稀疏度，以及每把 key 的 NP/beta。
加载时与当前完整参数描述比较，并核对每项请求的覆盖范围。
密钥可跨模型复用的前提是参数一致、秘密密钥身份一致且实际请求被覆盖；
文件名不承担兼容性判断。权重、图片内容、pre/stream 模式不改变密钥格式。

文件使用明确顺序的定宽标量，不直接 dump C++ 对象或 GPU 指针。
每文件验证 magic/version、little-endian/word 位宽、密钥身份和 index；
每向量验证长度和 64-bit word checksum，并拒绝截断或多余数据。
checksum 用于检测存储损坏，不提供抵御主动篡改的密码学认证。
当前测试客户端和原随机密钥生成器的测试用途边界保持不变。

## 加载、准备与内存

载入按既有准备阶段执行：构造客户端时恢复 secrets 和基础 key；模型准备时加载模型 key；
Boot 准备时加载 Boot 请求及 dense/sparse key。首次请求直接加载包内最终覆盖版本，
后续请求覆盖足够则复用。共享 key 的读取归首次需要它的准备阶段；Boot 复用已在模型阶段
加载的高 level key，因此评价总准备收益时应同时查看两个准备阶段。进入推理前全部实际所需密钥已在 GPU，
推理期间沿用 runner 的密钥请求锁，缺请求直接失败。

使用 4 MiB host-staged 分块传输，每次只保留一个临时主存块；完整密钥常驻 GPU。
本次实现不引入 GDS、并行 I/O、磁盘压缩或按需换出。这些不是正确加载所必需的功能。
是否比生成更快取决于磁盘、校验和传输，需查看实际准备时间。

权重仍按 `--encode-mode auto|pre|stream` 编码；密钥包不包含预编码权重或 Boot 矩阵明文。
KMS 不减少全量密钥的显存占用。`KEY_RESOURCES` 提供 mode、identity、key_count、
keys_generated、keys_loaded、bytes_loaded、bytes_saved，便于核对 load 模式生成次数为零。
`key_count` 是最终不同 key 的数量；`keys_generated` 统计生成调用，包含较低 level 的 key
被更高 level 请求替换的次数，因此 save 模式可能大于最终数量。文件字节数为原始系数及
头部大小，不包含 RMM 分配取整、内存池保留或求值临时空间。

三段总时间保持不变。加载、校验和上传计入实际对应的模型/Boot 准备阶段。
保存的额外序列化在完整 key set 准备后执行，统一计入模型准备及进程开销；
这次离线导出费用不应混入 load 模式的在线推理时间。

## 保存完成与持久性取舍

先在带随机身份的同盘 `.pending-*` 目录写文件，成功写完 manifest 后原子重命名。
失败清理本次创建的临时目录，已存在的目标目录不覆盖。目录权限为 owner-only，
密钥二进制尤其 secrets 不加入 Git。格式包含身份绑定，但不提供秘密密钥加密存储。
本实现未对每个文件执行 fsync；异常掉电的持久性不作额外保证，重启后的加载仍通过
文件大小和 checksum 拒绝不完整内容。关机前先完成进程并核验文件/结果。

## 第二个提交的 GPU 验证

RTX 4090 / CUDA 12.8.93：针对性测试核对全部 6 把测试密钥的每个 RNS 系数（含 P、
全部 beta 和两分量）完全一致，并使用原客户端加密、加载客户端求值/解密。
旋转最大误差 `1.42123e-7`，乘法/Rescale 通过 1e-3 门槛。
缺文件、损坏 evaluation key、损坏 secret、错误参数、level 不足、未完成目录和覆盖现有目录均拒绝。

另对 ResNet20/pre、MobileNet/pre、VGG16/stream 分别在新进程 save 1 张、load 2 张，
共 9 次整网推理，保持原有逐层 1e-2 门槛及三条输出 top-1 一致。
load 全部开启三分类采集，整数纳秒加和误差为 0，生成次数全部为 0。

| 模型 | 最终 evaluation keys | 包大小，含 secrets 和头部（bytes） | load 生成次数 |
|---|---:|---:|---:|
| ResNet20 | 219 | 5202007864 | 0 |
| MobileNet | 398 | 14793348832 | 0 |
| VGG16 | 382 | 14683246944 | 0 |

本地原始证据位于 `validation/ks-kms-20261030/commit2/` 与 `key-store-test.log`。
每个密钥文件的大小和 SHA-256 已归档；临时密钥包在完成该模型测试后回收，避免占满 50 GB 数据盘。

本轮运行验证覆盖 uint32、N16/N17、normal/Boot scale 40/58 的十个模型预设；
uint64 序列化代码完成编译，其他自定义配置尚未纳入本轮 GPU 测评。
