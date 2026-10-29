# Keyswitch 计时、KMS 与融合拆分执行计划

更新日期：2026-10-30。本文保存用户确认的实施范围，后续执行及恢复任务时先阅读本页。

## 当前状态与执行约定

- 工作分支：`feature/keyswitch-profile-kms`。
- 基线：`2b4646129ba2bac599efbab5ff073b6ef9c78917`，来自 `feature/cnn-decomposition-gpu96`。
- 三项功能按下述顺序形成 **三个独立 Git commit**；每个提交保持可编译、可运行。
- 本计划随第一个功能提交保存，后续在相应功能提交中更新状态和验证记录，不另建计划提交。
- 三项功能与全部十个模型最终验证已完成：85 个配置、225 次整网推理通过；本页随第三项提交。
- 文中四个运行参数均已实现，最终数值与限制见[48 GB 实测报告](keyswitch-kms-gpu48.md)。
- 用户已同意：KMS 在准备阶段加载全部所需密钥；融合拆分允许变慢；全部模型都要验证正确性。
- 用户已授权：所有工作完成、结果回传后，关闭本页指定的服务器。

测试服务器已核对：RTX 4090，49140 MiB，驱动 595.71.05，CUDA 12.8.93，架构 89；
容器主存限制 90 GiB、CPU 配额 20 核、数据盘 50 GB。

```bash
ssh -p 25331 root@connect.bjb1.seetacloud.com
```

## 1. 第一个 commit：可选的三分类计时

拟定提交主题：`Add optional keyswitch, non-keyswitch and hybrid timing`。

运行开关：

```text
--ks-profile off|on       默认 off
```

开启后，在现有 Inference Time 下输出三个互斥类别：

```text
Inference Time
  Keyswitch Time
  Non-keyswitch Time
  Hybrid Time

inference_ns = keyswitch_ns + non_keyswitch_ns + hybrid_ns
```

实施要求：

1. Keyswitch 覆盖密钥切换所需的分解、ModUp、KeyMult、ModDown 及相关变换，包含旋转、
   共轭、重线性化和 Boot dense/sparse 切换中的对应工作。共享 hoisting 只计一次。
2. Non-keyswitch 覆盖独立的明文乘法、加法、Rescale 等，以及现有 `inference_ms` 中的
   输入加密、实时权重准备、明文参考、解密验证、输出和调度开销。
3. Hybrid 覆盖同时承担两类工作的融合/共享路径，逐项核查实际调用，不能只根据函数名分类。
   重点检查 baby-step KeyMult/MAC/Automorphism、ModDownAndRescale 及其他混合路径。
4. 沿用现有完整 `inference_ms` 边界，并保留 `fhe_eval_ms` 与已有阶段明细；两套细分是同一
   父区间的不同视角，不能彼此相加。保持原三段总时间：

   ```text
   total_ms = model_prepare_ms + boot_prepare_ms + inference_ms
   ```

5. 使用整数纳秒检查加和；处理异步 GPU 执行、CPU 间隙、同步等待和嵌套调用，防止重复计时。
   GPU activity 时间作为辅助数据，不直接冒充完整墙钟。归属规则和未分类事件覆盖检查需可审计。
6. 关闭开关时，不增加用于细分测量的同步或事件采集；默认仍运行当前融合优化路径。
7. 开启后的结果明确标记为 profiling 运行，单独测量采集开销；普通日志保持简洁。

验证：关闭/开启两种状态的输出正确性、完整路径覆盖、每图与汇总加和、跨图片状态重置、
异常后的状态恢复，以及开启前后的整体耗时差异。

## 2. 第二个 commit：KMS 离线保存与准备阶段全量加载

拟定提交主题：`Add persistent CKKS key storage and preload mode`。

运行开关：

```text
--key-mode generate|save|load       默认 generate
--key-dir /path/to/keys
```

| 模式 | 行为 |
|---|---|
| generate | 保留当前生成流程 |
| save | 生成全部实际所需密钥，保存可供后续进程使用的完整密钥包 |
| load | Prepare 阶段读盘并将全部所需密钥上传 GPU；推理开始前全部就绪 |

实施要求：

1. 参考本地 ACE KMS 的保存/加载流程，为 Cheddar 当前对象布局实现序列化：
   `/Users/zzzz/Projects/active/06_ACE_CHEDDAR/ace-compiler/fhe_lib/include/fhe/kms/key_mgr.h`。
   复用现有 key request 规划和去重结果，不生成全部可能的旋转距离。
2. 覆盖模型和 Boot 的 rotation keys、multiplication/relinearization key、conjugation key，
   以及 dense-to-sparse / sparse-to-dense 切换密钥。
3. 当前测试程序要恢复与 evaluation keys 匹配的 main/sparse secrets，调整构造流程，
   避免 load 模式先生成新秘密密钥或基础 evaluation keys。
4. 密钥包记录格式版本、真实 Q/P 素数、N、word 位宽、分解配置、密钥身份、NP 形状、
   旋转请求及 level 覆盖范围，并验证文件完整性。秘密密钥和密钥二进制不提交到 Git。
5. load 模式缺失、不匹配或损坏时明确失败，禁止静默回退到生成。推理期间密钥生成次数为零。
6. 密钥在准备阶段全量常驻；本次不需要运行时按需加载或换出。
7. 读盘、校验、GPU 上传计入对应准备阶段，共享密钥只计一次。准备子项可区分生成与加载，
   原有三段总时间边界保持一致。
8. 权重仍按现有 pre/stream 流程编码；KMS 不自动包含持久化预编码权重。
   KMS 不减少全量常驻密钥的显存，加载是否更快需通过实测判断。

验证：跨进程 save/load，加载后完整密文推理；检查密钥身份、实际使用密钥的哈希/内容、
生成次数为零，以及缺文件、参数不匹配、损坏数据的拒绝行为。比较明文参考和误差，
不要求不同次加密的原始 ciphertext 字节一致。

## 3. 第三个 commit：可选的 Hybrid 拆分执行

拟定提交主题：`Add unfused diagnostic execution for two-way timing`。

运行开关：

```text
--split-hybrid off|on       默认 off
```

实施要求：

1. off 保持现有融合路径。on 对第一项识别出的每种 Hybrid 路径提供能够独立执行、分类的实现，
   例如把 ModDownAndRescale 拆成独立 ModDown 和 Rescale；同时检查其他融合/共享操作。
2. 拆分开关与计时开关独立：不开启采集也可以运行拆分路径，用于排查数值与测量本身的影响。
3. 当 `--ks-profile on --split-hybrid on` 时，成功覆盖的完整路径应满足：

   ```text
   inference_ns = keyswitch_ns + non_keyswitch_ns
   hybrid_ns = 0
   ```

   对仍然存在的混合事件明确报错或标记未完成，不能靠改标签、忽略或按比例分配隐藏 Hybrid。
4. 拆分后的计时属于拆分执行模式，不代表原融合 kernel 内部的真实时间分摊。
   用户接受其整体变慢；分别报告融合与拆分路径的整体时间。
5. 保持相同数学模型、CKKS 参数、Boot 日程与精度门槛；验证 level、scale、残差对齐、
   输出 logits 和误差。特别检查联合 ModDown/Rescale 改为顺序执行时的舍入差异。

验证：混合路径的针对性 GPU 用例，融合与拆分的完整网络对照，所有开关组合的可运行性，
以及每图和汇总的两分类加和及 Hybrid 覆盖审计。

## 全部模型的测试要求

以下十个模型均需完成真实 GPU 密文整网推理：

```text
ResNet20  ResNet32  ResNet44  ResNet56  ResNet110
VGG11     VGG16     MobileNet  SqueezeNet  AlexNet
```

先读取实际 GPU 容量、驱动、CUDA、可用主存、磁盘和原始模型文件，核对权重与 CIFAR 数据。
各模型独立进程顺序运行，归档源码和模型哈希、命令、环境及日志。

### 能完整预编码的模型：严格验证

- 依据全部模型密钥、Boot 密钥/常量、预编码权重、上下文与临时工作区的实际峰值判断，
  不能只比较权重大小与 48 GB，也不能把内存池保留空间当作活跃数据大小。
- 强制 `--encode-mode pre`，确认推理阶段实时权重准备为零，密钥生成/加载均为零。
- 核心对照配置使用相同图片，计划每配置 1 张预热加 5 张正式图片，记录性能分布、
  profiling 开销、KMS 准备耗时和拆分后的时间变化。
- 拆分路径临时空间若增加，必须重新验证能否完整预编码；两条路径比较时明确编码方式。

### 无法完整预编码的模型：降低性能采样要求，保持正确性要求

- 仍加载全部所需密钥，采用现有有界 `--encode-mode stream` 构造、编码和释放权重。
  不能仅因整网预编码放不下而跳过模型。
- 核心对照配置至少使用相同的 2 张图片完成整网正确性检查；减少采样次数，
  不将短样本耗时宣称为稳定性能结论。
- 实时权重编码进入 `runtime_weight_prepare_ms`，在新增分类中归入 Non-keyswitch。
  与 pre 的时间分别报告，不能合并成同一种在线基线。
- 保留真实密文运算、原 CKKS 参数和精度门槛；不能用明文 replay、局部层或更宽误差门槛替代。

两组共同检查：FHE 对同多项式明文的最大/平均绝对误差、逐类别输出、top-1，
以及 FHE 对普通 ReLU 参考的误差；区分同态误差与激活近似误差。
明文路径继续与原始 ONNX 的独立参考核对。性能采样要求可不同，正确性门槛不降低。

### 最终功能矩阵

| 配置 | key-mode | ks-profile | split-hybrid | 目的 |
|---|---|---|---|---|
| A | generate | off | off | 当前默认路径基线 |
| B | save | off | off | 生成可在新进程复用的密钥包 |
| C | load | off | off | 验证 KMS 全量加载及其准备耗时 |
| D | load | on | off | 验证融合路径三分类计时 |
| E | load | on | on | 验证拆分路径两分类计时 |
| F | load | off | on | 验证拆分执行与采集开关独立 |

- 全部模型覆盖 A–F；A/C/D/E 为上述核心对照，F 至少 1 张完整图片，B 完成保存后再运行 C。
- key-mode 与计时/拆分组合的额外控制流在小模型补齐；每个 commit 完成相应可用功能的验证，
  第三个 commit 后执行全部模型的最终矩阵。
- 所有失败保留原始日志，修复后重新验证相关配置。OOM 本身不是其他模型正确性通过的证据。
- 每张图、各正式样本均值与整个进程分别检查加和；完整三段总时间与外部 `time` 边界核对。

## 交付、归档与关闭服务器

1. 完成三个功能提交，更新本计划、详细使用文档和 README 的必要入口。
   主 README 保持简洁，底层分类、KMS 格式和拆分细节放入详细文档。
2. 归档每个模型的实际编码方式、测试配置、原始 logits、误差、计时、显存峰值、
   密钥大小/数量、模型与二进制哈希，以及三个提交的 SHA。
3. 将代码、报告、原始日志与复现所需清单回传本地仓库或本地忽略的 validation 目录，
   核验回传成功；远程关机前不将唯一结果留在服务器。
4. 全部工作完成后按用户授权关闭 `connect.bjb1.seetacloud.com:25331` 对应服务器。
   先确认平台实际支持的关机方式，再执行并验证停机状态；仅退出 SSH 或停止测试不算关机。
   无法确认实际停机时明确报告，不能把 SSH 连接失败单独当作平台已停止的证据。

## 执行状态清单

- [x] 创建新分支并保存用户确认的范围。
- [x] 核对 48 GB 服务器、模型、数据、工具链和可用空间。
- [x] 完成 commit 1：三分类计时与验证（提交 SHA 见 Git 历史）。
- [x] 完成 commit 2：KMS 保存/全量加载与验证（提交 SHA 见 Git 历史）。
- [x] 完成 commit 3：Hybrid 拆分与验证（本次提交）。
- [x] 完成全部十个模型的最终矩阵与正确性报告（85 配置 / 225 次整网推理）。
- [x] 回传并核验代码、日志、结果和复现清单；源码 109 个文件与实际测试版本逐一核对。
- 收尾顺序：最终提交/推送和归档核验后执行指定服务器关机；实际执行与复查状态以
  本地 `validation/ks-kms-20261030/shutdown-verification.json` 为准（提交后生成）。

## 已确认的实施取舍

- 第一项采用同步 scope，而非 CUPTI activity 归因。关闭时无新增同步；开启时扰动执行重叠，
  必须独立报告 off/on。28 次 GPU 整网验证全部通过，详见[计时说明](keyswitch-profiling.md)。
- 50 GB 数据盘按模型顺序测试和回收临时密钥包，保留参数、密钥身份、哈希及验证日志；
  不要求十个模型的二进制密钥包同时保存在服务器上。

- 第二项采用 4 MiB staged I/O、非认证的损坏 checksum 和同盘目录原子发布；没有加入 GDS/并行 I/O/压缩。
  save 模式保存后继续推理验证，额外导出费用归模型准备；详见[KMS 文档](kms.md)。

- 第三项保留同类融合，拆开跨类别融合。Boot 从相同短基代表元分别生成 Q/P，
  明确共享 INTT 归非切换、P 扩展归切换。启用回退路径时修正首项写回与别名问题；
  N16/N17 的 Q/QP、1/2/10 项、额外别名项和完整 Boot 均已通过。

## 运行现场与恢复入口

- 已有提交：`dca81b6`（三分类计时）、`3672828`（KMS）。第三项随本页提交；完整三次提交 SHA 见 Git 历史。
- 服务器工作目录：`/root/autodl-tmp/ks-kms`，源码 `src/`，构建 `build/`，权重 `models/`。
- 远程进度：`results/final-progress.log`；逐配置原始日志、内存采样和 JSON 为 `results/final/`。
- 本地证据目录：`validation/ks-kms-20261030/`；包含 commit1/commit2/final、GPU 单元日志、
  `verified-inputs.json` 及矩阵脚本。原始证据保持 Git 忽略，摘要进入文档。
- 最终矩阵脚本：本地 `validation/ks-kms-20261030/run_matrix.py`，远程工作目录下同名文件。
- 环境 PATH：`/usr/local/cuda/bin:/root/miniconda3/bin:$PATH`；CMake 架构 89，RMM branch-22.12。
- 每个模型 A/C/D/E 为核心对照，B 保存、F 无采集拆分；ResNet20 补齐其他开关组合。
  严格组每核心配置 1 张预热加 5 张正式图，大模型每核心配置同样的 2 张完整图。
- 当前参考复用此前独立 ONNX/PyTorch float64 结果，并已重新核对全部十个原始 ONNX、
  FP32 权重、CIFAR 与参考文件的哈希；每次整网仍逐 logits 比较普通 ReLU 明文输出。
- 已读取服务器 `/usr/bin/shutdown`：平台脚本会停止容器的 supervisord；归档和核验前不调用。

- 主矩阵中 ResNet56 的 auto 保守预算选择 stream；主矩阵结束后已补跑
  `run_pre_resnet56.py resnet56` 强制 pre 的严格 A–F 矩阵，判断是否实际能全量常驻。
  补测结果在 `results/final-pre-resnet56/`，与原 stream 结果分别归档。

- 已核对 [AutoDL 官方自动关机说明](https://www.autodl.com/docs/save_money/)：使用平台专用
  `/usr/bin/shutdown`。归档后调用，保留执行记录和后续连接状态。

## 最终验证结论

- 主矩阵 78 配置 / 198 次整网推理通过；ResNet56 强制 pre 补测 7 配置 / 27 次通过。
- 六个模型采用严格 pre：ResNet20/32/44/56、MobileNet、AlexNet。
  ResNet110、VGG11/16、SqueezeNet 采用 stream，全部 key 仍提前加载，精度门槛不降低。
- 所有 load 生成次数为 0；所有拆分采集 Hybrid 次数/耗时为 0；整数纳秒加和误差为 0。
- Boot 日程保持不变；全部测试 top-1 与两条明文及独立 ONNX 参考一致。
- 普通人类可读日志另完成一张 ResNet20 拆分推理，正常显示参数、10 类输出和两类时间。
- 所有实际测试源码与工作区 109 个文件哈希一致；GPU 二进制、构建记录与原始证据已打包，
  本地逐文件核验后执行发布和平台关机。
