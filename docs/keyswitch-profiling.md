# Keyswitch 细分计时

当前公开输出和三种独立开关见[三种计时方式](timing-modes.md)。本页的 schema v2 扁平字段
是 CPU/verbose 内部审计格式，公开 `CPU_TIMING` 三分类以 FHE 求值为父区间。

统一运行入口增加 `--ks-profile off|on`，默认 `off`。例如：

```bash
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin --ks-profile on
```

开启后，公开 `CPU_TIMING` 按 FHE 求值区间报告三分类。
在 `--verbose 1` 的内部详细报告中，完整 `inference_ms` 另按三类拆分。
保留原来的阶段细分，两种视角不能再相加；`fhe_eval_ms` 仍表示纯密文图求值所在的阶段。
JSON 额外提供整数纳秒、各操作独占耗时和调用次数，检查三类之和严格等于 `inference_ns`。
输入加密、明文参考、解密输出、实时权重编码和未处于切换/混合区间的调度都在 Non-keyswitch。

## 测量方法与取舍

本实现使用 `synchronized_scopes`，不引入 CUPTI。每次独占操作区间切换前调用
`cudaDeviceSynchronize`，将此前排队的工作及等待记给此前的拥有者，再开始下一个区间。
嵌套区间暂停父区间，返回后恢复；相同操作的嵌套不重复记录耗时。全量输入/输出阶段
不进入 Keyswitch 或 Hybrid，剩余墙钟归 Non-keyswitch。报告使用同一次运行的父区间，
不将不同开关运行的时间拼接。

取舍：这种方法保持完整墙钟加和并可直接审核边界，但同步会破坏 CPU/GPU 重叠，
细分开启时可能明显变慢。其数值表示带采集开销的同步实现，不能解释为默认运行的
无扰动分解，也不是 GPU kernel activity 时间。应单独比较 off/on；不把差值强行扣给某一类。
关闭时仅保留未激活 scope 的轻量分支，不采集、不引入额外同步，保留现有融合优化。
本实现按已检查的源码调用路径分类，不宣称取得硬件级所有 kernel 的 activity 覆盖证据。
口径面向当前单模型、单 host 推理序列；device-wide 同步会等待其他并发 GPU 工作，
因此测评时逐个模型运行，不将并发场景中的外部等待归因结果用作纯模型性能。

## 分类边界

| JSON 操作 | 类别 | 范围 |
|---|---|---|
| modup | Keyswitch | 切换分解、相关 INTT/基转换/NTT；拆分模式含 Boot 的 P 扩展 |
| pseudo_modup | Keyswitch | 延迟切换路径的 P 乘积准备 |
| key_mult | Keyswitch | evaluation-key 乘加及对应密文项修正 |
| moddown | Keyswitch | 从 QP 到 Q 的切换降模 |
| automorphism | Keyswitch | 旋转/共轭所需的独立置换 |
| sparse_to_dense | Keyswitch | Boot 的专用 StD 乘密钥与修正；内部 ModDown 单独独占计时 |
| key_correction | Keyswitch | 必要的原密文项修正，不包含网络结果累加 |
| fused_baby | Keyswitch | baby-step KeyMult、必要修正、automorphism 融合；拆分模式保留融合 |
| moddown_rescale | Hybrid | 联合 ModDown 与 Rescale |
| automorphism_accum | Hybrid | 置换与多项累加融合 |
| shared_lift | Hybrid | 融合模式中 Boot ax 的共同前置及 Q/P 提升 |
| shared_lift_common | Non-keyswitch | 拆分模式仍由 Q/P 两路共同使用的短基 INTT/归一化，计一次 |

独立明文乘法/累加（包括 giant-step 的 plaintext multiplication/accumulation）、
普通 Rescale、Boot 的 bx 提升和其余运算属于 Non-keyswitch。NTT 的归属取决于调用语义，
不能把所有 NTT 都计为 Keyswitch。`calls` 是进入该源码 scope 的次数，不是旋转密钥数量、
逻辑旋转次数或 CUDA kernel 数。

全部操作 ns 使用独占记录，按表分别汇总成 Keyswitch/Hybrid；Non-keyswitch 为父区间
减去这两类，涵盖明确划归此类的其他执行与管理开销。CPU 和 GPU 同步等待不被重复计时。
校验器同时检查浮点毫秒与整数纳秒对应、操作汇总及父子加和。

## 第一个提交的 GPU 验证

2026-10-30，RTX 4090（49140 MiB），驱动 595.71.05 / CUDA 12.8.93，架构 89。
ResNet20 和 MobileNet 均整网预编码，每个开关状态 1 张预热加 5 张正式图片；
VGG16 使用 stream，每状态相同的 2 张图片，无预热。共 28 次完整密文推理。

| 模型 / 编码 | off 完整单图均值 ms | on 完整单图均值 ms | on 相对开销 |
|---|---:|---:|---:|
| ResNet20 / pre | 1268.191 | 1438.410 | 13.42% |
| MobileNet / pre | 5082.237 | 5519.252 | 8.60% |
| VGG16 / stream | 8100.489 | 8452.513 | 4.35% |

三类整数纳秒逐图与汇总加和误差均为 0。各配置保持原有逐层 1e-2 数值门槛，
FHE/同多项式/普通 ReLU top-1 一致；FHE 求值内编码调用均为 0。
VGG16 的实时权重准备正确保留在推理阶段并归入 Non-keyswitch。
这些是该硬件、该样本和同步采集方法的结果，不是默认路径的无扰动算子占比。
本地原始证据：`validation/ks-kms-20261030/commit1/`（Git 忽略目录）。
CPU 的嵌套区间、异常恢复、关闭无额外同步、launcher/JSON 加和检查也均通过。

## Hybrid 拆分诊断模式

```bash
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --key-mode load --key-dir /data/keys/resnet20 \
  --ks-profile on --split-hybrid on
```

`--split-hybrid off|on` 默认 off，与三个采集开关独立。开启后采用诊断执行，允许更慢。
三类之和仍等于相应父区间；拆分模式要求 Hybrid 为零，共同短基前置 `shared_lift_common` 归 Non-keyswitch。
C++ 与 Event/NVTX 分析器拒绝残留的 `moddown_rescale`、`automorphism_accum`、`shared_lift`。

| 原路径 | 当前拆分执行 |
|---|---|
| ModDownAndRescale | QP→Q ModDown（KS），随后独立 Rescale（NonKS） |
| BSFusedKeyMult | 保持融合，整段归 KS；其中没有明文矩阵乘法 |
| HRotAdd/HConjAdd 的 PermuteAccum | 独立 Permute（KS）与额外 Add（NonKS），保留别名安全的临时缓冲 |
| giant-step 的延后 b 修正 | 每次旋转先补 b（KS）并置换（KS），再做跨 giant-step 累加（NonKS） |
| Boot ax 的共享 Q/P 提升 | 共同短基 INTT/归一化（NonKS），Q 扩展/NTT（NonKS），P 扩展/NTT（KS） |

Boot 两条扩展分支使用与原路径相同的短基整数代表元，不从已提升 Q 任意重构 P。
共同短基变换只做一次、计一次，按约定归 Boot 执行内的 Non-keyswitch；P 路独立扩展归 Keyswitch。

`PermuteAccum` 还按实际输入区分：全零旋转的加法归 NonKS；只有一个置换且没有额外项时归 KS。
在 giant-step，必要 b 修正与网络结果累加由调用层区分，不能把所有 Add 统一划入 NonKS。

其余同类融合保留，例如 giant-step 明文乘法/累加均属 Non-keyswitch，NTT 内部优化
均随所在操作归类，不要求全部 GPU kernel 都取消融合。
`Parameter::split_hybrid_` 只控制求值派发，不改变 CKKS 参数和 KMS 格式；
同一包密钥可在融合/拆分模式间使用。

拆分增加中间 Q 密文、累加缓冲、内存读写和 kernel launch。联合舍入改成顺序舍入可能
产生小的数值差异，因此比较 level、scale、误差和输入不变性，不要求输出字节相同。
不开启采集时仍保留拆分执行开销；不开启拆分时仍保留当前优化路径。

启用旧敏感性测试回退路径时修正了一个首项写回问题：当首项需要置换且没有额外未旋转项时，
旧实现写到 tmp 却从尚未初始化的 dst 继续累加。新实现以独立 sum 累加，所有原始输入
读完后再写 dst，也覆盖单项/零旋转以及额外项与 dst 别名的情形。

针对性验证在 N16/N17、Q 与 QP 下逐系数比较取模后的结果，覆盖 1、2、10 项累加，
有/无额外别名项，并验证乘法、线性层及完整 Boot。下述是旧拆分归属约定的历史结果；当时拆分运行的 Hybrid 次数/耗时均为 0。
当前方案也要求 Hybrid 清零，但保持 FusedBaby 融合，旧分项耗时不能直接套用。Boot 融合/拆分最大差异分别为 `7.32914e-7`、
`2.06027e-6`，输出 level 分别保持 8、9；均通过 1e-3 门槛。

全部十个模型最终 85 配置 / 225 次整网验证及运行取舍见[48 GB 报告](keyswitch-kms-gpu48.md)。

分类采用表中的语义边界：密钥乘加、必要密文修正和 fused_baby 都归 Keyswitch；
网络结果的独立累加归 Non-keyswitch，共同短基前置归 Non-keyswitch。
因此 Keyswitch 是这里明确定义的切换相关墙钟区间，不能解释为仅 key-GEMM 指令时间。
比较不同执行模式时应同时看完整推理与三类之和，不能把某类时间变化直接当作
同一组 GPU 指令的净加速或精确分摊。这一归属约定与同步开销均保留在原始结果中。
