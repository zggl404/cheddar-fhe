# Keyswitch 细分计时

统一运行入口增加 `--ks-profile off|on`，默认 `off`。例如：

```bash
python scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin --ks-profile on
```

开启后，完整 `inference_ms` 按 Keyswitch、Non-keyswitch、Hybrid 三类拆分。
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
| fused_baby | Hybrid | baby-step KeyMult、MAC、automorphism 融合 |
| moddown_rescale | Hybrid | 联合 ModDown 与 Rescale |
| automorphism_accum | Hybrid | 置换与多项累加融合 |
| shared_lift | Hybrid | Boot 的 ax 同时提升到 Q 与 P，P 部分供 StD 使用 |

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

`--split-hybrid off|on` 默认 off，与采集开关独立。开启后执行独立步骤，允许更慢；
关闭采集也可以运行拆分路径。两类计时满足 `inference_ns = keyswitch_ns + non_keyswitch_ns`。
人类可读日志只显示这两类，JSON 保留 `hybrid_ns=0` 及各混合 scope 的零调用数便于审计。
如果拆分模式仍进入任何 Hybrid scope，C++ 和 Python 校验均拒绝结果，不能改标签消掉时间。

| 原混合路径 | 拆分执行 |
|---|---|
| ModDownAndRescale | QP→Q ModDown，随后独立 Rescale |
| BSFusedKeyMult | KeyMult、独立 Add、独立 Automorphism |
| PermuteAccum | 独立 Permute 与 Add，临时 sum 完成后写回 |
| Boot ax 的共享 Q/P 提升 | 原短基 INTT 与 Q 提升/NTT 属于 bootstrap 非切换部分；再独立提升/NTT 到 P，归切换 ModUp |

Boot 拆分使用与原路径**相同的短基整数代表元**，不从已提升的 Q 任意重构 P；
后者会改变代表元，可能影响 StD 的数值正确性。短基 INTT 的结果被复用一次，
将该共同前置步骤归 bootstrap Non-keyswitch，单独的 P 扩展归 Keyswitch。
这是明确的数据流归属约定，不能解释为原共享 kernel 内部可精确分割的指令时间。

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
有/无额外别名项，并验证乘法、线性层及完整 Boot。四类 Hybrid 均在融合运行中被覆盖，
拆分运行的 Hybrid 次数/耗时均为 0。Boot 融合/拆分最大差异分别为 `7.32914e-7`、
`2.06027e-6`，输出 level 分别保持 8、9；均通过 1e-3 门槛。

全部十个模型最终 85 配置 / 225 次整网验证及运行取舍见[48 GB 报告](keyswitch-kms-gpu48.md)。

分类采用表中的实现级边界：`key_mult` 包括该 helper 内必要的密文项修正，
`fused_baby` 保守地整段记为 Hybrid；拆分后的独立 Add 记入 Non-keyswitch。
因此 Keyswitch 是这里明确定义的切换相关墙钟区间，不能解释为仅 key-GEMM 指令时间。
比较不同执行模式时应同时看完整推理与三类/两类之和，不能把某类时间变化直接当作
同一组 GPU 指令的净加速或精确分摊。这一归属约定与同步开销均保留在原始结果中。
