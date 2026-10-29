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

## 分类边界

| JSON 操作 | 类别 | 范围 |
|---|---|---|
| modup | Keyswitch | 切换分解、相关 INTT/基转换/NTT |
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
