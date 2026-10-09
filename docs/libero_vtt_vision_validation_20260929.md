# LIBERO 验证 VTT 与感知端改动

2026-09-29：用户决定先在 LIBERO 验证，再决定 RoboTwin 长训练方案。

## 当前执行

检查发现现有组合实验已经运行，沿用该任务，不重复启动：
`gawm_libero_lila_vtt_norm_c_160k_b160_20260929`。

- 五机 32 卡，global batch 160，seed 42；两路真实相机 agentview、eye_in_hand。
- 冻结 DINO；LiLa 多层融合、每路 64 个 query、fixed LayerNorm bridge、VTT；ACT 和现有世界模型目标保持该运行的冻结快照。
- 完整训练数据 6,585 条示范、1,027,125 帧；VTT 来自训练示范，评测不计算目标向量。
- 当前控制器会在 **160k 训练完成后** 依次评测 40k、80k、120k、160k。不是到 40k 自动提前评测。
- 每个检查点四套件共 40 个任务，每任务 10 条，共 400 回合；seed 7、执行长度 8、无 temporal ensemble。
- 历史旧版 160k 为 369/400（92.25%），只作整体参考；历史数据、预处理和训练历史不完全相同，不能据此单独归因 VTT 或视觉前端。

本次检查时已经到 1280 步，日志数值全部有限、视觉 content RMS 约 1。训练仍处早期，不能据此判断是否解决 latent 振荡或闭环退化。当前计算耗时估算剩余约 16.2 小时，未计 checkpoint、通信计时遗漏和评测开销，以实际进度为准。

## 预备消融

为节省资源，默认先看正在运行的组合结果，再按结果补消融。以下四组配置已生成并校验 recipe 展开、batch 和预算，**未启动、未排入自动队列，也未做实际训练 smoke**：

| 组别 | 感知端 | 任务条件 | 用途 |
|---|---|---|---|
| A | 旧网格方案 | 文本 | 同数据的基线 |
| B | 旧网格方案 | VTT | B 对 A：VTT 在旧视觉上的作用 |
| C | LiLa 方案 | 文本 | C 对 A：感知方案在文本条件下的作用 |
| D | LiLa 方案 | VTT | D 对 C：VTT 在新视觉上的作用；D 对 B：感知方案在 VTT 下的作用 |

配置目录：`examples/LIBERO/train_files/ablation_vtt_vision_20260929/`。

各组使用相同的补齐数据、两路相机、seed 42、global batch 160、动作头、未来时刻、latent objective 和 C 学习率规则，从头训练。VTT 两组复用同一训练集向量资产。每组先看 40k 更新，保留 10k/20k/40k。40k 是原 160k 调度的前缀；不压缩 cosine、不提前做 12 epoch 优化器重置（原切换为 77,028 步，周期 256,760 步）。所以 40k 结果只衡量相同训练预算下的表现，不代表最终收敛上限。

D 可直接使用当前运行的 40k checkpoint，避免重训。若后续启动其余配置，沿用当前实验冻结源码；先核对资源、各节点数据和 VTT 资产，再做短训/推理检查。32 卡对应每卡 5；改变卡数必须重新配平 batch。配置本身不分配 GPU。

“感知端”是整套改动：旧方案每路 16 个固定网格 token、224×224、bicubic；LiLa 每路 64 个 query、256×256、OpenCV linear、多层融合和 bridge norm。四组可以比较整个感知方案，不能将差异进一步归因于 query adapter 本身。若需要 adapter 级别结论，再统一输入像素、token 数和输出归一化做额外消融。

采用相同四套件、同任务/官方初始状态逐条比较，报告逐套件成功率及成败翻转；不以 raw latent MSE 的跨表示数值大小判断优劣。单一训练 seed、每任务 10 次用于筛选，确认收益后再增加训练种子或 rollout 数。

## 后续世界模型改版

冻结 DINO patch 未来监督和时间平滑仍按 [GAWM_FUTURE_PREDICTION_OBJECTIVE.md](../design/GAWM_FUTURE_PREDICTION_OBJECTIVE.md) 执行，但应成为下一组独立实验。当前先验证已经实现的 VTT/视觉改动，不将未实现的目标函数混入这组归因实验。LIBERO 成功也不能直接证明 RoboTwin 同样有效。

## 产物

- 当前训练与自动评测说明：`docs/libero_lila_vtt_norm_160k_20260929.md`
- 当前控制器：`playground/Queues/libero_lila_vtt_norm_160k_20260929/status.json`
- 配置校验与检查时指标：`examples/LIBERO/train_files/ablation_vtt_vision_20260929/validation.json`
