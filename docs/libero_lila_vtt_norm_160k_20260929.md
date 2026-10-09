# 当前 GAWM 的 LIBERO 160k 训练与退化检查

> 2026-09-29 晚：按用户要求已停止此运行及自动评测，保留10k完整检查点。正在切换至[固定DINO监督版本](libero_fixed_dino_training_20260929.md)。下文启动状态为历史记录。

2026-09-29，用户明确选择从头训练、沿用此前 160k 预算。正式结果尚未产生，不能判断是否退化。

运行：`gawm_libero_lila_vtt_norm_c_160k_b160_20260929`。已启动独立后台实验控制器；正式训练前，32 卡四步训练、完整 checkpoint 保存及 40 个任务各一次仿真预检均完成。预检使用几乎随机的策略，只验证运行链路，不作为成功率比较结果。

## 配置

- 当前工作区代码已冻结，包含未提交模型修改；LiLa 三层 DINO 特征融合、每相机 64 个 query、VTT 条件、fixed LayerNorm bridge。
- 两路真实 RGB：agentview、eye_in_hand；256×256，OpenCV linear 缩放，与新版训练路径一致。冻结 DINOv3 ViT-L 预训练权重；可训练模块重新初始化，不加载旧策略。
- 本轮没有加入文档中尚未实现的 DINO patch 未来监督和时间平滑。
- 补齐后的 LIBERO 数据：6,585 条训练示范、1,027,125 帧，frame_epoch 无放回采样。VTT 从全部训练示范生成，共 112 个任务键，覆盖 40 个评测任务。
- 32 卡，每卡 batch 5，全局 batch 160，seed 42，FP32 参数/Adam 状态、BF16 autocast。优先探测 40 卡，但部分卡被其他作业占用，本次不抢占。
- GPU：本机、worker-3、worker-2 各 0–7；worker-1、worker-4 各 0–3。
- 沿用 C 配方：第一阶段 12 epoch，即 77,028 步，之后重置优化器；总计 160,000 步。每 epoch 6,419 次更新；各阶段 cosine 名义周期 256,760 步。与旧 160k 相比，新增数据使阶段切换从 65,400 变为 77,028 步。
- 已通过 32 项相关测试、2 项子测试；短训 4 步日志全部有限，视觉 content RMS 约 1。

## 自动评测与对照

正式训练完成后依次评测 40k、80k、120k、160k；最终 160k 是预先确定的主要比较点。每个检查点 Spatial/Object/Goal/Long 四套件，各 10 个任务，每任务前 10 个官方初始状态，共 400 回合，seed 7、动作执行长度 8、franka 归一化、无 temporal ensemble、FP32 policy/BF16 vision。不复用历史 rollout。

| 参考结果 | 成功数 | 成功率 |
|---|---:|---:|
| 旧版从头训练 160k | 369/400 | 92.25% |
| 最近补齐数据后微调 40k | 372/400 | 93.00% |
| 旧版最佳 200k | 374/400 | 93.50% |

控制器会在每个正式评测完成后写出 `comparison.json` 和 `comparison.md`，包含总成功率差、逐套件差、同一任务/初始状态的成功转失败与失败转成功数，以及精确 McNemar 配对检验。单次 seed、每任务 10 次属于快速检查；统计结果不能替代多次独立训练。

新版和旧 160k 的数据规模、图像预处理、架构不同；最近微调又继承了旧 200k 权重。因此本次是整体训练结果比较，不是只改变架构的严格消融。

## 状态与产物

- [实时实验状态](../playground/Queues/libero_lila_vtt_norm_160k_20260929/status.json)
- [实验计划](../playground/Queues/libero_lila_vtt_norm_160k_20260929/plan.json)
- [实际启动配置](../playground/Queues/libero_lila_vtt_norm_160k_20260929/launch_config.yaml)
- [训练状态](../playground/Checkpoints/gawm_libero_lila_vtt_norm_c_160k_b160_20260929/cluster_status.json)
- [训练指标](../playground/Checkpoints/gawm_libero_lila_vtt_norm_c_160k_b160_20260929/metrics.jsonl)
- [VTT 校验](../playground/Queues/libero_lila_vtt_norm_160k_20260929/vtt_validation.json)
- [短训校验](../playground/Queues/libero_lila_vtt_norm_160k_20260929/smoke_validation.json)
- [仿真预检](../playground/Checkpoints/gawm_libero_lila_vtt_norm_c_160k_b160_20260929_preflight/evaluations/preflight_40rollouts/summary.json)

最终对比报告将生成于 `playground/Queues/libero_lila_vtt_norm_160k_20260929/comparison.md`。训练或评测失败时控制器记录 failed，保留日志，不将部分结果伪装为完整评测。
