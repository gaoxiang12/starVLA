# 补齐数据后的 LIBERO 微调队列（2026-09-26）

用户已确认：排在当前 RoboTwin 训练之后，从此前成功率最高的 200k 权重微调 40k 步，再评测。

后台队列已启动，当前为 `waiting_robotwin`。依赖任务为 `gawm_robotwin_2view_rgb_c_20260925_180457`，目标 765,120 步。只有集群状态为 complete、完成标志及最终训练状态匹配目标步数时才允许启动；失败或中断不会被当成完成。

后续任务为 `gawm_libero_completed_ft200k_40k_after_robotwin_20260926`。初始化权重来自 `gawm_libero_c_240k_resume160k_b160_32gpu_20260924_163932/checkpoints/steps_200000_pytorch_model.pt`。本次数据和统计发生变化，因此只初始化权重，使用新的优化器与步数计数；新增训练 40,000 步，不加载旧优化器状态。

训练使用 `unified_libero_completed_20260926_wm`，完整遍历 1,027,125 帧，global batch 160，FP32 可训练参数、BF16 混合精度，DINO 冻结。学习率从 4e-5 按余弦下降，配置下限 1e-5，周期 44,933 步；40k 预算内不会触发阶段切换和优化器重置。优先使用 40 张空闲 GPU；若部分 GPU 有其他作业，可采用 32 或 20 卡并调整每卡 batch 保持全局 batch 160。不会终止其他 GPU 作业。

保存新增 20k、40k 两个里程碑。微调结束后自动评测两个检查点，每个为 Spatial / Object / Goal / Long 四个套件、每任务 10 次，共 400 次，seed 7，沿用既有 bicubic / FP32 policy / BF16 vision 评测流程。评测也等待空闲 GPU。

已完成预检：8 项队列相关测试通过；CPU 上实际构建模型并加载 200k 权重，592 个 checkpoint tensor 全部匹配，没有缺失或跳过；40 / 32 / 20 卡预算一致性通过；数据全量验收和五节点同步已通过。训练、评测代码和初始化权重已固定在队列快照中。

新增数据不是只有 LIBERO-90：90 为 577 条，10 为 100 条，Goal 为 73 条，Spatial 为 66 条，Object 为 46 条，合计 862 条。

- [实时队列状态](../playground/Queues/libero_completed_after_robotwin_20260926/status.json)
- [队列计划](../playground/Queues/libero_completed_after_robotwin_20260926/plan.json)
- [微调配置](../playground/Queues/libero_completed_after_robotwin_20260926/training_config.yaml)
- [模型预检](../playground/Queues/libero_completed_after_robotwin_20260926/preflight.json)
- [队列日志](../playground/Queues/libero_completed_after_robotwin_20260926/queue.log)
- [数据补齐报告](libero_data_completion_20260926.md)
