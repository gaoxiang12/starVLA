# RoboTwin GAWM C 训练方案（已启动）

2026-09-23 22:29 更新：按用户要求从第 36,000 步完整断点切换至五台共 32 卡，每卡 batch 4，全局 batch 128。运行记录见 [GAWM_ROBOTWIN_TRAINING.md](GAWM_ROBOTWIN_TRAINING.md)。模型、优化器与 12+4 epoch 预算保留，按新 rank 分配随机数流。

## 已确认条件

- 本机数据：`playground/Datasets/LiLaWAM_RoboTwin_Official/`，50 任务，27,071 条轨迹，6,120,962 帧，约 279.72 GB。
- 本次重新检查全部 27,173 个清单文件的存在性和大小，无差异。全轨迹状态/动作哈希及图像抽检详见 `ROBOTWIN_DATA_CHECK.md`。
- 启动前检查：本机各卡均有占用；`worker-4` 的 GPU 0–3 被其他作业使用。`worker-1`、`worker-2`、`worker-3` 各 8 卡及 `worker-4` 的 4–7 空闲。首次选用 `worker-1`、`worker-2` 共 16 卡；22:29 扩展到 `worker-1/worker-2/worker-3` 各 8 卡、本机 GPU 1–4、`worker-4` GPU 4–7，共 32 卡；未抢占其他作业。四台远端的数据均已完成内容校验。
- 现成 DINOv3 ViT-L/16 权重和官方动作/状态统计已经在本机。

## 建议配置

| 项目 | 方案 |
| --- | --- |
| 模型 | GAWM：DINOv3 ViT-L/16 + 视觉世界模型 + 原生文本编码器 + ACT |
| 初始化 | DINO 使用已有预训练权重并冻结，其余模块随机初始化，seed 42 |
| 数据范围 | 50 任务，Clean + Randomized 全部已审计数据 |
| 图像 | 官方头部相机单视角，320×240，保持官方预处理 |
| 状态与动作 | 16 维末端状态，14 维双臂绝对关节/夹爪动作；使用附带官方 min/max 统计 |
| 动作窗口 | 预测 32 步，后续闭环评测每次执行 16 步 |
| 世界模型 | 当前帧与记录帧偏移 +16、+32；使用公共配方的有效性 mask，避免把尾部补齐当作真实未来 |
| 设备 | `worker-1/worker-2/worker-3` 各 GPU 0–7，本机 `controller` GPU 1–4，`worker-4` GPU 4–7，共 32 卡 |
| Batch | 每卡 4，累积 1，全局 128，保持公共 C 默认优化更新规模 |
| 分布式 | Accelerate/DDP，BF16；可训练参数 BF16，尺度统计 buffer FP32；跨 rank 同步 latent 统计 |
| 优化器 | AdamW，LR 2e-4，betas (0.9, 0.99)，WD 0.01，梯度裁剪 1.0，无 warmup |
| 阶段一 | 12 epochs，573,840 步；40-epoch cosine，最低 LR 5e-5 |
| 阶段二 | 4 epochs，191,280 步；保留权重、重置 Adam，起始 LR 4e-5、最低 1e-5，重启 40-epoch cosine |
| 总预算 | 16 epochs，765,120 步；每 epoch 47,820 步、丢弃随机尾部 2 帧 |
| 保存 | 每 1,000 步、阶段边界、最终步；模型/优化器/调度器/各 rank RNG；保留最近 2 份及阶段一终点 |
| 日志 | 每 20 步记录动作、世界模型 loss、epoch、学习率、数据与计算耗时 |
| 闭环评测计划 | 阶段一和最终模型；每任务 10 条，Clean 和 Randomized 分开汇总；仿真环境另行部署验证 |

这是使用当前公共 C 配方的新训练，不承诺复现旧独立实验 C 的数值结果：旧实验采用的尾部 clamp 监督与公共配方的有效性 mask 不同。不要直接复用旧脚本中的 B provenance 校验门槛或固定八卡限制。

不按 GPU 数线性放大学习率。当前使用 32 卡每卡 4；卡更多不保证获得同比加速。若后来决定每卡 16、全局 512，则会改为每 epoch 11,955 步、总计 191,280 步；那是另一种 batch 配置，不能只改步数而忽略调度和优化更新次数。

## 已完成的启动验证

1. 已实现 `starVLA/dataloader/robotwin_official_hdf5.py`，直接使用已审计的 HDF5 清单；不依赖旧上游源码、任务 embedding 或 B 实验 provenance。
2. 初始配置：`examples/Robotwin/train_files/starvla_gawm_official_c_12plus4.yaml`；五台恢复配置：`examples/Robotwin/train_files/starvla_gawm_official_c_32gpu_resume.yaml`。官方 min/max 不裁剪，所有夹爪维度连续归一化；尾部使用有效性 mask。
3. 启动器支持远端训练、本机后台监督、日志/配置/指标回传，以及完整断点恢复。每次启动会检查所选 GPU 占用，使用独立源码快照。
4. 21 项相关测试通过；真实数据覆盖 50 个任务的 100 个首尾样本通过。
5. 16-rank 短训完成 12 步，保存模型、优化器、调度器和全部 rank RNG；从完整断点恢复到第 16 步成功。正式训练以 seed 42 重新初始化非视觉模块。

本方案使用全量数据，不设置独立留出验证集；训练 loss 不代表闭环成功率。仿真评测尚未启动。
