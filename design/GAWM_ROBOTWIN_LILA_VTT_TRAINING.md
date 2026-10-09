# RoboTwin LiLa 视觉前端＋VTT：审查与正式训练

**运行状态更新：旧 run 已按用户要求停止，最后完整 checkpoint 为 step183000。归一化修复版已从头启动，见 `GAWM_ROBOTWIN_LILA_VTT_NORM_TRAINING.md`。**

## 后续诊断修正（2026-09-27）

训练到约 6 万步发现 latent loss 持续上升。已通过固定样本和独立实验确认：新增视觉 bridge 漏掉原 GAWM 输出归一化，导致可学习特征尺度增长。此前短训通过不能证明长期稳定，原“未发现接入错误”结论需修正。详见 `GAWM_ROBOTWIN_LATENT_LOSS_DIAGNOSIS.md`。当前 run 未热改或重启。

2026-09-27 已完成审查并启动正式训练。审查未发现本次视觉/VTT 改动引入的接入错误；实现和执行链路通过验证，成功率仍需训练后闭环评测。

## 审查结论

- 冻结本地 DINOv3 ViT-L/16，取 `hidden_states[-12,-8,-4]`，保留 CLS、register 和 patch。每个真实视角独立使用同一套 LiLa 融合和 64-query、768 维、4 层/8 头适配器，没有混合当前与未来图像的信息。
- 适配器后接 `768→384` 投影和相机身份编码，接入现有 GAWM。这两项是多视角 GAWM 的接口扩展，不属于 LiLa 原模型。
- VTT 为 50 个任务全部 27,071 条训练轨迹的头部相机末帧 CLS 减首帧 CLS 的任务均值；不使用评测数据。向量作为 FP32 checkpoint buffer，投影可训练，推理不需要示范首尾图。
- 用户确认使用头部＋前方两路；真实 RGB，320×240。官方 Clean + Randomized 共 6,120,962 帧，16D 实测末端状态、14D 绝对关节目标，动作窗口 32；新生成的缺状态 Clean 批次不使用。
- `VisualTokenLatentWorldModel` 源文件与上轮训练快照逐字节一致，SHA256 `f7ea4ce22331ce63b0252a7855f79f916044c65b5db0213bc15fc775a6fe9126`。维度、深度、头数、detach、损失权重和 ACT 配置保持原方案。
- 可训练参数 58,950,702，冻结参数 303,129,600。用原配方从头训练可训练模块，DINO 载入现有预训练权重。

## 验证证据

1. LiLa 官方固定版本的视觉模块对比：前向最大误差 0，梯度最大误差 1.397e-9。
2. 本轮架构/相机/颜色/C 配方/分布式统计相关 31 项测试通过。另运行旧 latent-adaptation 实验测试：1 项通过、6 项失败；在上轮已训练模型的冻结源码上复现了完全相同的 6 项失败。
3. 真实四任务样本、真实 DINO、BF16 和 C 配方梯度裁剪下运行三步：视觉融合、适配器、桥接、VTT 投影、世界模型和 ACT 梯度均正确，DINO 无梯度。VTT 因原残差输出零初始化在第一步梯度为零，第二步起有梯度。
4. 移除外部 VTT 路径后，从 checkpoint buffer 严格重载，动作输出最大误差 0。
5. 四机 32 卡短训 32 步通过，全部指标和 651 个 checkpoint 张量有限；动作 L1 0.4574 → 0.2975。完整 checkpoint 含 32 份 rank RNG，VTT 保持原始 FP32。短训只验证链路，不代表收敛。
6. VTT、数据审计和归一化统计在训练节点的 SHA256 匹配；VTT SHA256 为 `f72abe57a1680a45e315e2648874fe2b5141b87d2189d10d0fcd229c39911375`。

审查资料保存在 `.cache/robotwin_lila_training/`，并复制到正式 run 的 `review/`。包含数值对齐、真实梯度、短训结果、当前测试与旧快照失败对照日志，以及实际 VTT 资产副本。

## 保留的已有行为与限制

- 原世界模型对被完全 mask 的未来时刻使用零余弦值，然后对全部未来时刻取平均。因此无效时刻会贡献余弦 loss 常量，并稀释有效时刻权重；不是本次视觉改动产生的行为。本轮按用户要求保持世界模型一致，不借这次视觉实验修改该损失定义。
- 原残差输出零初始化配合余弦损失，会产生较大的首步有限梯度。本轮保留 C 配方的全局梯度裁剪 1.0；真实验证后续步梯度正常，分布式短训无 NaN/Inf。
- 另外四项旧实验失败来自当前和旧快照都不具备的状态适配/残差校正接口，本轮配置与训练入口不调用这些接口。没有删除或弱化这些测试。

## 正式运行

- Run：`gawm_robotwin_head_front_lila_vtt_c_20260927_114502`。
- 启动时间：`2026-09-27T11:45:02.604396`（北京时间）。
- 路径：`playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_c_20260927_114502`。
- 四台 `controller / worker-2 / worker-3 / worker-4`，各 GPU 0–7，共 32 卡；`worker-1` 存在其他作业，未抢占。
- 每卡 batch 4，全局 batch 128，12+4 epochs；每 epoch 47,820 步，阶段一 573,840 步，总 765,120 步。
- AdamW，初始 LR 2e-4，betas 0.9/0.99，weight decay 0.01，40-epoch cosine，无 warmup；阶段二重置优化器，学习率乘 0.2。
- 每 1,000 步保存完整状态，保留最近两份及阶段边界；rank 0 本机保存 checkpoint。
- 正式 run 使用短训验证过的源码快照 `playground/Checkpoints/gawm_robotwin_lila_vtt_smoke32_20260927_114306/source_snapshot`，从头初始化，没有载入短训权重或旧双相机权重。
- Supervisor PID：`3019811`；日志：`.cache/robotwin_lila_training/gawm_robotwin_head_front_lila_vtt_c_20260927_114502.log`。
- 配置：`examples/Robotwin/train_files/starvla_gawm_robotwin_head_front_lila_vtt_c_12plus4.yaml`；完整实际参数见 run 下 `config.full.yaml`。
- 最新启动指针：`.cache/robotwin_lila_training/latest_run.json`。
- 进度：run 下 `cluster_status.json` 和 `metrics.jsonl`。

文档写入时运行至 step 660，动作 L1 0.059515，latent loss 0.635950，检查到的全部正式训练指标有限。
