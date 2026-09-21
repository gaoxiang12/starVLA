# 默认训练流程：C 配方

`starVLA/training/train_starvla.py` 现在默认使用 `trainer.recipe=c`。配方配置在
`starVLA/config/training/c_recipe.yaml`，由公共 `VLATrainer` 执行，不需要调用实验 C 的监督脚本，也不依赖 B 的检查点、评测结果或初始化哈希。

## 启动

```bash
STARVLA_PYTHON=/data/gaoxiang/Code/.venvs/starVLA/bin/python \
  bash scripts/train.sh YOUR_MODEL_AND_DATA.yaml --run_id new_c_run
```

默认 8 卡、每卡 16、累积 1，全局 batch 128。四卡时可设置 `NUM_PROCESSES=4` 并添加
`--trainer.gradient_accumulation_steps 2`，两卡则累积 4。全局 batch 不匹配时在加载模型前报错。
可用 `DRY_RUN=1` 只打印命令。也可使用 `accelerate launch --config_file starVLA/config/ddp.yaml`。

公共启动脚本不固定任何模型、数据路径或归一化方案。RoboTwin、LIBERO 和 UnifiedPretrain 的主启动脚本已切换为此默认流程。

## 默认行为

| 项目 | 默认值 |
|---|---|
| 后端 | Accelerate/DDP，BF16；可训练浮点参数转 BF16，保留冻结参数和 FP32 buffers |
| 全局 batch | 128 |
| 种子 | 配置 seed，未指定时 42；构造模型前各 rank 使用相同种子 |
| AdamW | lr=2e-4，betas=(0.9,0.99)，weight_decay=0.01，eps=1e-8，fused=false |
| 阶段一 | 12 个数据 epoch，cosine 周期 40 epoch，min_lr=5e-5，无 warmup |
| 阶段二 | 4 个 epoch，保留模型权重，清空 Adam 状态；lr/min_lr 都乘 0.2，重新开始 40 epoch cosine |
| 梯度裁剪 | 1.0，只在优化器更新边界执行 |
| 保存 | 每 1,000 步、阶段边界及结束；保存模型、优化器、调度器、RNG 完整状态 |
| 保留 | 最近 2 份完整检查点，另保留阶段一终点和 `trainer.milestone_steps` |
| 验证 | 默认关闭离线验证；启用时按帧训练要求独立 held-out split，避免验证消耗训练帧 |

DDP 通用默认允许不同 step 使用不同参数（`find_unused_parameters=true, static_graph=false`），
以支持多机器人动作头。固定计算图可通过覆盖设置为实验 C 的 `static_graph=true,
find_unused_parameters=false`。这是通用适配与固定 C 的一个明确差别。

预算由过滤后实际可训练 frame 数计算，而不是固定 40k/160k：

```
steps_per_epoch = floor(frames_per_epoch / global_batch_size)
stage1_steps = 12 * steps_per_epoch
max_train_steps = 16 * steps_per_epoch
scheduler_period = 40 * steps_per_epoch
```

例如 C 的 6,120,962 帧对应 47,820 步/epoch，阶段一 573,840 步，总计 765,120 步。
这不是对所有数据集都强制 765,120 步。日志额外记录 global batch、epoch 和累计 anchor draws。

## 按帧采样与多机器人

`sampling_mode=auto` 对同一机器人、等权数据集使用 `FrameEpochSampler`：每个 epoch 将真实 frame
整体洗牌、无放回采样，丢弃不足一个全局 batch 的尾部。完整恢复时按 epoch 和已完成 step 定位数据。

不同机器人/显式不等权任务/事件优先采样使用共享 weighted mixture；任务权重乘数据帧数，轨迹按长度采样。
不同机器人必须显式指定 `embodiment_sampling_weights` 并保持 `homogeneous_embodiment_batches=true`。
此时 epoch 是按总帧数定义的训练预算单位，不承诺每个 frame 恰好一次；机器人权重仍由配置控制。

不会把官方 HDF5 自动加入统一 mixture，也不会把单视角复制成真实三视角、将 14 维 state 改成
16 维，或修改动作时序、归一化和视角语义。默认启用 action/future 有效性 mask；通用流程的
padding 不计入监督，不推广官方单任务复现实验的尾部 clamp 例外。模型/数据语义仍由对应已审计配置定义。
GAWM 默认冻结 DINO 编码器，保留其 B/L 型号和图像配置，避免将原先微调编码器的低学习率意外
替换为策略的 2e-4。其他显式 freeze_modules 保留；默认取消旧配方的 pretrained_checkpoint，
若需热启动应显式覆盖。若需微调 DINO，应同时显式覆盖 train_encoder 和对应学习率分组。

## 配置优先级与历史复现

优先级：原模型/数据 YAML < C 管理的配方字段 < YAML 的 `training_overrides` < 命令行。
因此旧 YAML 中 40k 预算、旧学习率分组和 DeepSpeed 默认不会静默保留下来。
覆盖示例：

```yaml
training_overrides:
  trainer:
    max_train_steps: 60000
    milestone_steps: [60000, 117000, 316000]
  datasets:
    vla_data:
      per_device_batch_size: 8
  # 同时设置 trainer.gradient_accumulation_steps: 2，保持八卡全局 batch 128。
```

`max_train_steps` 是停止步数覆盖，不会把 40 epoch cosine 压缩到该步数。自定义 warmup 或其他
调度器请使用 legacy 配方。C 使用其两阶段 cosine 调度。

完整旧流程复现需显式添加 `--trainer.recipe legacy`，并使用对应旧配置/旧启动参数。
如需 DeepSpeed，请显式使用 `--trainer.distributed_backend deepspeed` 和对应 Accelerate 配置。
独立 VLM、VLN、VLA/VLM cotrain 入口没有在本次改动中迁移。

新运行使用新的 run_id；若目录已有训练配置，拒绝覆盖。
完整恢复：相同配置加 `--run_id EXISTING_RUN --trainer.is_resume true`，从最近带完成标记的完整
检查点恢复。要求 world size、全局 batch、帧数与调度预算保持一致；旧权重不能冒充完整恢复。
加载旧权重开启新训练使用 `--trainer.pretrained_checkpoint PATH --trainer.is_resume false`。

## 验证与适用范围

测试包含真实 Accelerate CPU 训练、跨进程完整保存/恢复、带 dropout 的相同输入顺序与逐位相同
最终权重、跨阶段 fresh Adam 对照、阶段检查点保留、采样选择、C 原有分片和梯度累积回归。
CPU 烟测验证流程和保存恢复；本次没有新增八卡训练来验证所有模型的 BF16 数值表现。
原 C 在冻结源码下继续运行；默认配方迁移不等于所有模型/任务已经验证相同成功率。
