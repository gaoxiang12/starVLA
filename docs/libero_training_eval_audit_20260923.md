# LIBERO 训练与评测检查（2026-09-23）

结论：当前 C 任务完整完成预算，没有提前停止或 epoch 少跑。训练和评测的模型、统计、动作/状态接口一致，但存在图像缩放和计算精度差异。扩大 batch 后优化更新次数明显减少；现有证据不足以认定成功率下降仅由步数不足导致，末期训练 loss 已趋于平台。

## 实际结果

| 项目 | 上一轮 full | 当前 C |
| --- | ---: | ---: |
| run_id | gawm_libero_full_12ep_resume_20260922_115515 | gawm_libero_c_12plus4_40gpu_20260922_173436 |
| 全局 batch | 280 | 640 |
| 数据轮数 | 12 | 12+4 |
| optimizer updates | 37,380 | 21,792 |
| Spatial | 94/100 | 89/100 |
| Object | 98/100 | 96/100 |
| Goal | 93/100 | 88/100 |
| Long | 82/100 | 74/100 |
| 总计 | 367/400 = 91.75% | 347/400 = 86.75% |

两轮均为 seed 7、每任务前 10 条官方初始状态、动作执行长度 8、franka 归一化、无 temporal ensemble。逐任务日志中前 10 条 `Success` 标记与 summary 一致。上一轮部分任务复用同 checkpoint 旧评测的前 10 条；当前 C 没有复用结果。

证据目录（相对项目根目录）：

- `playground/Checkpoints/gawm_libero_c_12plus4_40gpu_20260922_173436/`
- `playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/`

具体文件：各任务 `config.full.yaml`、`cluster_status.json`、`metrics.jsonl`；C 的 `training_complete.json`、`auto_eval/status.json`；两轮 `evaluations/libero4_10ep*/summary.json` 和逐任务日志。

`design/` 下 [GAWM_LIBERO_C_TRAINING.md](../design/GAWM_LIBERO_C_TRAINING.md) / [GAWM_LIBERO_EVALUATION.md](../design/GAWM_LIBERO_EVALUATION.md) 中当前 C 尚待评测的文字已过时，以实际产物为准。

## 一致性核对

- C 评测模型文件与最终训练 checkpoint 是同一 inode；评测配置与训练 `config.full.yaml` 字节一致，归一化统计也字节一致。
- 训练和评测快照的 137 个 starVLA Python 文件逐一一致；LIBERO 和 UnifiedPretrain 的数据注册文件也一致。策略服务日志无 missing/unexpected checkpoint key 警告。
- 两路视角顺序为 primary、wrist；第三视角均用黑图补齐并标记无效，未把复制视角当作真实视角。
- 动作均为 8×7，前六维使用统一数据配置的 q99 归一化，夹爪 binary；评测通过相同训练 transform 反归一化，再将 open 概率转换为 LIBERO 的 ±1 指令。
- 状态为 8 维，按训练字段顺序使用 mean/std；评测输入为 eef position、axis-angle、两维 gripper qpos。字段中的 `state.pad` 名称不表示评测应补零。
- 当前状态与预测未来 latent 一起输入动作头；训练中的真实未来图像用于 world-model 监督，动作头消费预测未来，与推理路径一致。训练 future cadence 为 +4/+8 帧，20 Hz 对应 +0.2/+0.4 秒。
- 语言模式为 metadata；执行长度 8 与训练 action horizon 一致。

### 已确认差异 1：缩放插值

`starVLA/dataloader/gr00t_lerobot/datasets.py:1490` 对 RGB 图像调用 Pillow 默认 resize，实际为 bicubic。原视频是 256×256，训练缩到 224×224。

`examples/LIBERO/eval_files/model2libero_interface.py:64` 在服务端未提供 `image_resize_resample` 时默认 bilinear；实际 C 握手日志没有该字段，因此评测 256→224 使用 bilinear。模型内后续同尺寸 resize 无法恢复差异。

真实 Spatial 视频首帧验证：训练默认结果与显式 bicubic 完全相同；与 bilinear 的平均像素绝对差为 0.542/255，最大差为 21/255。这是确定的输入差异，成功率影响尚未测量。上一轮也使用相同评测客户端，因此不能单凭此差异解释两轮间下降。

### 已确认差异 2：参数与计算精度

C 的 `trainer.parameter_dtype=bfloat16` 会直接转换可训练参数（`starVLA/training/recipe.py:68`）。最终模型 checkpoint 中 590 个 tensor 为 BF16、2 个 buffer 为 FP32；Adam 的 exp_avg/exp_avg_sq 也是 BF16（350 个 tensor），step 计数为 FP32。

评测服务默认 `use_bf16=False`（`deployment/model_server/policy_wrapper.py:49`），构造 FP32 模型后加载权重，策略核心在 FP32 计算，视觉编码器仍在 BF16 autocast。数值路径与 C 训练不完全相同，但升为 FP32 不会恢复训练中已经舍入的权重信息。影响需单独比较，不能据此断言必然降低成功率。

原生 BF16 参数/Adam 状态也值得作为训练 loss 平台的候选原因检查。现有审计未测量逐步更新舍入，不能认定它已经造成更新停滞；若做训练对照，可保留 BF16 autocast，改用 FP32 可训练参数及 optimizer 状态。

## 是否训练步数不足

过滤后 872,087 帧，C 的全局 batch 为 640：

```text
steps_per_epoch = floor(872087 / 640) = 1362
stage1 = 1362 * 12 = 16344
stage2 = 1362 * 4 = 5448
total = 21792
anchor draws = 21792 * 640 = 13946880
```

每 epoch 丢弃 407 帧（0.0467%），采样每轮重新洗牌。完整完成标记和最终 checkpoint 均为 21,792 步；阶段二所有 Adam step 均为 5,448，与重置 Adam 状态的设计一致。不存在把 microstep 误记为 optimizer step 的证据。

上一轮全局 batch 280，每轮 ceil 得到 3,115 步，末尾补齐，共 37,380 次更新。C 样本抽取总量更多，但更新次数少 41.7%。若沿用 C 默认全局 batch 128，同样 16 epoch 会得到 109,008 步（阶段一 81,756）；本次为 40 卡显式改成 640，更新数约为默认的五分之一。这是优化配方变化，不是程序漏训练，也不代表 109,008 是必须达到的收敛阈值。

学习率调度同样影响判断：C 每阶段按 40 epoch cosine 周期，12 epoch 后清空 Adam 并把 LR 从约 1.69e-4 降为 4e-5。阶段二只跑 4 epoch，最终 LR 3.9266e-5，未走到该阶段 min LR 1e-5。此行为符合配置，不能把 16/40 直接解释为训练中断。

按每个 epoch 的日志点取均值（是记录批次的均值，不是完整训练集离线评估）：

| C epoch | action L1 | continuous action L1 | first action L1 |
| --- | ---: | ---: | ---: |
| 12 | 0.082061 | 0.090147 | 0.083773 |
| 13 | 0.075426 | 0.083488 | 0.077938 |
| 14 | 0.074189 | 0.082223 | 0.076772 |
| 15 | 0.074369 | 0.082336 | 0.076676 |
| 16 | 0.074054 | 0.082028 | 0.076594 |

末期已近平台，不能仅凭训练 loss 宣称闭环收敛，也没有证据保证简单延长能恢复到 91.75%。上一轮末期 action L1 约 0.0652，但其日志为主 rank 批次，C 为 rank 汇总，且精度/配方不同，不能当成严格同条件的误差比较。

两轮还同时改变了视觉编码器（随机初始化并训练 ViT-B → 预训练冻结 ViT-L）、LR（1e-4 → 2e-4）、betas、weight decay（1e-8 → .01）、warmup、阶段重置、参数精度、latent 统计同步等；成功率差异不能隔离归因到步数。

## 建议验证顺序

1. 固定当前最终 checkpoint，只对齐 bicubic 预处理复评相同 400 条；随后单独做策略核心 FP32/BF16 推理对照，分开量化两项差异。
2. 用相同协议评测仍保留的 stage1 checkpoint（16,344）与最终 checkpoint（21,792），检查第二阶段是否改善闭环表现。尚无这些中间 checkpoint 的闭环曲线。
3. 固定模型和数据做单变量训练对照：先检查 FP32 参数/Adam + BF16 autocast，再比较全局 batch 与更新预算。若延长训练，明确 LR schedule，并按中间 checkpoint 成功率选择，而非只追加 epoch。

这里的四套件包含在训练数据中，无独立 held-out split；400 条是快速评测。LIBERO-90 也参与训练（569,249 帧，约占总帧数 65.3%），未包含在本次四套件闭环结果中，且该本地目录实际只有 73 个任务。结果不表示未见任务泛化或完整 LIBERO-90 覆盖。

## 本次验证范围

只读审计实际配置、源码快照、checkpoint/optimizer 元数据、训练指标、已有 rollout 日志；真实视频帧 resize 比较；已有 CPU 测试 `test_libero_execute_horizon.py`、`test_eval_after_training.py`、`test_gawm_epoch_sampling.py`：11 passed、2 subtests passed。

未修改训练/评测源码，未启动新训练或闭环评测；新增此审计报告。未对全部原始演示重新做图像方向、动作时序或仿真重放审计。
