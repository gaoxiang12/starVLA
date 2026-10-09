# RoboTwin GAWM C 正式训练

> 2026-09-25：已新启动双视角RGB训练，详见 [GAWM_ROBOTWIN_2VIEW_TRAINING.md](GAWM_ROBOTWIN_2VIEW_TRAINING.md)。下文为已完成的旧单视角训练记录。

2026-09-23 21:35 首次启动；22:29（北京时间）从第 36,000 步完整断点切换至五台 32 卡。**2026-09-24 14:30 左右完成全部 765,120 步（16 epochs）。**

最终完整断点已在 `worker-1` 保存，包含模型、优化器、调度器和全部 32 份 rank RNG；最终推理权重已复制到本机。

2026-09-24 检查：拼接原始 run 的前 36,000 步与续训指标，共 38,256 条记录，每 20 步一条，无缺失、无 NaN/Inf；最终 592 个权重张量均有限。动作 L1 的早期窗口均值为 0.03129，最后 10,000 步均值为 0.009889；末段 latent loss 为 0.23497，预测误差/复制基线约 0.4788，未见明显发散。第 573,840 步阶段切换及 LR 缩放符合计划。完整数值和曲线位于 run 下 `training_loss_review.json`、`training_loss_review.png`。

- Run ID：`gawm_robotwin_c_32gpu_resume_20260923_222947`
- 节点：`worker-1` GPU 0–7、`worker-2` GPU 0–7、`worker-3` GPU 0–7、本机 `controller` GPU 1–4、`worker-4` GPU 4–7，共 32 rank。其他作业的 GPU 未占用。
- 原本机 supervisor PID：`3106011`（训练已完成）。本机曾承担 4 卡训练。
- 全局 batch：128 = 32 卡 × 每卡 4 × 累积 1。
- 数据：50 任务，27,071 episodes，6,120,962 frames，Clean + Randomized。
- 模型：冻结 DINOv3 ViT-L/16；首次训练时其余 GAWM/文本/ACT 模块随机初始化，本次扩容恢复已有权重；320×240 单头部视角，状态 16 维，动作 14 维，动作窗口 32，记录帧偏移 0/16/32。
- 训练预算：每 epoch 47,820 步；阶段一 573,840 步，阶段二 191,280 步，总计 765,120 步。阶段切换自动重置 Adam，LR 缩放到 0.2。
- AdamW：LR 2e-4，betas 0.9/0.99，WD 0.01，无 warmup，40-epoch cosine，BF16。
- 每 1,000 步保存完整断点，保留最近 2 份和阶段一终点。阶段一后继续训练，无需人工启动第二阶段。
- 数据与训练接入测试：21 项通过；50 任务/100 个真实首尾样本通过。扩容相关 10 项测试通过；三节点 4+2+2 卡测试从原 16 卡断点第 12 步恢复至第 20 步并保存完整状态，Adam step 确认为 20。
- 原始训练归档：`gawm_robotwin_c_12plus4_16gpu_20260923_213541`，状态 `stopped`，最后完整断点为 36,000 步。原始断点和源码快照保留。新任务继续使用原模型/数据/训练源码，启动器支持各节点不同卡数；节点须按全局 rank 偏移能被本节点卡数整除的顺序排列。

## 文件位置

本机与训练节点使用相同项目部署位置，以下为相对仓库根目录的 run 路径：

`playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947`

本机保存源码快照、输入配置、supervisor 状态、五节点实时日志，并每约 10 秒镜像 rank 0 的配置、统计和指标。**完整模型/优化器断点在 `worker-1` 的上述目录下 `checkpoints/`，不会持续复制到本机。** 所有 rank RNG 会汇总到 `worker-1`，恢复时启动器自动分发完整状态。

- 本机状态：`cluster_status.json`
- 本机指标：`metrics.jsonl`
- 本机日志：`logs/worker-1.log`、`logs/worker-2.log`、`logs/worker-3.log`、`logs/controller.log`、`logs/worker-4.log`
- 启动日志：`.cache/robotwin_training/gawm_robotwin_c_32gpu_resume_20260923_222947.log`
- 固定源码：run 目录 `source_snapshot/`；来源记于 `source_snapshot_origin.txt`
- 迁移说明与文件哈希：run 目录 `migration.json`；完整切换记录：`.cache/robotwin_training/cutover_step36000_20260923_222944/cutover_context.json`
- 配置：`examples/Robotwin/train_files/starvla_gawm_official_c_32gpu_resume.yaml`

## 查看进度

```bash
tail -n 1 playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/metrics.jsonl
cat playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/cluster_status.json
```

以下是历史恢复命令；当前训练已经完成，无需再次运行：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_gawm_c_cluster.py \
  --config examples/Robotwin/train_files/starvla_gawm_official_c_32gpu_resume.yaml \
  --run-id gawm_robotwin_c_32gpu_resume_20260923_222947 \
  --hosts worker-1,worker-2,worker-3,controller,worker-4 \
  --gpu-map '{"worker-1":"0,1,2,3,4,5,6,7","worker-2":"0,1,2,3,4,5,6,7","worker-3":"0,1,2,3,4,5,6,7","controller":"1,2,3,4","worker-4":"4,5,6,7"}' \
  --controller-host controller --resume
```

训练已完成且通过 loss/权重健康检查。Clean 闭环评测记录见 `GAWM_ROBOTWIN_EVALUATION.md`；训练 loss 不能代替成功率。
