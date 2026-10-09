# GAWM LIBERO：默认 C 配方，五机 40 卡

2026-09-26 更新：补齐数据后的 LIBERO 微调已排队，等待当前 RoboTwin 成功完成后，自动从最佳 200k 权重微调 40k 步并评测。详见[队列说明](../docs/libero_completed_training_queue_20260926.md)。

> 后续训练默认已设置为 **160k 步、40 卡 × 每卡 batch 4、全局 batch 160**，已于 2026-09-23 09:22 启动。[新配置说明](../docs/libero_160k_training.md)。下文为已完成的 12+4 训练记录。

> 2026-09-23 更新：训练和评测均已完成。原评测 347/400（86.75%）；统一 bicubic 后复评 348/400（87.00%）。[结果与精度建议](../docs/libero_bicubic_reevaluation_20260923.md)。

本次使用 `dev.pretrain`（起点 `99ec227`）的公共 C 训练器，12+4 epochs。
正式任务：`gawm_libero_c_12plus4_40gpu_20260922_173436`。
启动状态以 `.cache/gawm_c/active_run.json` 和对应任务的 `cluster_status.json` 为准。

## 配置

| 项目 | 设置 |
| --- | --- |
| 模型 | 基础 GAWM / ACT，Franka 7 维动作，8 步动作窗口 |
| 视觉 | 用户指定的 ModelScope DINOv3 ViT-L/16 预训练权重，冻结 |
| 其余参数 | 随机初始化，不加载上一轮 GAWM |
| 数据 | 全部 8 个 LIBERO 目录；排除已审计损坏的 Goal episode 82，共 872,087 帧 |
| 节点 | `controller / worker-1 / worker-2 / worker-3 / worker-4` |
| GPU | 每机 0–7，共 40 卡；用户已允许本机 GPU 0 与原有作业共享 |
| Batch | 默认每卡 16，累积 1，全局 640；为 40 卡覆盖配方默认的全局 128 校验值 |
| 预算 | 每轮 1,362 步；阶段一 16,344 步，阶段二 5,448 步，总计 21,792 步 |
| 采样 | C 默认 frame_epoch，无放回洗牌，每轮丢弃不足全局 batch 的 407 帧 |
| 优化器 | AdamW，LR 2e-4，betas (.9,.99)，WD .01，无 warmup |
| 调度 | 默认 40 epoch cosine 周期（54,480 步），min LR 5e-5；阶段二 LR/min LR 乘 .2 并重置 Adam 状态 |
| 精度 | DDP / BF16，可训练参数 BF16；特征尺度 buffer FP32 |
| 保存 | 每 1,000 步、阶段边界、结束；默认保留最近 2 份及阶段一终点 |
| 验证集 | 本次全量训练，无独立留出验证；训练 loss 不等于仿真成功率 |

为控制 ViT-L 编码时的显存，视觉输入按 32 张图像分块编码；这不改变每卡训练 batch。
配置显式启用 `sync_latent_stats`，保持 40 个 rank 的尺度 EMA 一致；没有新增 EMA target encoder。

## 实现与可追溯性

- 配置：`examples/LIBERO/train_files/starvla_gawm_c_12plus4.yaml`
- 多机管理：`scripts/run_gawm_c_cluster.py`
- 各节点执行同一任务目录中的 `source_snapshot`，避免 IDE 后续改动影响长训练。
- 权重目录：`playground/Pretrained/dinov3-vitl16-pretrain-lvd1689m`。下载文件与 ModelScope 清单 SHA256 核对，保留 manifest 与 LICENSE。
- 沿用本地数据副本；修复工厂对解码参数的传递，首次解码前显式限制 PyAV codec 为 1 线程。
- 普通 DDP 模型/优化器状态由主 rank 保存；在完成标记落盘前，额外收集 40 个 rank 的 RNG 文件到主节点，支持节点磁盘不共享的场景。
- 独立 supervisor 管理五节点 launcher；任一节点失败时终止本次任务其余进程组，不匹配或终止其他作业。

## 查看与停止

读取 `.cache/gawm_c/active_run.json` 中的 `output_dir`，任务目录包含：

```text
input_config.yaml       # 输入配置
config.full.yaml        # 配方展开后的实际配置
metrics.jsonl           # 每 20 步汇总全部 rank 的指标，由主 rank 写入
cluster_status.json     # running / complete / failed / stopped
logs/<node>.log     # 各节点日志
source_snapshot/       # 本次使用的源码
git_commit.txt          # 基准提交
git_diff.patch          # 本次未提交的已跟踪文件修改
checkpoints/            # 完整训练状态和权重
```

需停止时对该任务 `supervisor.pid` 中的 PID 发送 SIGTERM；不额外触发保存。
不要对整个机器的 Python 或 torchrun 进程做批量终止。

恢复时需保持 C 配方要求的 world size、batch、数据和优化配置，将主节点完整训练状态
分发到各节点并使用当前训练器的完整恢复流程。当前集群启动脚本只创建新任务，不自动恢复。

## 启动验证

40 卡预验证 `gawm_libero_c_40gpu_preflight_20260922_173257` 完成 4 步后正常退出。
完整保存模型/优化器/调度器和 40 份 RNG 状态；415 个视觉编码器参数张量与下载权重
转为 BF16 后逐值一致，确认使用预训练且保持冻结。正式任务从同一种初始化重新开始。
C 配方、视频解码和数据选项的 17 项测试通过；新增 RNG 收集路径的完整保存/恢复
CPU 测试也通过。五机长训练的持续状态以实际日志为准。

正式任务启动后已核对五台机器每台 8 个训练 GPU 进程，共 40 卡；160 个 DataLoader
worker 均为 6 线程，supervisor PPID 为 1、独立会话。记录见任务目录
`startup_gpu_processes.json` 与 `startup_verified.json`。

## 结束后自动评测

本次任务已配置独立后台等待程序，不需要重新启动训练。
训练成功结束且第 **21,792 步**模型、优化器、调度器与 40 份 RNG 状态保存完整后，
自动选择本机空闲 GPU（显存占用不超过 512 MiB）评测；无空闲卡时继续等待。
训练失败或停止时不启动评测。

- 四套件：Spatial / Object / Goal / Long。
- 每套件 10 任务，每任务 10 条官方初始状态；seed 7，共 **400 次 rollout**。
- 动作执行长度 8，归一化键 franka，不复用旧模型的结果。
- 模型实现固定为训练源码快照；部署和评测脚本也已快照保存。
- 最终模型配套完整配置与归一化统计保存到独立评测目录，不覆盖训练配置。

任务根目录下的查看路径：

```text
auto_eval/plan.json          # 固定评测计划
auto_eval/status.json        # waiting_training / waiting_gpus / evaluating / complete / failed
auto_eval/watcher.pid        # 独立后台等待程序
auto_eval/watcher.log        # 等待程序日志
auto_eval/evaluation.log     # 评测调度日志（开始评测后产生）
evaluations/libero4_10ep_seed7_step21792_auto/status.json
evaluations/libero4_10ep_seed7_step21792_auto/summary.json
evaluations/libero4_10ep_seed7_step21792_auto/rollouts/
```

`summary.json` 在评测退出时生成，只有 `status=complete` 且 `episodes=400` 才表示完成。
等待程序的训练成功/失败判定、完整检查点判定、空闲 GPU 筛选 3 项测试通过；
同架构四步预验证 checkpoint 的策略加载、归一化和 CPU 动作推理通过。
这不是提前完成的闭环评测，最终成功率需等待训练和 rollout 结束。
