# GAWM 多机训练验证（2026-09-21）

2026-09-22 更新：全量 LIBERO 的 **12 epoch 正式训练已启动**，
任务配置、进度与日志见 [GAWM_FULL_TRAINING.md](GAWM_FULL_TRAINING.md)。
下文保留此前短跑验证记录。

**完整 GAWM 在五机 35 卡上训练、保存和恢复均通过。**
按用户选择随机初始化，使用真实 LIBERO-Goal 数据和项目原生 `VLATrainer`。
这是短跑功能验证，尚未进行正式长训、仿真成功率评估或吞吐调优。

## 实测配置

| 项目 | 配置 |
| --- | --- |
| 节点 | `controller, worker-1, worker-2, worker-3, worker-4` |
| 项目目录 | 所有节点 `.` |
| GPU | 每台 GPU 1–7，合计 35 × RTX PRO 5000 72GB Blackwell |
| 模型 | GAWM，102,319,511 个参数 |
| 编码器 | 完整 DINOv3 ViT-B/16，参与训练 |
| 世界模型 | 384 维、4 层、6 个注意力头，预测两个未来时刻 |
| 动作头 | 完整 Franka ACT，7 维动作、8 步 horizon、8 维状态 |
| 文本 | 项目自带 CompositionalTextEncoder，参与训练 |
| 数据 | 完整 LIBERO-Goal 副本，428 条轨迹 / 52,042 帧 |
| 输入 | 当前 / +0.2s / +0.4s 图像及状态；两路真实相机，第三路 padding 并掩码 |
| 分布式 | Accelerate 1.5.2 + DeepSpeed 0.16.9 ZeRO-2，NCCL over `eth0` |
| Batch | 每卡 1，累积 2，全局 batch 70 |
| 步数 | 首次 0→4，完整退出后恢复 4→6 |

保留统一模型的 ViT、世界模型、文本编码器和 ACT 规模，仅保留本任务需要的
Franka embodiment head；未创建没有对应训练数据的 Aloha / Bridge head。

GAWM 在视觉编码阶段使用 BF16 autocast，世界模型和动作分支显式使用 FP32。
`scripts/gawm_deepspeed.json` 因此保留 FP32 参数，不让 DeepSpeed 将整个模型
转换为 BF16。它仍使用 ZeRO-2 分片优化器。普通轻量 smoke 的全 BF16 配置与之独立。

## 验证结果

- 两机四卡 GAWM：通过；五机 35 卡 GAWM：通过。
- 每个 rank 的 16 MiB NCCL all-reduce 数值检查通过。
- 35 个 rank 均有报告，视觉编码器、世界模型、文本编码器和 ACT 权重均发生更新。
- 所有参数与持久化 buffer 的 SHA256 在各 rank 完全一致。
- 恢复前模型 hash 等于首次训练结束的 hash；优化器 step 从 4 恢复，
  DeepSpeed 步数、训练器步数与调度器 `last_epoch` 最终均为 6。
- rank 0 六步总 loss：`0.61665, 0.50065, 0.42854, 0.41066, 0.47169, 0.39164`，
  action / latent loss 均有限。这些是训练 batch 指标，不是评估集结果。
- 五机测试的 PyTorch 峰值 allocated 显存最大约 **1.76 GiB/卡**；
  该值不包含 CUDA context、NCCL 等全部进程开销，不能当作 `nvidia-smi` 总占用。
- 完整现有测试集：**85 passed，2 subtests passed**。

此次不验证策略收敛、长期稳定性、扩卡加速比，也未验证恢复后数据游标与不中断
训练逐样本一致。未使用或下载预训练权重。

## 修复的分布式问题

首次 GAWM 训练虽然完成了 4 步，但严格模型一致性检查失败。比较两节点 checkpoint
发现只有 `world_model.delta_scale` 不同（约 `0.22269` 与 `0.12423`）。
原实现仅对本 rank 的残差计算 RMS，ZeRO 的梯度同步不会同步该运行统计量。

`visual_token_delta_world_model.py` 现在先对残差平方和、有效元素数进行全局
all-reduce，再计算 RMS 和更新 EMA。按元素数汇总，避免不同 rank 的有效 mask
数量不同时简单平均 RMS 产生偏差。参数名、形状和 checkpoint 格式保持兼容。

新增两进程 Gloo 回归测试，覆盖不等有效样本数、某 rank 全被掩码、不同本地
batch 大小；随后重新完成两机与五机 GAWM 训练和恢复测试。

## 复跑

在主节点 `controller` 的项目根目录执行：

```bash
python3 scripts/run_multinode_smoke.py \
  --model gawm \
  --hosts controller,worker-1,worker-2,worker-3,worker-4 \
  --gpus 1,2,3,4,5,6,7
```

默认生成新的 run ID；若手动指定 `--run-id`，该目录必须尚不存在。
脚本只支持从 hosts 中第一台发起，远程节点需有相同路径的环境和数据。
当前节点均已部署。它会同步本次 harness、GAWM 配置及相关修复，运行 4 步，
收集并分发完整 checkpoint，然后新进程恢复至第 6 步。
每阶段默认超时 600 秒，可用 `--timeout` 调整。

仅复跑两机四卡可省略 `--hosts` 和 `--gpus`，默认使用 `controller / worker-3` 的 GPU 1、2。
启动前应检查目标 GPU 占用；此次避开 GPU 0 上已有作业。

配置文件：

- `examples/LIBERO/train_files/starvla_gawm_multinode.yaml`：完整 GAWM 与 LIBERO 数据参数。
- `scripts/gawm_accelerate.yaml`、`scripts/gawm_deepspeed.json`：多机 launcher 和 ZeRO 参数。
- `scripts/multinode_smoke.py`：调用原生模型、数据加载器和训练器，执行数值与恢复断言。
- `scripts/run_multinode_smoke.py`：协调各节点及收集 checkpoint。

## 数据与 checkpoint 存储

各节点数据位于 `.cache/multinode/data/libero_goal_no_noops_1.0.0_lerobot`。
原始数据目录未修改；其他 LIBERO 套件尚未全部分发到远端。

`/data` 是节点各自本地磁盘。DeepSpeed 配置启用
`checkpoint.use_node_local_storage=true`，各节点保存自己的 optimizer / RNG 分片。
协调脚本在首次运行结束后收集所有分片到主节点，再同步完整目录到其他节点恢复。
收集时不覆盖主节点已有文件，避免远端旧的指标或配置副本覆盖新文件。
最后第 6 步的完整分片已收集到主节点；后续再次恢复前仍需将该完整目录分发给各节点。
本次只验证相同 world size 的恢复。

主要结果目录：

```text
.cache/multinode/runs/gawm-five-nodes-35gpu-20260921/
  verification.json                 # 70 份 rank 报告汇总及最终 PASS
  metrics.jsonl                     # rank 0 的六步训练指标
  report_{initial,resume}_rank*.json # 各 rank 参数 hash、步数、模块更新、显存
  checkpoints/steps_6_training_state/ # 完整模型、优化器、调度器与 RNG 状态
  final_model/pytorch_model.pt       # 第 6 步模型权重
.cache/multinode/logs/gawm-five-nodes-35gpu-20260921/
  {initial,resume}_<node>.log    # 每节点两个阶段的日志
```

两机 GAWM 通过记录：`gawm-two-nodes-4gpu-v2-20260921`。
之前的轻量 ACT 基础验证记录：`two-nodes-4gpu-v2-20260921` 和
`five-nodes-35gpu-20260921`。带 GAWM 名称且不带 `v2` 的首次两机记录用于保留
统计量不同步的失败证据，不应作为可用 checkpoint 继续训练。
