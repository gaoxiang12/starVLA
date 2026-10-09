# GAWM 全量 LIBERO：12 epochs

2026-09-22 修复解码线程问题后，已恢复后台任务：`gawm_libero_full_12ep_resume_20260922_115515`。
训练不会随 IDE 或 SSH 终端断开而停止；主节点的独立 supervisor 持续管理五台节点。

## 本次配置

| 项目 | 值 |
| --- | --- |
| 模型 | 完整 GAWM，DINOv3 ViT-B/16 + 世界模型 + 文本编码器 + Franka ACT |
| 初始化 | 原正式任务从随机初始化开始；本次完整恢复其第 1,000 步模型、优化器、调度器和 RNG 状态 |
| 数据 | 本地全部 8 个 LIBERO 目录，可用 5,723 条轨迹，872,087 帧（原始 5,724 条 / 872,216 帧） |
| 节点 | `controller / worker-1 / worker-2 / worker-3 / worker-4` |
| GPU | 每台 1–7，共 35 卡，避开已有 GPU 0 作业 |
| Batch | 每卡 8，梯度累积 1，全局 280 |
| Epoch | 12，每轮 3,115 步，共 **37,380 步** |
| 数据采样 | 全局随机排列，不放回；每轮覆盖全部帧 |
| 尾部补齐 | 每轮重复 113 帧，使所有 rank 的 batch 数和大小一致 |
| 学习率 | 所有可训练模块峰值 `1e-4`，warmup 1,121 步，cosine 最低 `1e-6` |
| 优化器 | AdamW，betas `[0.9, 0.95]`，梯度裁剪 1.0 |
| 精度 / 分布式 | GAWM 视觉编码 BF16 autocast，其他分支 FP32；DeepSpeed ZeRO-2 |
| DataLoader | 每卡 4 workers，persistent workers，prefetch 2；每个视频解码器显式限制 1 个线程 |
| 保存 | 每 1,000 步及最后一步，包含完整训练状态 |
| 指标 | 每 10 步写 `metrics.jsonl`，包括 action / latent loss 与 epoch |

“全量”包含四个标准套件、LIBERO-90 目录和三个补充轨迹目录。LIBERO-90 目录实际
覆盖 73 个任务，仍以本地实际数据为准。没有加入 RoboTwin 或 Bridge 数据。

逐帧扫描全部 **11,448 个视频**发现 1 个损坏文件：Goal 第 82 条轨迹的腕部视频
在 105 帧后解码失败。两种 AV1 解码器均失败，原始文件与副本 SHA256 一致。
本次仅在训练索引中排除这条 129 帧轨迹，其他数据完整遍历，原始文件不修改。
排除清单：`examples/LIBERO/train_files/libero_video_exclusions.json`；
扫描报告：`.cache/gawm_full/video_decode_audit.json`。
首次正式启动 `gawm_libero_full_12ep_20260922_103654` 因该数据错误停止，
随后任务 `gawm_libero_full_12ep_20260922_104449` 在约第 1,210 步因视频解码器线程累积而停止。
当前有效任务是文首的 `resume_20260922_115515` 版本，从旧任务完整 checkpoint 1,000 恢复。
本次使用全部数据训练，未保留独立验证集；日志中的 loss 是训练指标。

原 mixture 使用有放回抽样，其长度不能直接当作完整数据轮次。
本次先沿用项目的多数据集归一化统计，再通过 `ConcatDataset` 和确定性的
epoch 随机排列遍历实际帧；Accelerate 负责跨 rank 分片。epoch 边界重新打乱，
关闭会额外消费训练 batch 的周期预测评估，因此 12 轮的步数和覆盖关系明确。

原始数据未修改，五台节点都使用项目内的
`playground/Datasets/LIBERO_FULL/` 完整副本。

## 查看进度

任务目录：

```text
playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/
  training_plan.json       # 实际帧数、batch、epoch 与总步数
  config.full.yaml         # 实际训练配置
  dataset_statistics.json  # 归一化统计
  metrics.jsonl            # 主节点训练指标
  cluster_status.json      # 集群状态、已汇总的 checkpoint、异常
  supervisor.pid          # 主节点后台管理进程
  train.pid               # 本节点 Accelerate launcher
  logs/<node>.log      # 各节点日志
  source_snapshot/         # 本次新增入口、配置和相关修复的启动时副本
  resume_source.json       # 恢复来源与步数
  resume_rank_*.json       # 35 个 rank 的模型哈希和恢复状态校验
  restart_resources_*.json # 重启后五节点 GPU / DataLoader 资源检查
  checkpoints/            # 权重及完整训练状态
```

```bash
tail -n 1 playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/metrics.jsonl
cat playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/cluster_status.json
```

已有集群看板 `scripts/cluster_dashboard.py` 会从默认目录识别该任务。
主节点有训练曲线，远端主要显示 GPU / 进程占用，因为指标只由全局 rank 0 写入。
看板的使用方法见 [docs/cluster_dashboard.md](../docs/cluster_dashboard.md)。

Supervisor 日志：
`.cache/gawm_full/gawm_libero_full_12ep_resume_20260922_115515_supervisor.log`。

## Checkpoint 与停止

各节点先在本地保存 DeepSpeed 分片。所有 rank 完成后，主节点写
`checkpoint_ready_N.json`；supervisor 随后汇总其他节点的 optimizer / RNG
分片，检查 35 份 optimizer 文件齐全，再写 `checkpoint_collected_N.json`。
恢复时以已经收集完整的 checkpoint 为准，并将完整目录分发到各节点。
模型权重文件单独不能恢复优化器与调度器状态。

需要停止本次训练时，可向 supervisor 发送 SIGTERM，它只终止该 run ID 对应
的各节点 launcher 进程组：

```bash
kill -TERM "$(cat playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/supervisor.pid)"
```

停止不额外触发保存，以最近一次已完成 checkpoint 为准。任一节点训练报错时，
supervisor 会停止该任务在其他节点的进程并标记失败；已有其他任务不受影响。

## 入口与启动验证

- 配置：`examples/LIBERO/train_files/starvla_gawm_full_12epochs.yaml`
- 训练入口：`scripts/train_gawm_epochs.py`
- 集群管理：`scripts/run_gawm_full_cluster.py`
- 新任务可用不同 `--run-id` 启动；该目录必须不存在，且须先确认 GPU 空闲。

正式启动前，`gawm_full_preflight_20260922` 使用相同数据和 batch 在五机 35 卡
完成了 3 步训练。所有 rank 完整模型状态一致，35 份 optimizer 分片自动汇总通过。
新增采样测试检查 35 rank 下每轮完整覆盖、尾部补齐和连续 12 轮重新打乱。

重启后已确认：35 张目标卡均有训练进程，每卡显存约 9 GiB；五台机器各 28 个
DataLoader worker，每个 worker 6 个总线程。所有 35 个 rank 的恢复模型哈希一致，
优化器、调度器与训练步数均为 1,000；恢复时直接跳过此前 batch，不重新解码。
检查记录位于任务目录的 `resume_rank_*.json`、`restart_resources_*.json`。

## 本次故障修复与验证

PyAV 后端没有应用 `torchvision.VideoReader(num_threads=...)` 的参数，自动解码
在单个视频上创建 256 个线程。旧解码器等待垃圾回收期间线程累积，最终在
`av.codec.context.open` 报 `av.error.MemoryError`，并非 GPU 显存不足。
现于首次解码前直接设置 PyAV codec context 的 `thread_count=1`，配置经数据集
工厂传递；原数据、目标 epoch 数、batch 和学习率计划保持不变。

单进程压力验证中，默认自动线程打开 100 次视频后达到 5,378 个线程；
修复后连续打开并读取 5,000 次，线程数保持 2，RSS 约 688 MiB 且趋于稳定。
AV1 / H.264 测试验证解码像素一致、重复时间戳与资源释放；断点采样测试验证
跳过 batch 不解码且样本顺序与连续训练一致。完整测试集 **97 passed，2 subtests passed**。
日志：`.cache/gawm_full/restart_all_tests.log`、`.cache/gawm_full/decoder_one_thread_5000.log`。

恢复命令如下（必须使用新的 run ID，并先确认目标 GPU 可用）：

```bash
source scripts/activate_env.sh
python scripts/run_gawm_full_cluster.py \
  --run-id <新的任务名> \
  --resume-from playground/Checkpoints/gawm_libero_full_12ep_20260922_104449 \
  --resume-step 1000
```

当前任务由独立后台 supervisor 管理，正常运行时无需再次执行该命令。
