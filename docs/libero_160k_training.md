# LIBERO 160k 训练配置

2026-09-23：新增 `examples/LIBERO/train_files/starvla_gawm_c_160k.yaml`，并将
`scripts/run_gawm_c_cluster.py` 的默认配置切换到该文件。已于 2026-09-23 09:22 启动正式训练。
旧的 12+4 配置、已完成训练和评测产物保留。

## Batch 与数据预算

推荐五机 40 卡，每卡 batch 4，梯度累积 1，全局 batch 160。
沿用 872,087 帧的八目录 LIBERO mixture、损坏 episode 排除及无放回 frame_epoch 采样。
该推荐兼顾用户优先使用 40 卡的偏好与更新次数，并非吞吐实测得出的最优 batch。

| 全局 batch | 40 卡每卡 batch（累积 1） | 160k 步对应近似数据轮数 |
| --- | ---: | ---: |
| 80 | 2 | 14.68 |
| **160** | **4** | **29.36** |
| 320 | 8 | 58.72 |
| 640 | 16 | 117.47 |

全局 128 需要例如 32 卡 × 4 或 8 卡 × 16，不适用于 40 卡等 batch 的直接分配。
只有 20 张卡时，每卡 8 可保持全局 160；10 张卡时每卡 16。调整卡数必须一起调整配置，
启动器按实际卡数、配方展开后的每卡 batch 和梯度累积检查，不再硬编码每卡 16。

```text
steps_per_epoch = floor(872087 / 160) = 5450
max_train_steps = 160000 optimizer updates
anchor draws = 160000 * 160 = 25600000
```

每个完整 epoch 丢弃 87 帧，下一轮重新洗牌；停止点约为第 29.36 轮。
预算指新训练总共 160k，未设置从旧 21,792 步 checkpoint 继续训练。

## 配方与精度

- 架构沿用预训练冻结 DINOv3 ViT-L、GAWM/ACT、8 步动作窗口；策略部分重新初始化。
- 可训练参数 FP32，Adam 状态 FP32，训练计算采用 BF16 autocast；统计 buffer 保留 FP32。
- AdamW LR 2e-4、betas (.9,.99)、WD .01，保持 C 无 warmup 的设计。
- 保留阶段一 12 epoch：第 65,400 步后重置 Adam 状态，阶段二起始 LR 4e-5。
- `stage_epochs=[12,18]` 是名义预算；显式 `max_train_steps=160000` 提前截断第二阶段，
  第二阶段实际 94,600 步（约 17.36 epoch）。没有保留原来的 4 epoch 停止限制。
- 两阶段各使用 40 epoch cosine 周期，即 218,000 步；第 160k 步 LR 约 2.809e-5，
  未达到阶段二 min LR 1e-5。这是保留 C 调度规则的扩展，不是完整 160k 单周期 cosine。
- 每 10k 步保存；保留最近 2 份、阶段一终点及 40k/80k/120k/160k 里程碑，便于闭环比较。
- 沿用全量训练，无独立验证集；epoch 增加不保证成功率提高。

## 查看与启动

仅展开配置、无 GPU/SSH 操作、不创建训练目录：

```bash
.venv/bin/python scripts/run_gawm_c_cluster.py --run-id libero_160k_config_check --dry-run
```

后续启动时先检查五台机器 GPU 占用，并使用新的 run_id。启动命令：

```bash
source scripts/activate_env.sh
python scripts/run_gawm_c_cluster.py --run-id gawm_libero_c_160k_b160_40gpu_<时间戳>
```

卡数改变时需配置相应的 per-device batch，不能沿用不匹配的 160 校验值。
既有自动评测 watcher 固定指向旧任务第 21,792 步，不会自动迁移到新任务。

## 验证

新配置真实 recipe 展开和启动器 dry-run 通过；配置预算、累积 batch、旧配置兼容、卡数不匹配拒绝、
FP32 参数/Adam 状态及 frame_epoch 采样测试共 9 项通过。未进行 40 卡吞吐或显存实测。

## 当前正式任务

- run_id：`gawm_libero_c_160k_b160_40gpu_20260923_092237`
- 状态：启动核对时 running，已到第 340 步；最终状态以任务目录为准。
- Supervisor PID：`1950996`，PPID 1，独立会话。
- 五台节点各 8 个本任务 GPU 进程，40 卡；保留此前获准共享的本机 GPU 0 策略服务。
- 预验证：`gawm_libero_c_160k_b160_40gpu_preflight_20260923_092107`，4 步完整保存、40 份 RNG、FP32 策略参数和 Adam 状态核对通过。
- 正式源码快照与预验证一致；阶段切换 65,400 步，最终停止 160,000 步。
- 任务目录：`playground/Checkpoints/gawm_libero_c_160k_b160_40gpu_20260923_092237`。
- 入口指针：`.cache/gawm_c/active_run.json`；进度读 `cluster_status.json`、`metrics.jsonl`；启动记录 `startup_verified.json`。
- 正式训练首次周期 checkpoint 在 10,000 步；此次仅验证启动，尚未完成 160k。
