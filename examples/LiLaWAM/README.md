# LiLa-WAM：starVLA 三相机训练接入

2026-09-17 新增 [RoboTwin 三相机对齐配方](ROBOTWIN_3VIEW_ALIGNED.md)：使用本地
50 任务 clean + randomized 数据、原生动作顺序、320×240 图像和官方两阶段训练预算，
通过 starVLA 通用 trainer 训练。下文旧模板的 224×224、next-recorded action、20k 步
设置保持不变，不应与新配方混用。

`framework.name: LiLaWAMTrain` 使用 starVLA 的共享 LeRobot loader、归一化、
Accelerate/DeepSpeed trainer、checkpoint 和通用策略服务器。它是三相机训练变体。
`LiLaWAM` 仍是原有官方单相机权重评测入口，两者分开注册。

## 模型与数据契约

- 冻结预训练 DINOv3-L/16，取 `[-12,-8,-4]` 层（包含 CLS/register tokens）。
- 每相机经过共享的多层特征融合及 4 层适配器，输出 64 个 token；加相机标识后，
  三路共 192 个视觉 token 与动作、当前 state、VTT、register tokens 一起进入
  12 层、768 维 DiT。每种机器人有独立的动作输入、state 输入和动作输出投影。
- 目标为 **flow matching MSE + 0.5 × future cosine loss**。未来解码器从
  DiT 更新后的条件 token 预测每个相机的密集 DINO patch 特征；推理不调用未来解码器。
- 输入图像为共享 loader 的 **224×224 RGB**。未来图像使用相同预处理。
- 预测 32 步，推理用 10 步 Euler；客户端可设置 `execute_horizon=16`。
  当前训练变体不做官方 B-spline 后处理，也没有语言模型。
- VTT 保留官方主相机首尾 CLS 差值按任务平均的定义。文本只通过
  `starVLA/task_language.py` 规范化为查表键；未知任务报错，不自动生成随机向量。
- LIBERO 两路真实图像占前两个位置，第三路补图并设 `view_valid_mask=[1,1,0]`。
  无效相机在每层 attention 中被屏蔽，也不计未来损失。RoboTwin 的顺序为
  `cam_high, cam_left_wrist, cam_right_wrist`，三个位置均有效。
- 动作 padding 同时从 attention 和 loss 排除；未来越界帧按共享
  `future_frame_valid_mask` 排除。推理不读取 action 标签、未来图像或未来有效掩码。

| 数据 | state | action（concat 顺序） | 动作索引 |
|---|---|---|---|
| LIBERO | 原有 8 维 state，mean/std | 原有 7 维末端增量＋夹爪；连续维 q99、夹爪 binary | 0…31 |
| RoboTwin | 原有 14 维 command state；关节 q99、夹爪 unit_interval | 左6关节、右6关节、左夹爪、右夹爪；绝对关节＋连续夹爪 | 1…32 |
| RoboTwin endpose | 世界系 `[L xyz,wxyz,L grip,R xyz,wxyz,R grip]`，16维 | 与上行完全相同 | 1…32 |

RoboTwin 的普通 LeRobot 数据没有实际末端反馈，不能把 command state 冒充 endpose。
`robotwin_endpose.yaml` 对接此前独立准备的 RGB 排序末端反馈数据。

本次未来监督取第 32 个**记录帧**。LIBERO 删除过 no-op，RoboTwin 也有采集分段，
视频 FPS 不保证真实控制周期。配置不伪造 `control_hz` 或物理秒数。
当前提供各 benchmark 的独立训练配方；跨 LIBERO/RoboTwin 的联合 world-model
mixture 需补充并核对真实时间戳后再注册。

## 配置与准备

配置位于 `examples/LiLaWAM/train_files/`：

| YAML | 数据范围 |
|---|---|
| `libero.yaml` | 现有 LIBERO object、goal、spatial、10 四套件 |
| `libero90.yaml` | 现有 LIBERO-90 |
| `libero_goal.yaml` | Goal 套件，已用于本轮真实烟测 |
| `robotwin.yaml` | 现有 RoboTwinGenerated clean 49 任务 |
| `robotwin_ranking.yaml` | blocks_ranking_rgb，已用于本轮真实烟测 |
| `robotwin_endpose.yaml` | blocks_ranking_rgb 的实际末端反馈输入版本 |

默认共享根目录 `/data/gaoxiang`，DINO 权重位于
`/data/gaoxiang/ckpts/dinov3-vitl16-pretrain-lvd1689m`。
修改 YAML 路径即可迁移；原 LiLa-WAM checkout 不再是训练运行时依赖。

先为所选配置准备训练统计和 VTT，例如：

```bash
CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  /data/gaoxiang/Code/.venvs/starVLA/bin/python -m examples.LiLaWAM.prepare \
  --config examples/LiLaWAM/train_files/libero_goal.yaml --device cuda
```

准备器使用共享 DataConfig/metadata，不复制训练 loader。按配置训练划分检查数值、
帧索引、轨迹重复、视频路径、动作/state 维度和任务数量；发现错误则落盘审计并停止。
每任务至少需要 20 条训练轨迹。默认按 seed=42 选每任务 20 条示范计算统计和 VTT，
这是有记录的代表性估计；`--max-episodes-per-task 0` 使用全部训练示范。
每个代表轨迹的所有相机都会做首尾帧解码检查，不声称完成所有视频的逐帧检查。

资产保存到 `/data/gaoxiang/Checkpoints/lila_starvla_assets/<配置名>/`：
`statistics.json`、`task_vectors.json`、`preparation_audit.json`。审计包含选中训练
episode、排除 ID、metadata SHA256、代表轨迹来源与质量评分。两份模型输入资产
均不使用验证集。现有 v2.1 LIBERO 若缺少 `stats.json`，准备器从已有 raw-column
`stats_gr00t.json` 增补兼容文件并记录来源；训练仍使用新算的独立统计。

## 标准训练与续训

```bash
CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
  /data/gaoxiang/Code/.venvs/starVLA/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 --main_process_port 29871 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LiLaWAM/train_files/libero_goal.yaml
```

将配置替换为 RoboTwin 等即可。默认单 GPU 微批次 2、累积 64，有效 batch=128，
AdamW 2e-4、betas=(0.9,0.99)、weight decay=0.01、clip=1.0。
每个模板的 20,000 steps 是起始预算，并非官方 11–12 epochs 的精确复现；请按
数据量和验证表现调整。多 GPU 时有效 batch 还会乘进程数，应相应调整累积。

新训练只加载 DINO，其他模块随机初始化。低学习率第二阶段应使用**新的 run_id**，
设 `trainer.pretrained_checkpoint` 为第一阶段的 starVLA 权重，
`trainer.is_resume: false`，`learning_rate.base: 4e-5`，并将调度下限明确设为
小于等于 4e-5（例如 1e-5）。这会重新建立 optimizer/scheduler。
中断恢复则使用原 run_id 和 `trainer.is_resume: true`，继承完整训练状态。
原官方 14/16 维单相机 checkpoint 不直接加载为三相机训练权重。

2026-09-14 的累计 100k 续训配置为 `train_files/libero_4suite_100k_20260914.yaml`。
它从原 20k checkpoint 继承模型和 Adam 状态，学习率在恢复点为 5e-5，平滑余弦
下降到累计 100k 的 1e-5，有效 batch 128、每 5k 保存、每 2k 离线验证。
配置中的 base LR 约 5.4222e-5 是按累计步数反解的余弦幅度，恢复时实际 LR 为
5e-5；不能把此配置当作从零训练配方。新目录保留 `continuation_manifest.json`。

检查发现原标签 20k 的 DeepSpeed 实际更新次数为 19998：旧配置在 dataloader
边界重置 Accelerate 累积计数，导致外部计数多了两步。续训以实际 optimizer
步数 19998 为准，在新目录中重设 scheduler/microstep 元数据，原 checkpoint
保持不变，并开启 `trainer.sync_with_dataloader: false`，避免后续 epoch 边界
产生部分累积。Adam moments 和 RNG 状态保留，数据迭代器不保证逐样本续接。
`prepare_continuation.py` 记录上述转换，使用同一共享 trainer；该配置选项默认
保持其他现有任务的行为。保留 40k、60k、80k、100k 权重供同条件闭环比较。

单卡 A100 40GB 的 LIBERO 吞吐优化配置见
`train_files/libero_4suite_fast_20260911.yaml`：微批次 32、累积 4，仍为有效 batch 128；
`encoder_batch_size: 64` 批量执行冻结 DINO，`efficient_views: true` 跳过无效相机的
编码、adapter 和 future decoder，并移除整个 batch 都无效的 DiT 条件 token。
三个相机槽位、所有参数形状和 loss 公式保持兼容；真实第三视角仍参与计算。
该开关默认关闭，旧配置行为保持兼容。改变微批次会改变随机采样顺序和各微批次
mask 均值的分组，因此不承诺逐步训练轨迹完全相同。

2026-09-11 空闲 A100 实测：原微批次 8、编码块 3 的前后向约 7.95 秒/128 样本；
优化后微批次 32、编码块 64 约 1.45 秒，均不含 optimizer/data 开销。
标准 DeepSpeed 完整续训实际约 2.1 秒/更新步。相同 batch 的 bf16 对照 loss 相对
差异 4.83e-6，选定 action head/fusion 梯度相对 L2 差异 3.36e-4；28 项测试通过，
含混合相机 mask、全无效 future 的损失/梯度/推理等价检查。具体记录见
`audits/throughput_20260911.json` 与 `audits/efficient_views_tests.log`。

`resume_when_saved.py` 可等待指定完整 checkpoint，在空闲 GPU 启动新 run 并恢复
模型、优化器、调度器。只有新 run 确认恢复并超过保存步数后才停止旧 supervisor，
通过 PID、创建时间、用户和命令核对目标。它不迁移尚未保存的更新；数据迭代器也
不保证逐样本续接。新 run 的 `handoff_status.json` 记录交接，原 checkpoint 保留，
新目录通过符号链接引用，需同时保留原目录。原 supervisor 接收停止信号后其状态
会显示 `failed / Signal 15`，应结合交接状态判断是否为预期换卡。

参数改动优先写入独立 YAML。当前 trainer 不识别裸 `key=value` 参数；使用
`--key value` 形式或 YAML，避免步数/累积覆盖未生效。

## 验证与部署

LIBERO 最终权重闭环评测可复用共享 websocket server/client，四张 GPU 各跑一个
suite。以下默认每任务 50 次，共 2000 次，预测 32 步、执行 8 步后重规划；环境种子
7，策略使用已有 `SeededEpisodePolicy` 固定每次 episode 的查询噪声序列。
输入为两张真实相机图像、8 维原始 state，服务端复用训练归一化并屏蔽第三相机。

```bash
/data/gaoxiang/Code/.venvs/starVLA/bin/python -m examples.LiLaWAM.run_libero_eval \
  --checkpoint /path/to/run/final_model/pytorch_model.pt \
  --output /data/gaoxiang/ckpts/lila_starvla/eval_unique_run \
  --gpus 0,1,2,3 --trials 50 --execute-horizon 8 --seed 7 --detach
```

运行中查看输出目录的 `run_status.json`，结束后查看 `results.json`。
每个 suite 保存 `server.log`、`eval.log` 和策略查询 seed/hash 日志。默认关闭视频。
任务成功率来自 LIBERO 仿真环境的 `done`，与训练器离线动作验证指标区分。

2026-09-14 推理对齐修正：LiLa 的 `_pixels` 和 LIBERO 客户端统一使用 bicubic
缩放，与共享 loader 的 RGB Pillow 默认缩放一致。服务端通过
`image_resize_resample: bicubic` 握手传递此规则，其他未声明策略保持旧客户端默认值。
LiLa 在 Euler 积分结束后将连续动作裁剪到训练 q99 标签的 `[-1,1]` 范围，
gripper 仍走已有 binary/unit_interval 反变换。参数形状、权重、训练 loss 不变。
已运行的 100k 训练进程继续使用其启动时的推理代码做离线验证；后续新启动的
部署进程使用修正版，离线验证数字跨口径比较时需注意这一点。

旧 20k 完整闭环基线为 1600/2000（80.0%）：Spatial 435/500、Object 454/500、
Goal 407/500、Long 304/500。修正版使用相同 20k checkpoint、50 次/任务、种子 7、
execute horizon 8 重评，输出目录名为 `libero_4suite_50ep_exec8_seed7_aligned_20260914`。
两项修改同时应用，本轮比较不能分别归因于缩放或动作裁剪。

```bash
NO_ALBUMENTATIONS_UPDATE=1 /data/gaoxiang/Code/.venvs/starVLA/bin/python \
  -m unittest tests.test_lila_training tests.test_lila_wam tests.test_task_language -v

CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  /data/gaoxiang/Code/.venvs/starVLA/bin/python -m examples.LiLaWAM.smoke \
  --config examples/LiLaWAM/train_files/robotwin_ranking.yaml \
  --output /data/gaoxiang/ckpts/lila_starvla_smoke/robotwin_ranking --device cuda

CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  /data/gaoxiang/Code/.venvs/starVLA/bin/python -m examples.LiLaWAM.verify_checkpoint \
  --checkpoint /path/to/run/final_model/pytorch_model.pt --device cuda
```

正式权重使用通用 `deployment/model_server/server_policy.py --ckpt_path ...`，
客户端使用已有 LIBERO／RoboTwin 适配器，输入原始 state，由服务器复用训练 transform
做归一化。LIBERO 只传两张图也可，模型根据数据契约屏蔽第三个位置。
末端反馈版本使用 `examples/RobotwinEndPose/interface.py`。
部署需一起保留配置、`dataset_statistics.json` 和对应的 VTT 文件／DINO 权重。
VTT 数值进入 checkpoint buffer；另有任务键指纹，防止同维度的不同任务表被误配。

训练日志的 `action_dit_loss` 为总目标、`latent_loss` 为未来 cosine；模型 API
另返回 `flow_matching_loss`。L1/gripper 指标来自单步去噪估计，仅作诊断，
不参与优化目标。通用验证器用完整 Euler 推理计算动作误差。
`delta_to_copy_ratio` 为预测特征位移 RMS 与真实位移 RMS 的比值，
`delta_direction_cosine` 为两者的方向余弦；两项也都排除无效未来/相机。

本轮是工程接入验证，不是 LIBERO/RoboTwin 成功率复现。完整训练和闭环成功率
需要后续实验；不能将 smoke loss 或两步训练结果当作任务表现。验证记录见
`audits/` 与共享 checkpoint 目录。原始失败日志保留，包括 CLI 覆盖未生效的
已停止烟测，以及已修复的 ZeRO bf16 native-MHA 对齐问题。

2026-09-11 验证汇总：27 项测试通过；LIBERO Goal 与 RoboTwin RGB 排序均通过
标准 DeepSpeed 两步训练、held-out 动作验证和完整 checkpoint 保存；LIBERO
从第 2 步完整续训到第 3 步。两类 trainer 权重通过 723 个张量严格检查及通用
服务器输入输出一致性检查。RoboTwin endpose 版本另通过真实数据两步梯度烟测
及部署一致性检查。完整记录见 [integration_report.json](audits/integration_report.json)。
