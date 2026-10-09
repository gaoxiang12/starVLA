# RoboTwin GAWM-L＋VTT 固定归一化重训

2026-09-27：按用户要求停止发生尺度漂移的旧训练，保留完整 checkpoint；补齐视觉接口归一化，验证后从头启动新训练，并持续监控。

## 修复内容与验证

- `GAWMLVisualPooler` 使用显式配置 `gawm_l_bridge_norm: fixed_layernorm`。`768→384` bridge 后、相机身份编码前，在 FP32 中计算无可训练 affine 增益的 LayerNorm，然后恢复原 dtype。这里采用当前命名，历史运行记录保留原名。
- 官方 LiLa 融合和适配器内部保持原样；世界模型、ACT、VTT 和原 loss 权重未改变。世界模型源文件与旧 run 逐字节一致。
- 未配置该字段时默认 `none`，保持旧 checkpoint 行为。新接口保存 `bridge_norm_version` 标记，严格载入会拒绝新旧模式误配，避免旧权重被静默重新解释。
- 新增训练日志 `visual_content_rms`、`visual_tokens_rms`、`latent_mse_over_delta_scale_sq`。原始 latent MSE 和复制基线误差仍保留。
- 模型/颜色/C 配方相关 25 项测试通过，监控逻辑 5 项测试通过。
- 五机 32 卡、64 步真实短训通过，输出 content RMS 始终约 1.0；完整 checkpoint 含 32 份 RNG，全部权重和指标有限。短训 run：`playground/Checkpoints/gawm_robotwin_lila_norm_smoke32_20260927_205830`。
- 停止旧训练后部分设备曾显示显存 0、无进程但利用率 100%；逐卡限时 CUDA 微量计算自检通过后状态恢复，未重置 GPU 或终止其他作业。证据见 `gpu_self_test.json`。

## 正式训练

- Run：`gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039`。
- 启动时间：`2026-09-27T21:00:39.761850`，北京时间。
- 路径：`playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039`。
- 五机 32 卡：`controller/worker-2/worker-3` 各 GPU0–7，`worker-4/worker-1` 各 GPU0–3。保持 batch 4/卡、全局 batch 128。40 卡无法按相同整数每卡 batch 保持全局 128，因此本轮使用 32 卡覆盖五机。
- 数据：官方 Clean + Randomized，50 任务、27,071 轨迹、6,120,962 帧；头部＋前方两路 RGB，16D 状态，14D 动作。新生成缺末端状态的数据不使用。
- C 配方 12+4 epochs，总 765,120 步，阶段一 573,840 步；AdamW LR2e-4、BF16、全局梯度裁剪 1.0，阶段二重置 Adam 并将 LR 乘 0.2。
- 可训练模块从头初始化，DINO 使用本地预训练权重，不载入旧 run 或短训模型状态。
- 训练配置：`examples/Robotwin/train_files/starvla_gawm_l_robotwin_head_front_vtt_norm_c_12plus4.yaml`；实际完整配置见 run 的 `config.full.yaml`。
- Supervisor PID：`3299354`；启动日志：`.cache/robotwin_lila_norm_training/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039.log`。
- 启动指针：`.cache/robotwin_lila_norm_training/latest_run.json`，通用 `.cache/robotwin_lila_training/latest_run.json` 也已更新。

## 持续监控

监控由独立后台进程 `scripts/monitor_gawm_latent.py` 每 60 秒执行，关闭会话后仍继续；训练结束/失败/停止后写入末次状态并退出。

- 状态：`playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039/latent_monitor/status.json`。
- 历史：`playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039/latent_monitor/history.jsonl`。
- 进程和日志位置记录在 `latest_run.json` 的 `monitor_pid`、`monitor_log` 字段。
- 全量记录 NaN/Inf；连续 3 条 content RMS 不在 [0.95,1.05] 时报警。
- 检查最近窗口的 raw MSE、按 delta scale 归一的 MSE、相对复制基线误差和动作 L1。
- 持续绝对 MSE >4、持续预测/复制比值 >1.5，或绝对与尺度归一误差同时显著增长时标记 warning；15 分钟无新指标或训练失败也会报警。
- 这些是诊断阈值，不是收敛/成功率判定；`healthy` 只表示当前未触发异常规则。监控只写入状态和日志，不自动终止训练。

## 旧 run 保留

旧 run `gawm_robotwin_head_front_lila_vtt_c_20260927_114502` 已确认 `status=stopped`。最后完整训练状态保留在 `playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_c_20260927_114502/checkpoints/steps_183000_training_state`，另有此前单独保留的 step60000 诊断权重。本次不删除旧实验数据。

## 未来关键帧预测能力的边界

本轮仍是当前图像＋VTT 预测 +16/+32 记录帧的视觉 token，尚无未来 RGB 解码、事件关键帧选择或动作条件推演。归一化修复尺度，不改变这一定义。详细 loss、目标变化问题及建议的固定教师/独立预测验证见 `GAWM_FUTURE_PREDICTION_OBJECTIVE.md`。

## 初始稳定性观察

2026-09-27T21:04:48.864342，已运行到 1360 步。step1000 完整 checkpoint 已保存；所有已记录数值有限，content RMS 范围 [0.9999951813369989, 0.9999993275851011]。最新 latent MSE 0.072576，预测/复制比值 0.625622，动作 L1 0.040883。后台每分钟继续监控。早期稳定不代表全程稳定或未来预测/闭环能力已得到验证。

## 2026-09-28 loss 复查

截至 step372280，latent loss 波动上升，但 content RMS 仍约 1。action_dit_loss 实际为包含 latent 的总损失，纯动作 l1_action_loss 仍缓慢改善。详见 [本轮 loss 诊断](GAWM_ROBOTWIN_NORM_LATENT_LOSS_DIAGNOSIS.md)。已保存 step371000 诊断权重和完整指标快照，当前训练未中断；工作区增加 total_loss 日志别名，后续启动生效。监控 healthy 仅代表未触发异常阈值，不代表已证实长期收敛。
