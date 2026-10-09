# LIBERO：固定 DINO 未来监督

2026-09-29，用户要求停止移动 latent 目标版本，修改后重新训练。

旧运行 `gawm_libero_lila_vtt_norm_c_160k_b160_20260929` 与自动评测控制器均已停止；五节点旧训练进程退出。保留 `steps_10000_pytorch_model.pt` 及完整训练状态。新目标改变了监督和优化路径，本轮从冻结 DINO 预训练权重初始化，不继承旧策略或 Adam 状态。

## 本轮模型

- 两路 RGB：agentview、eye_in_hand；256×256，OpenCV linear。LiLa 多层融合、每路64 query、384维 bridge、fixed LayerNorm、训练集 VTT 均保留。
- 当前紧凑 latent 与 VTT 输入原结构的残差预测器，预测 +0.2/+0.4 秒的未来 latent。固定单位残差系数，不再使用在线 adapter 差分的 EMA 缩放；残差加到当前 latent 后，以无仿射 LayerNorm 限定预测尺度。训练和部署调用同一 `regress_future`。
- 新增训练期特征解码器：每路共享解码器，256个空间位置 query、384维、2层、6头 cross-attention（同时含 query self-attention）。各路独立解码为 16×16×1024 DINO patch 特征；没有跳过未来 latent 的当前图像 skip connection。
- **教师是冻结 DINO 的 `last_hidden_state`，去掉 CLS/register 后的 patch features**。同一次 DINO 前向同时提供 LiLa 的中间层和最终层教师，不额外跑一遍编码器。教师不经过可训练 fusion/adapter/bridge，固定 eval，detach。
- 未来损失通过 decoder、predictor 回传到当前视觉 adapter；真实未来图像从不输入预测器或 ACT。世界模型仍为确定性、固定时间偏移、任务条件预测，没有增加动作条件或事件选择。
- 推理不执行特征解码器；保持 ACT 的当前＋预测未来输入、8步动作窗口与执行长度。

总损失：

```text
action_L1
+ 1.0 * future_DINO_patch_cosine
+ 0.1 * current_DINO_patch_reconstruction_cosine
+ 0.02 * token_diversity
+ 0.02 * token_variance
+ 0.0 * temporal_smoothness
```

当前帧重建用于约束 adapter 保留固定视觉信息。旧 adapter-space MSE 和残差方向 cosine 不再参与新版目标。不存在训练中的教师投影或 EMA 教师。

时间平滑已作为独立开关实现：按0/.2/.4秒计算相邻速度变化，用弱化大偏差的 Huber 形式惩罚；本轮权重0，单独验证固定监督。当前仅三点粗时间量，不能声称逐帧平滑，也没有事件标签。后续独立试验启用前需验证事件响应与密集时间采样。

## 训练与评测

数据、VTT、seed42、global batch160、C配方、160k预算沿用停止的实验，第一阶段77,028步，cosine名义周期256,760步。五机32卡：本机/worker-3/worker-2各8卡，worker-1/worker-4各前4卡，每卡5。先做了8卡32步检查，后做32卡32步检查。远端在旧进程退出后一度残留100%利用率读数；无显存占用的原训练卡完成微型CUDA计算后恢复空闲，其他作业未停止。

新运行：`gawm_libero_fixed_dino_c_160k_b160_20260929`。
控制目录：`playground/Queues/libero_fixed_dino_160k_20260929/`。
正式训练仅在32步短训和40任务各一次的仿真预检通过后启动；短训使用单独目录，不把预检权重当作正式策略。

计划在完整训练结束后评测40k/80k/120k/160k；四套件各10任务，每任务前10个官方初始状态，共400回合，seed7。当前控制器不是边训练边评测。

## 验证与监控

- 32项相关测试通过，包含新版固定教师梯度、未来无泄漏、无效未来mask、训练/推理预测一致、冻结编码器、严格重载及旧版本兼容。
- 8卡和32卡真实数据各32步短训检查：指标有限、预测latent RMS约1，完整优化器/RNG状态可保存。
- 第一次仿真预检因快照缺少 `deployment/` 入口失败，已补齐快照；失败日志保留。核心训练源码未因此改变。
- `dino_future_loss`：主要固定教师误差，`latent_loss` 是其兼容别名，**不再是旧版latent MSE，不可跨版本直接比较数值**。
- `dino_current_loss`、`dino_copy_loss`（复制当前冻结DINO特征）、`dino_batch_mean_loss`（同批有效目标均值，诊断用）、`dino_to_copy_ratio`、分时刻loss。
- `l1_action_loss` 才是纯动作损失；`total_loss`/历史 `action_dit_loss` 为联合损失。
- `predicted_latent_rms`、`visual_content_rms`、`latent_batch_std`、`temporal_observed_delta_rms`、`temporal_smoothness_loss` 用于尺度与信息变化诊断；方差/重建不能保证排除所有塌缩。
- 后台监控每30秒记录到 `latent_monitor.json`，对固定教师误差、动作误差及复制基线的退化报警；仅遇非有限指标时请求停止本次控制器，普通上升不自动判为发散。

当前没有独立留出数据验证，也没有闭环收益结论。固定教师消除了监督参照随adapter更新而移动的问题，不能保证紧凑latent坐标本身完全不漂移，更不能预先保证成功率恢复。

配置：`examples/LIBERO/train_files/starvla_gawm_libero_fixed_dino_c_160k.yaml`；实际卡数对应配置在控制目录 `launch_config.yaml`。源码与资产冻结在其 `source_snapshot/`，正式运行另有快照副本。

## 正式启动核对

2026-09-29 21:40（北京时间）已正式启动五机32卡训练。32步集群短训完整保存、40任务各1回合仿真链路检查均通过（未训练成熟的预检策略为0/40，仅作接口测试）。正式源码哈希与已验证快照一致。

启动核对到第300步，DINO未来损失 0.2139，动作L1 0.1520，预测latent RMS 0.999995。这是早期检查，不是收敛或成功率结论。后台监控PID见控制目录 `monitor.pid`，正式训练状态见运行目录 `cluster_status.json`。
