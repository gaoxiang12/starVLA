# RoboTwin LiLa＋VTT latent loss 上升诊断

2026-09-27。对象：`gawm_robotwin_head_front_lila_vtt_c_20260927_114502`。

**主要原因是新增视觉接口缺少输出尺度约束，导致世界模型所预测的可学习特征持续放大。绝对 MSE 随目标尺度增长；现有证据不支持“预测器数值爆炸”。这是本次 LiLa→GAWM 接入中的遗漏，之前的数值对齐和 32 步短训没有验证长期尺度稳定性。**

## 正式训练证据

所有统计来自完整真实训练记录；下表按窗口取平均，避免用单个 batch 的波动比较。

| 步数窗口 | latent MSE | 复制当前特征 MSE | 比值（两列均值之比） | 目标差分 RMS | 动作 L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k–5k | 1.223 | 2.647 | 0.462 | 1.578 | 0.03390 |
| 10k–20k | 4.784 | 11.498 | 0.416 | 3.335 | 0.02132 |
| 40k–60k | 12.210 | 30.810 | 0.396 | 5.495 | 0.01588 |

- 检查到 60,140 步的全部指标均有限，无 NaN/Inf；随后 63,200 步 latent loss 14.028、动作 L1 0.01559，仍符合相同趋势。
- 相对复制基线的预测误差没有随绝对 MSE 同步恶化，动作训练误差也继续降低。因此不能仅以 raw latent loss 上升断言策略已训练崩溃。
- 日志的 `delta_to_copy_ratio` 是各 rank 比值的均值，与上表“均值之比”不完全相同。40k–60k 的前者约 0.415。
- `delta_scale` 是目标残差 RMS 的 EMA，用来放大/缩小预测残差；它没有把监督 MSE 除以方差，所以不能自动消除 loss 的量纲变化。

![完整训练指标](../.cache/robotwin_lila_training/latent_diagnosis/loss_diagnosis.png)

## 具体机制与代码位置

1. 原 `VisualTokenPooler` 在投影后执行 `out_norm`，再加入位置编码，见 `starVLA/model/framework/WM4A/GAWM.py:156`。
2. 新 `LiLaVisualPooler` 在 `self.bridge(adapted)` 后直接加入相机编码，没有对应输出归一化，见 `starVLA/model/modules/gawm_l_vision.py:91`。LiLa 适配器是 pre-norm Transformer，输入/块内归一化不等于输出幅值有界。
3. 原世界模型对 context、anchor、future target 使用 detach，当前实现位于 `starVLA/model/modules/world_model/GAWM.py` 的 `VisualTokenLatentWorldModel`。视觉前端仍受动作和正则分支训练，但 latent 监督不会把目标的尺度压回去。
4. 视觉 diversity loss 基于余弦，基本不约束整体幅值；variance loss 只防止方差过小，超过阈值即为零，本轮一直几乎为零。因此没有有效的输出尺度上界。
5. LiLa 原版未来特征监督是冻结 DINO 特征上的余弦损失。把它的可学习视觉适配器接到 GAWM 原有可学习 latent 的 raw MSE 目标，需要额外维持 GAWM 的接口尺度约定。逐模块数值对齐并不能验证这种跨模块约定。

世界模型源码确实与上一轮一致，但它接收到的特征分布已经改变。此前“世界模型代码未改”不足以推导整体系统稳定，原审查结论应按本报告修正。

## 固定样本与梯度验证

在空闲 `worker-1` GPU 0 上独立执行；正式训练四节点未改动。使用固定 32 个真实样本，冻结 DINO 特征缓存，比较 seed42 重新初始化和保留的 step60000 checkpoint。

| 特征阶段 | 初始化 RMS | step60000 RMS |
| --- | ---: | ---: |
| 多层融合输出 | 0.584 | 2.508 |
| LiLa 适配器输出 | 1.466 | 11.628 |
| 384D bridge 输出，即世界模型 content | 0.840 | 15.128 |

相同样本上，step60000 的 latent MSE=13.066、复制基线 MSE=33.213，动作 L1=0.01426。

- checkpoint 的全部 `backbone.*` 张量与初始化时的预训练编码器逐一相等，排除 DINO 被意外训练。
- 对 `latent_loss + 0.1 * latent_cosine_loss` 求梯度：visual bridge 和 adapter feature projection 均为 `None`；VTT bridge 和世界模型输出层有梯度。
- 对总 loss 求梯度：visual bridge 和 adapter 均有非零梯度，说明它们持续被其他目标更新，尺度却不由 latent MSE 直接约束。
- 没有证据表明这次增长来自相机通道、损坏数据、VTT 文件变化或旧尾帧余弦常量；主要证据集中在可学习视觉输出的幅值。

## 归一化机制对照

两个实验均从相同初始化开始，用相同的 32 个真实训练样本、相同随机采样序列、每步 batch4、600 步、BF16 AdamW 和梯度裁剪 1.0。唯一干预是在 `768→384` bridge 后、相机编码前增加无可训练仿射参数的 LayerNorm。世界模型未改。

| 600 步结果 | 原 bridge | bridge 后固定 LayerNorm |
| --- | ---: | ---: |
| 世界模型输入 content RMS | 4.283 | 1.000 |
| raw latent MSE | 0.5274 | 0.0491 |
| 复制基线 MSE | 1.1795 | 0.0791 |
| 预测/复制比值（batch 均值） | 0.5413 | 0.6520 |
| 动作 L1 | 0.04787 | 0.05013 |

**归一化控制了尺度，但两列 MSE 的特征量纲不同，数值降低不能解释成预测质量提高。** 归一化实验的相对预测比值还略高，动作误差接近；这是训练样本上的小实验，不是独立验证集、全数据分布式重训或闭环评测。归一化之前的适配器/bridge 幅值仍可变化，本实验只保证送入世界模型的 content 尺度。

![归一化对照](../.cache/robotwin_lila_training/latent_diagnosis/normalization_ablation.png)

## 建议修复路径

- 保留 LiLa 官方融合/适配器内部结构，在 GAWM 接口 bridge 后、相机身份编码前增加显式输出 LayerNorm。固定 affine=False 能直接约束尺度；使用可训练 affine 时需额外监控其增益。
- 保持原世界模型、VTT 和 ACT 定义。新增 content RMS、target-delta RMS、按目标尺度归一的 MSE 与预测/复制比值，避免只观察 raw loss。
- 修复配置单独标记，旧 checkpoint 默认保持旧接口。不要给已经训练到大幅值特征的 checkpoint 静默加归一化后直接续训，因为世界模型和 ACT 都已适应原尺度。更稳妥的是保留当前 run 作为对照，用新 run 从头做较长稳定性验证，再进行完整训练。
- 32 步链路测试仍有价值，但不能替代数千步的尺度趋势检查；后续启动检查应增加该项。

**当前正式训练继续运行，未修改代码快照、未停训、未重启；上述归一化只在隔离实验中通过 hook 实现。** 保存了 step60000 权重以防滚动清理。该结论不是“继续长训一定会发散成 NaN”，而是已确认接口尺度漂移，继续原 run 无法验证修复方案。

诊断证据目录：`.cache/robotwin_lila_training/latent_diagnosis/`。包含 `checkpoint_probe.json`、`metric_windows.json`、`ablation_summary.json`、两个实验的逐阶段结果、保留 checkpoint、探针脚本及日志。全部操作未修改原始训练数据。
