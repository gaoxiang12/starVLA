# RoboTwin GAWM 双视角 RGB 重新训练

2026-09-25 18:04（北京时间）启动独立后台任务。用户将三视角方案改为使用现有两路相机。

- Run：`gawm_robotwin_2view_rgb_c_20260925_180457`。
- 路径：`playground/Checkpoints/gawm_robotwin_2view_rgb_c_20260925_180457`。
- Supervisor PID：`1556023`；日志：`.cache/robotwin_2view_training/gawm_robotwin_2view_rgb_c_20260925_180457.log`。
- 最新启动记录：`.cache/robotwin_2view_training/latest_run.json`。

## 模型与数据

- 相机顺序固定为 **head_camera、front_camera**；两路均320×240，输入为正确物理RGB，ImageNet归一化。未来+16/+32记录帧分别读取同样的两路相机，不重复图像充当第二视角。
- 修正官方HDF5额外R/B交换问题：官方直接将RGB数组送入OpenCV编码，解码后保留通道顺序。
- 50任务，Clean+Randomized，27,071条轨迹，6,120,962帧。全部轨迹双相机存在性/帧数核查通过；100个任务/场景组合的200个首尾样本、共1200幅当前/未来图像解码通过。
- 冻结已有预训练DINOv3 ViT-L/16；其余GAWM、文本和ACT模块以seed42从头初始化，不加载旧单视角权重，也不续用短训权重。
- 每视角64视觉tokens，合计128；16D末端状态，14D绝对关节/连续夹爪动作；动作窗口32。

## 训练配方及设备

- 配置：`examples/Robotwin/train_files/starvla_gawm_robotwin_2view_c_12plus4.yaml`。
- 五台32卡：本机`controller`、`worker-2`、`worker-3`各GPU0–7；`worker-4`GPU0–3；`worker-1`GPU4–7。
- `worker-1`GPU0–3已有其他作业。保持全局batch128，因此使用32卡×每卡4；其余4张空闲卡未加入，不改变优化批量。
- C配方12+4 epochs；每epoch47,820步，阶段一573,840步，总765,120步。
- AdamW LR2e-4、betas0.9/0.99、WD0.01，无warmup；40-epoch cosine；阶段二重置Adam、LR乘0.2。BF16可训练参数，FP32统计buffer。
- 每1000步和阶段/最终边界保存完整模型、优化器、调度器及32个rank RNG；保留最近2份及阶段一终点。
- Rank0为本机，因此本轮完整checkpoint直接保存在本机run下，无需从远端取最终权重。

## 启动验证

- 相机读取、RGB、旧权重颜色兼容及C配方相关19项测试通过。
- 四个远端的审计和动作/状态统计哈希与本机一致，真实双相机HDF5抽检通过；此前全量文件同步校验结果继续保留。
- 五机32卡真实双视角短训8步完成，所有数值指标有限；完整状态含32份rank RNG。
- 短训run：`gawm_robotwin_2view_rgb_smoke32_20260925`，最后动作L1约0.33868。短训只验证执行链路，不表示收敛。
- 正式任务使用短训验证过的冻结源码快照，正式配置仍为完整16epochs且从头初始化。
- 本轮已启动训练；未来闭环评测须使用匹配训练的front_camera位姿和相机内参，再验证双视角RGB接入。旧单头部评测适配器不能直接套用。

## 查看进度

```bash
cat playground/Checkpoints/gawm_robotwin_2view_rgb_c_20260925_180457/cluster_status.json
tail -n 1 playground/Checkpoints/gawm_robotwin_2view_rgb_c_20260925_180457/metrics.jsonl
```

## 2026-09-27 完成检查与评测

训练已于 2026-09-26 完成全部 765,120 步。最终权重检查正常，Clean 双相机评测已启动，详情见 `GAWM_ROBOTWIN_EVALUATION.md`；实时结果位于 `playground/Checkpoints/gawm_robotwin_2view_rgb_c_20260925_180457/evaluations/clean10_seed0_step765120_2view_rgb_20260927_081558/summary.json`。
