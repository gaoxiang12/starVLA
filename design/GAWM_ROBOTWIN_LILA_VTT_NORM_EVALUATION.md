# LiLa＋VTT＋固定归一化 GAWM：RoboTwin Clean 评测

启动时间：2026-09-29T10:13:02.696683。**已完成：38.6%（193/500），50/50任务正常结束。** 实时状态以 [summary.json](../playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039/evaluations/clean10_seed0_step765120_lila_vtt_norm_20260929_101302/summary.json) 为准。

- Run：gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039，最终 step765120。
- Clean 50 任务 × 10 回合，seed0，共500回合；本机 GPU0–7 并行。
- 头部＋前方两路 RGB，320×240；ACT 预测32步、平滑后执行16步；状态与动作使用训练保存的统计归一化。
- 模型从训练冻结 source_snapshot 加载，严格加载权重。652个权重张量有限，固定归一化版本标记1，50任务VTT词表完整。
- 单回合 adjust_bottle 链路验证1/1成功，计数完整、无OIDN错误；仅作兼容性检查，不算正式结果。
- Supervisor PID：191371，后台独立运行，关闭会话不中断。
- 输出目录：playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039/evaluations/clean10_seed0_step765120_lila_vtt_norm_20260929_101302
- 日志：playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039/evaluations/clean10_seed0_step765120_lila_vtt_norm_20260929_101302.supervisor.log
- 启动指针：.cache/robotwin_evaluation/latest_lila_norm_eval.json（通用 latest_eval.json 亦已更新）。
- 对照：旧双相机RGB版 Clean 同预算81.4%（407/500）；新版本38.6%（193/500），下降42.8个百分点。
- 评测协议/哈希：protocol.json、source_manifest.json；源码副本：eval_source_snapshot/。

## 最终结果复核

完成于 2026-09-29T11:13:06.445101+08:00，耗时60.1分钟。全部50任务日志计数复核通过，评测协议关键字段一致，源码和权重哈希未变化。结果下降原因尚未隔离验证，不能仅归因于 latent loss。

[逐任务结果](../playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039/evaluations/clean10_seed0_step765120_lila_vtt_norm_20260929_101302/RESULTS.md)；[复核记录](../playground/Checkpoints/gawm_robotwin_head_front_lila_vtt_norm_c_20260927_210039/evaluations/clean10_seed0_step765120_lila_vtt_norm_20260929_101302/result_review.json)。

注意：两轮启动 seed 都为0，但仅35/50任务的实际有效回合种子序列完全一致，另外15任务不同；因此上述差值是同协议汇总比较，不是全部500回合严格配对比较。种子差异原因尚未核查。
