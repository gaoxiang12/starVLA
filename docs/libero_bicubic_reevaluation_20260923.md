# LIBERO bicubic 复评（2026-09-23）

当前 C 配方最终第 21,792 步 checkpoint，在保持计算精度和原协议不变的情况下，将评测图像缩放从 bilinear 改为训练使用的 bicubic，400 条全部完成。

| 套件 | 原 bilinear | 对齐 bicubic |
| --- | ---: | ---: |
| Spatial | 89/100 | 88/100 |
| Object | 96/100 | 96/100 |
| Goal | 88/100 | 86/100 |
| Long | 74/100 | 78/100 |
| 总计 | 347/400（86.75%） | 348/400（87.00%） |

总成功率增加 0.25 个百分点（净 1 条）。逐初始状态配对：325 条两次均成功，30 条两次均失败，23 条由失败变成功，22 条由成功变失败。单次快速评测不能证明稳定提升；这个变化不足以解释之前与上一轮模型的 5 个百分点差距。

## 固定的评测条件

- 同一 checkpoint、配置、归一化统计；每任务前 10 个官方初始状态，seed 7，共 40 个任务/400 次 rollout。
- 动作执行长度 8，franka 归一化，无 temporal ensemble，无旧结果复用。
- 基于原自动评测源码复制独立快照，只改 `starVLA/model/framework/WM4A/GAWM.py` 的缩放元数据；未混入后续工作区的 world-model 改动。
- FP32 模型参数和策略核心；视觉编码器 BF16 autocast，与原评测一致。
- GPU 1–6；全部任务正常退出，400 条逐条成功标记与汇总一致；每个客户端握手均确认 bicubic。结束后评测 GPU 显存已释放。

## 代码修复

GAWM 新增 `image_resize_resample` 属性，由已有策略服务元数据传给客户端。默认数据打包路径返回 bicubic；显式 `packed_image_size` 路径遵循训练的 `image_resize_resample` 配置及 bicubic 默认值。不修改旧 checkpoint 或原训练配置。

新增回归测试覆盖配置优先级和客户端两路 RGB 图像的逐像素一致性。与原动作执行测试合计 9 passed，2 subtests passed；`git diff --check` 通过。

## 计算精度建议

1. 当前评测保持 FP32 策略核心 + BF16 视觉 autocast，作为对照基线；不与图像缩放同时更改精度。
2. 若需要选择 BF16 部署，再固定 bicubic、checkpoint、400 条初始状态，单独比较策略核心 BF16/FP32 的成功率与推理耗时。当前尚未执行该精度对照，不能预先判断哪种成功率更高。
3. 后续训练建议比较 FP32 可训练参数、FP32 Adam 状态 + BF16 autocast，而不是直接把可训练参数和 Adam 状态降为 BF16。视觉冻结权重可保持已验证的精度，latent scale 等统计 buffer 保留 FP32。
4. 已有 BF16 checkpoint 转 FP32 可以在更高精度下计算，但不能恢复历史训练中已舍入的信息。精度设置应分别描述参数、优化器状态、计算 autocast 和统计 buffer，不能仅用一个“BF16”标签代表全部。
5. 本次未改变训练配方，未启动 BF16 精度对照。

## 产物

- 评测输出：`playground/Checkpoints/gawm_libero_c_12plus4_40gpu_20260922_173436/evaluations/libero4_10ep_seed7_step21792_bicubic_20260923_085955`
- 原评测基线：`playground/Checkpoints/gawm_libero_c_12plus4_40gpu_20260922_173436/evaluations/libero4_10ep_seed7_step21792_auto`
- 冻结源码及启动记录：`playground/Checkpoints/gawm_libero_c_12plus4_40gpu_20260922_173436/reevaluations/libero4_10ep_seed7_step21792_bicubic_20260923_085955`

评测目录包含 `summary.json`、逐任务日志、`bicubic_comparison.json`；控制目录包含 `provenance.json` 与 `source_snapshot/`。
