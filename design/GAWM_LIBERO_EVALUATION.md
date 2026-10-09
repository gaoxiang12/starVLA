# GAWM LIBERO 评测

> 2026-09-23 更新：当前 C 配方已完成原评测 **347/400（86.75%）**；统一 bicubic 图像缩放后复评为 **348/400（87.00%）**，保持 FP32 策略核心及 BF16 视觉 autocast。详见 [复评报告](../docs/libero_bicubic_reevaluation_20260923.md)。下文等待状态和 367/400 说明保留作历史记录。

当前 C 配方任务 `gawm_libero_c_12plus4_40gpu_20260922_173436` 已挂载训练结束后的
自动评测，同样使用四套件、每任务 10 条、seed 7，共 400 次 rollout。
等待状态在该任务的 `auto_eval/status.json`；结果将在
`evaluations/libero4_10ep_seed7_step21792_auto/summary.json` 生成。
下文 **367/400** 为上一轮模型结果。新任务说明见 [GAWM_LIBERO_C_TRAINING.md](GAWM_LIBERO_C_TRAINING.md)。

2026-09-22 已完成最终第 **37,380 步** checkpoint 的四套件快速仿真评测。
训练任务：`gawm_libero_full_12ep_resume_20260922_115515`，已完成 12 epochs。

- 套件：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`（Long）。
- 快速评测：每套件 10 个任务，每任务前 10 个官方初始状态，共 400 次 rollout；seed 7。
- 动作执行长度 8，归一化键 `franka`，不使用 temporal ensemble。
- 参数保留 FP32，视觉编码器沿用模型内部 BF16 autocast。
- GPU：本机 **1、3、4、5、6、7**；GPU 0、2 已有其他作业。
- 本次 supervisor PID：`234614`；评测已完成，策略服务和仿真进程已清理。

按用户要求，原 50 条/任务的评测已主动停止，日志完整保留。
新评测复用旧日志中每个任务的前 10 条；不足 10 条的任务从初始状态 0 重跑 10 条。
汇总始终对每个任务只计前 10 条，不按成功与否筛选。
复制来的原始日志可能包含超过 10 条；以新目录的 `status.json` / `summary.json` 为准。

本次训练包含四个标准套件的数据；此结果反映这些任务上的闭环执行成功率，
不能当作未见任务泛化结果。LIBERO-90 不包含在本次四套件评测中。

## 结果与进度

最终状态 `complete`，40 个任务均完成 10 条，合计 **367/400，91.75%**。
其中 15 个任务复用旧日志的前 10 条，25 个任务新跑；已逐条核对成功标记。

| 套件 | 成功数 / 评测数 | 成功率 |
| --- | --- | --- |
| Spatial | 94 / 100 | 94% |
| Object | 98 / 100 | 98% |
| Goal | 93 / 100 | 93% |
| Long (`libero_10`) | 82 / 100 | 82% |
| 总计 | 367 / 400 | 91.75% |

这是每任务 10 条的快速结果；与每任务 50 条的结果对比时应注明评测次数不同。

目录：

```text
playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/evaluations/
  libero4_10ep_seed7_step37380_20260922_170211/
    plan.json                  # checkpoint、GPU、参数、LIBERO commit
    packages.txt               # 实际 Python 包版本
    status.json                # 每 15 秒更新：逐任务及逐套件成功数/完成数
    summary.json               # 退出时写入；status=complete 才是完整结果
    libero_*_task*.log         # 新跑 10 条或从旧评测复制的原始日志
    server_gpu*.log            # 策略服务日志
    supervisor.pid
```

Supervisor 日志：
`.cache/libero_eval/libero4_10ep_seed7_step37380_20260922_170211_supervisor.log`。
任一任务或模型服务失败会终止本次评测，写入 `error.txt` 和失败状态，清理本次子进程。

```bash
cat "$(cat .cache/libero_eval/latest_run.txt)/status.json"
```

预验证：Spatial 第一个任务完成 1/1，实际动作推理、EGL 渲染和视频保存通过；
这条预验证不计入本轮 400 次。视频在 `.cache/libero_eval/smoke_video/`。

## 环境与复跑

已在 `.venv` 增加 MuJoCo 3.2.3、robosuite 1.4.1、bddl 1.0.1、gym 0.26.2、
future、imageio-ffmpeg 等仿真依赖，原有包版本未改变。
下载使用国内镜像；LIBERO 官方源码位于 `playground/LIBERO/`。
EGL/OpenGL 加载库从系统配置的腾讯镜像获取，解压在
`.cache/libero_eval/egl/`，运行脚本自动加入库搜索路径。
LIBERO 路径配置为 `playground/LIBERO/libero/config.yaml`。
训练目录原先只有 `config.full.yaml`，已复制为评测入口需要的 `config.yaml`。

复跑时选择空闲 GPU 和新的输出目录：

```bash
source scripts/activate_env.sh
python scripts/run_libero_checkpoint_eval.py \
  --checkpoint playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/checkpoints/steps_37380_pytorch_model.pt \
  --output playground/Checkpoints/gawm_libero_full_12ep_resume_20260922_115515/evaluations/<新目录名> \
  --gpus 1,3,4,5,6,7 --trials 10
```

脚本启动前检查 GPU 占用，拒绝使用显存占用超过 512 MiB 的卡。
本轮已完成；只有需要重新评测时才执行以上命令。
