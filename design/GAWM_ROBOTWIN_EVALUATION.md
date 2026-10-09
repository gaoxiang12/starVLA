# GAWM RoboTwin Clean 评测

## LiLa＋VTT＋归一化版本（2026-09-29）

最终 step765120 的 Clean 50×10 评测已完成：**38.6%（193/500）**，详见 [本轮评测记录](GAWM_L_ROBOTWIN_VTT_NORM_EVALUATION.md)。实时状态以输出目录 summary.json 为准。

## 双相机 RGB 最终权重评测（2026-09-27）

**已完成：81.4%（407/500），50/50 任务，每任务 10 回合。** 2026-09-27 09:00:40（北京时间）完成，耗时 44.7 分钟；全部任务日志计数复核通过，无执行异常。逐任务结果：`playground/Checkpoints/gawm_robotwin_2view_rgb_c_20260925_180457/evaluations/clean10_seed0_step765120_2view_rgb_20260927_081558/RESULTS.md`。

- 训练 run：`gawm_robotwin_2view_rgb_c_20260925_180457`；最终 step 765120，16 epochs。
- Clean 50 任务 × 10 回合，seed=0，共 500 回合；本机 GPU 0–7 并行。
- 真实头部＋前方相机，320×240 物理 RGB，不交换 R/B。50 任务首条 Clean 轨迹的首尾标定共 200 个样本与现有仿真机器人资产匹配。
- 38,256 条训练指标及 592 个权重张量均有限；最后 10,000 步动作 L1 均值 0.009458。
- 单回合 `adjust_bottle` 验证 1/1 成功，日志无 OIDN 错误；此结果仅验证链路。
- 输出：`playground/Checkpoints/gawm_robotwin_2view_rgb_c_20260925_180457/evaluations/clean10_seed0_step765120_2view_rgb_20260927_081558`。
- 后台 supervisor PID：`2861317`；日志：`playground/Checkpoints/gawm_robotwin_2view_rgb_c_20260925_180457/evaluations/clean10_seed0_step765120_2view_rgb_20260927_081558.supervisor.log`。
- 实时结果：输出目录 `summary.json` 及各任务 `status.json` / `eval.log`。
- 指针：`.cache/robotwin_evaluation/latest_2view_eval.json`。
- 训练预检和相机标定结果保存在 `preflight.json`；评测与相机配置的源码副本及哈希保存在 `eval_source_snapshot/`。

## 颜色对齐重评（2026-09-25 01:19）

**颜色对齐重评已全部完成：81.0%（405/500），较旧结果77.2%提高3.8个百分点。** 2026-09-25 02:01:33完成，耗时42.2分钟；本机GPU 0–7，原supervisor PID `975347`。

[完整结果与逐任务对比](../playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/evaluations/clean10_seed0_step765120_bgr_aligned_20260925_011921/RESULTS.md)。blocks_ranking_rgb：0/10 → 10/10；stack_blocks_three：0/10 → 5/10；blocks_ranking_size：5/10 → 9/10。全部50个任务计数及权重/源码哈希复核通过。

- 同一最终权重、Clean 50 任务 × 10 回合、seed=0，合计500回合。
- 在归一化前交换在线 RGB 的 R/B，匹配现有权重训练时的颜色分布；显式参数 `--image-channel-order bgr`。
- 新输出：`playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/evaluations/clean10_seed0_step765120_bgr_aligned_20260925_011921`。
- 旧77.2%结果保留；新结果81.0%，全部500回合完成。
- 当前进度：新输出下 `summary.json` 与各任务 `eval.log`；最新启动指针 `.cache/robotwin_evaluation/latest_eval.json`。
- 18张真实Clean/Randomized图像验证，与冻结训练读取逐像素一致；10项测试通过。服务端与客户端核对颜色协议。
- `baseline_comparison.json`记录旧结果及协议一致性；`eval_source_snapshot/`保存本次适配器。

## 上一轮结果与环境记录

> 2026-09-25 颜色核查更正：本轮训练多交换了一次 R/B，在线评测使用正常 RGB，存在通道错位。77.2% 为该条件下的实测值，需要对齐后重新评测。详见 [相机与颜色核查](../docs/robotwin_camera_color_audit_20260925.md)。


**评测已全部完成：77.2%（386/500），50/50 任务，每任务 10 回合。** 2026-09-24 17:35 启动，18:43 完成（北京时间），耗时 67.2 分钟。

2026-09-25 已重新核对全部日志计数与权重/源码哈希：结果一致，无评测异常。完整逐任务结果见 [RESULTS.md](../playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/evaluations/clean10_seed0_step765120_20260924_173551/RESULTS.md)，CSV 见 [task_results.csv](../playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/evaluations/clean10_seed0_step765120_20260924_173551/task_results.csv)。

这次是 Clean 每任务 10 回合的初步评测。15 个任务达到 10/10；最弱任务为 blocks_ranking_rgb、stack_blocks_three（均 0/10），place_dual_shoes、place_mouse_pad（均 3/10），stamp_seal（4/10）。

- 输出目录：`playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/evaluations/clean10_seed0_step765120_20260924_173551`
- 实时进度：上述目录 `summary.json` 及各任务的 `eval.log` / `status.json`。
- 启动日志：`playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/evaluations/clean10_seed0_step765120_20260924_173551.supervisor.log`
- 快速定位：`.cache/robotwin_evaluation/latest_eval.json`。关闭当前会话不会中断评测。

## 模型及训练健康检查

- Run：`gawm_robotwin_c_32gpu_resume_20260923_222947`。
- 最终权重：`playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/checkpoints/steps_765120_pytorch_model.pt`。
- 训练完成 765,120 步 / 16 epochs。拼接扩容前后 38,256 条指标，未发现 NaN/Inf 或缺失记录；592 个权重张量全部有限。
- 动作 L1：早期 1,000–11,000 步均值 0.03129；最后 10,000 步均值 0.009889。末段 latent loss 0.23497，预测误差/复制基线 0.4788。未发现明显发散，训练 loss 不等于闭环成功率。
- 健康报告及曲线：run 下 `training_loss_review.json`、`training_loss_review.png`。
- 实际 HDF5 观测推理：严格加载权重成功，输出有限、可重复的 `(1, 32, 14)` 动作；单样本物理动作 MAE 0.03964，仅验证接入，不代表总体精度。

## 协议

- `demo_clean`，50 个官方任务，每任务 10 个有效回合，共 500 回合，seed=0。
- 这是初步评测；官方最终汇报的默认预算是每任务 100 回合。
- 官方专家可解性筛选、任务成功判定和回合步数上限保持原样；关闭视频保存。
- 本机空闲 GPU 1–7，每张卡一个仿真及模型实例。GPU 0 有其他作业。
- 使用训练 run 的 `source_snapshot/` 构建 GAWM；冻结 DINOv3，训练参数 BF16。
- head_camera RGB，OpenCV INTER_LINEAR 缩放到 320×240，ImageNet 归一化。
- 16D 状态：左末端位姿7、左夹爪1、右末端位姿7、右夹爪1。
- 14D 绝对关节动作：左臂6、左夹爪1、右臂6、右夹爪1。
- 使用训练保存的 `dataset_statistics.json`，float32 min/max 归一化，不裁剪。预测32步，B样条平滑，每次执行16步。
- 每任务保存 `server.log`、`eval.log`、`status.json`；汇总保存 `summary.json`，协议和源码/权重 SHA256 保存 `protocol.json`。渲染 OIDN 错误或计数不完整明确记为失败。

## 环境

- 评测独立环境 `.venv-robotwin`，复用 `.venv` 的 PyTorch 2.8 / CUDA 12.8；原训练环境未修改。
- 官方 RoboTwin 固定仓库子模块 commit `13c3c47ff4312dd62484bcd51be034af55c062d1`。
- SAPIEN 3.0.0b1、MPLib 0.2.1、NumPy 1.26.4、SciPy 1.10.1、Warp 1.12.0。
- 官方 CuRobo v0.7.8 commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，源码位于 `thirdparty/RoboTwin/envs/curobo-runtime`，针对 sm_120 编译。
- CUDA 12.8 编译工具和 Vulkan loader 放在项目 `.cache/robotwin_evaluation`；Vulkan ICD 使用系统 `libEGL_nvidia.so.0`，不修改系统驱动。
- 依照 RoboTwin 官方安装说明修复 MPLib IK 条件、SAPIEN URDF/SRDF UTF-8 读取，并生成机器人配置绝对路径。
- 官方 `TianxingChen/RoboTwin2.0` 的 objects、embodiments 资产已通过 hf-mirror 下载并解压。Clean 不使用随机背景纹理。
- 首次调试回合虽成功，但发现自带 OIDN 2.0.1 不支持 Blackwell，已标为不可用于汇报。正式评测已使用 OIDN 2.3.3（官方首次增加 Blackwell 支持的版本），仍保留官方 rt / 32 spp / depth 8 / OIDN 去噪配置。

## 启动器

```bash
PYTHONPATH=. .venv/bin/python scripts/run_robotwin_hdf5_clean_eval.py \
  --checkpoint playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/checkpoints/steps_765120_pytorch_model.pt \
  --output playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/evaluations/<新的输出目录> \
  --episodes 10 --gpus 0 1 2 3 4 5 6 7 --seed 0 --image-channel-order bgr
```

评测接入、HDF5 数据协议和图像推理对齐相关测试：10 项通过。修复后 `clean_smoke_20260924_b` 的 adjust_bottle 单回合成功，日志无 OIDN 错误；该单回合仅验证链路，不代表整体成功率。

环境版本与兼容修复哈希保存在 `.cache/robotwin_evaluation/environment.json`，启动时一同写入 `protocol.json`。OIDN 版本说明：[官方发布记录](https://www.openimagedenoise.org/#version-history)。
