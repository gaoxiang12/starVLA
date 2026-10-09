# RoboTwin 头部＋前方数据准备

## 2026-09-27 更新

VTT 已完成生成及五机同步，`queue_status.json` 为 `complete`，50×1024 向量覆盖全部 27,071 条训练轨迹。新模型正式训练已在审查和多机短训通过后启动，详情见 `GAWM_L_ROBOTWIN_VTT_TRAINING.md`。下文等待 GPU 的内容是 2026-09-26 的准备记录。

更新：2026-09-26。用户确认先用现有头部＋前方两路训练 LiLa 视觉前端＋VTT；新生成的 Clean 数据因缺少原始实测末端状态，本次不使用。

## 数据范围与校验

使用 `playground/Datasets/LiLaWAM_RoboTwin_Official`，保持官方已审计的 Clean + Randomized 训练集合：

| 项目 | 数量 / 设置 |
| --- | --- |
| 任务 | 50 |
| 轨迹 | 27,071：Clean 2,466，Randomized 24,605 |
| 帧 | 6,120,962 |
| 物理相机顺序 | `head_camera`, `front_camera` |
| 图像 | RGB，320×240；按此数据的编码约定，OpenCV 解码后不再交换 R/B |
| 状态 / 动作 | 实测 16 维末端状态 / 14 维绝对关节目标 |
| 动作预测长度 / 图像帧偏移 | 32 / `[0, 16, 32]` |

本机本次重新校验全部轨迹：文件清单、相机帧数、状态和动作形状、有限值及 SHA256 均通过；解码两路相机的全部轨迹首尾图像，共 108,284 张。没有重新解码每一张中间帧。

真实训练加载器对每个任务第一条轨迹的首、中、尾锚点检查通过，共 150 个样本，含两路当前图像、两组未来图像、归一化状态和动作、尾帧掩码以及 50 个 VTT 任务标识。官方数据此前已完成五节点同步；本次全量数值和首尾图像复核在本机执行，VTT 提取时会再次核验远端副本。

证据：

- `.cache/robotwin_official_prepare_20260926/source_validation.json`
- `.cache/robotwin_official_prepare_20260926/validation/`：逐任务、逐轨迹记录
- `.cache/robotwin_official_prepare_20260926/loader_smoke.json`
- `.cache/robotwin_generated_prepare/official_validation.log`

## 训练配置

`examples/Robotwin/train_files/starvla_gawm_l_robotwin_head_front_vtt_c_12plus4.yaml`

启用 LiLa 视觉前端和 VTT，设两路相机与上述真实数据路径，固定预期帧数与 50 个任务标识。相对于原三相机 LiLa 配置，世界模型配置仅改变 `num_views` 和 `camera_names`；其他世界模型和动作头设置保持不变。原三相机配置保留，不能用当前数据冒充腕部视角。

## VTT 状态与自动准备

**数据校验已完成；VTT 尚待生成，因此新模型的训练准备还没有全部完成。**

2026-09-26 22:56 检查时，五台机器均有 GPU 作业。远端 `worker-4` 刚释放的四张 GPU 已被排队的 LIBERO 训练占用；VTT 提取在启动前检查资源时停止，未占用这些卡。已启动后台队列，等待 `worker-4` 的实际空闲 GPU。

- 队列脚本：`scripts/queue_robotwin_official_vtt.py`
- 校验 / 提取脚本：`scripts/prepare_robotwin_official_vtt.py`
- 工作目录：`.cache/robotwin_official_prepare_20260926`（本机和 `worker-4` 同路径）
- 实时状态：该目录的 `queue_status.json`
- 队列日志 / PID：`queue.log` / `queue.pid`
- 每 60 秒检查资源，只使用显存占用和利用率均为零的卡，最多四个分片，不终止已有作业。
- VTT 使用每条训练轨迹头部相机的首、末帧，经冻结的 DINOv3-L 提取最终层 CLS，先求每条轨迹末减首，再对同任务全部训练轨迹取平均；不使用评测轨迹。
- 所有分片通过后，自动核对任务、轨迹及帧总数，合并 50×1024 的 VTT，验证模型条件模块可以读取，然后同步到五台节点并核对 SHA256。
- 最终资产：`playground/Pretrained/gawm_vtt/robotwin_official_head_front_train_20260926.json`。
- `queue_status.json` 中 `status=complete` 且包含四个远端 `replicas` 哈希时，表示 VTT 已生成并同步完成；`failed` 时检查日志。

本次仅准备数据和训练配置，队列不会启动正式训练。启动训练前仍需结合当时 GPU 占用选择资源，并完成新配置的真实模型短步验证。
