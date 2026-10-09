# GAWM 的 LiLa-WAM 视觉前端与 VTT

## 2026-09-27 运行更新

用户确认 RoboTwin 先使用现有头部＋前方两路。全量 VTT 已生成并通过节点哈希验证；新模型审查与 32 卡短训完成，正式训练已启动。当前配置与审查限制见 `GAWM_ROBOTWIN_LILA_VTT_TRAINING.md`。以下 2026-09-26 内容为最初实现记录，其中“尚未生成/启动”和三路数据缺项属于当时状态。

2026-09-26：新增可选视觉前端和两套训练配置。未启动新训练，也未修改现有训练 run 的源码快照。

## 架构

- 冻结 DINOv3 ViT-L/16，使用 `hidden_states[-12,-8,-4]`，保留 CLS、register 与 patch token。
- 复用项目已有的 LiLa-WAM `MultiLayerConcatFusion` 和 `VisualFeatureAdapter`：逐层 LayerNorm、拼接、投影回 1024 维；64 个可学习 query、4 层、8 头、768 维、dropout=0。
- 每个相机独立经过同一套共享视觉模块。RoboTwin 为 `head_camera,left_camera,right_camera`；LIBERO 为主视角和腕部视角。新配置要求所有实际相机都存在，不用黑图或复制图片补齐。
- 视觉适配器之后增加独立 `768→384` 投影和相机身份编码，接到现有 GAWM。query 不再解释成固定空间网格，不支持旧网格 checkpoint 的位置插值或 spatial_focus 实验。
- VTT 按任务计算全部训练示范的主相机 `last_hidden_state[:,0]` 首尾差的均值，即 `mean(CLS_last - CLS_first)`。RoboTwin 只用头部视角生成 VTT；腕部图像参与在线视觉输入。LIBERO 用主视角生成 VTT。
- VTT 的投影沿用 LiLa 的 LayerNorm → Linear(1024,768) → GELU → Linear(768,768)，再投影为 GAWM 的 384 维 goal。任务名仅用于检索训练时生成的向量，新配置不再训练文本编码器。
- VTT 向量是 checkpoint buffer，词表保存在配置，载入时校验任务指纹与向量。推理无需示范首尾图，也不会从评测回合生成 VTT；未知任务会报错。

GAWM 的 `VisualTokenLatentWorldModel` 源文件未修改：仍为当前帧预测两个未来、384 维残差预测器、4 层/6 头、原有统计更新、detach 和 loss 权重。ACT 动作头及各基准的动作/状态定义保持原配置。LIBERO 从每路 16 个网格 token 改成每路 64 个 query，因此其 token 数改变，但世界模型算法保持原样。

“视觉对齐”指每路 LiLa 的编码、融合和适配器；多视角共享、相机身份编码和接入 GAWM 的维度投影是本项目的扩展。不是换成 LiLa 的动作/未来特征解码器，也不是直接加载 LiLa 的策略权重。

## 配置

| 基准 | 配置 | 相机数与顺序 | 尺寸（宽×高） |
|---|---|---|---|
| RoboTwin | `examples/Robotwin/train_files/starvla_gawm_robotwin_lila_vtt_c_12plus4.yaml` | 头部、左腕、右腕，3 路 | 320×240 |
| LIBERO | `examples/LIBERO/train_files/starvla_gawm_libero_lila_vtt_c_12plus4.yaml` | 主视角、腕部，2 路 | 256×256 |

图像在训练、VTT 预计算和推理中统一使用物理 RGB、OpenCV INTER_LINEAR 与 ImageNet 归一化。LIBERO 客户端从服务器读取尺寸，避免先缩到旧尺寸再放大。RoboTwin 客户端按 checkpoint 中的相机列表取图；旧单路、双路配置仍可使用。

新配置使用原有 C 训练方案。多卡启动时全局 batch 仍需保持配置要求；本次未分配 GPU、未启动训练。旧配置默认继续使用网格池化和文本条件。

## VTT 准备

确认新配置的数据根目录、训练 mixture 和排除列表后，运行：

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_gawm_vtt.py \
  --config examples/LIBERO/train_files/starvla_gawm_libero_lila_vtt_c_12plus4.yaml \
  --device cpu
```

有空闲 GPU 时可将 `--device cpu` 改成 `--device cuda:0`。RoboTwin 使用对应的配置路径。

工具读取训练 loader 所选的 episode；遵守 LeRobot 的训练划分和排除列表，遍历全部训练 episode，按任务平均，不抽样 20 条。只读原始数据，不修改状态、归一化统计或轨迹。输出配置中指定的 `task_vectors_path`（JSON），并生成同目录的 `.prepared.yaml`；后续训练使用该 prepared 配置。文件已存在时拒绝覆盖，以免悄悄替换训练资产。

**当前尚未生成这两套正式训练的全量 VTT。** 单元测试使用的是明确隔离的合成测试资产。

**RoboTwin 数据仍有启动前置条件：** 旧官方 HDF5 数据只有头部＋前方两路；新 `RoboTwinGenerated/Clean` 有头部＋左右腕，但其 14 维 state 是控制目标，缺少当前 GAWM 所需的 16 维末端状态。三路配置仍指向待准备的 `RoboTwin3View_HDF5`，不能直接开始正式训练。本次没有将控制目标伪装成实测状态，也没有把前方视角重命名为腕部。详见 `ROBOTWIN_GENERATED_DATA_CHECK.md`。

旧 GAWM 策略 checkpoint 不能作为新架构的完整断点直接恢复；新 run 需重新初始化新增视觉适配器、VTT 投影等可训练参数，DINO 继续用本地预训练权重。

## 验证与来源

官方参考固定到项目已有 LiLa-WAM 版本：`65a320397faae07f311680fcb32f6c2f27dca1a0`。参考源码来自 [LiLa-WAM](https://github.com/teee000/LiLa-WAM/tree/65a320397faae07f311680fcb32f6c2f27dca1a0)，记录在 `.cache/gawm_lila_alignment/upstream.json`。

- 官方视觉模块对比：1024 维输入、3 层特征、305 token、3 路、64 query、768 维、4 层、8 头；同一权重的前向最大误差 **0**，梯度最大误差 **1.3969838619232178e-09**。
- 真实 DINO 检查：本地预训练权重、真实 HDF5 RGB 图像、CPU BF16；特征形状 `[1,1,1,3,305,1024]`，与直接调用官方取层路径的最大误差 **0**。
- 共 **48 项相关回归测试通过**。分布式统计测试的独立实例原先未设置 `sync_stats=True`，本次仅补齐测试设置；世界模型实现与当前训练快照 SHA256 均为 `f7ea4ce22331ce63b0252a7855f79f916044c65b5db0213bc15fc775a6fe9126`。
- 回归测试覆盖：两路/三路前向反向、冻结 DINO、VTT 梯度、独立相机、缺失相机拒绝、任务词表校验、checkpoint 重载、VTT 均值与配置保存、RoboTwin 在线相机顺序、LIBERO 在线尺寸/颜色协议及现有 C 配方。

复查数值对齐：

```bash
PYTHONPATH=. .venv/bin/python scripts/check_gawm_lila_vision_parity.py \
  --upstream-file .cache/gawm_lila_alignment/models/vla_model_fm.py \
  --output .cache/gawm_lila_alignment/parity.json
```

数值验证报告为 `.cache/gawm_lila_alignment/parity.json` 和 `real_dino.json`。这些验证不代表新模型的闭环成功率；尚未训练和评测该新架构。
