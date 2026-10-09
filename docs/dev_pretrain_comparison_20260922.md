# dev.pretrain 与上一版本的对比

核查时间：2026-09-22。仅比较和验证，未修改训练代码、启动训练或切换分支。

## 比较基准

- 当前：`dev.pretrain`，`99ec227`，与本地 `origin/dev.pretrain` 指针一致；本次未 fetch。
- 切换前：`starVLA_dev`，`a28550d`。reflog 显示 17:16:49 提交、17:17:01 切换分支。
- 两个提交的树差异：434 个文件，新增 38,619 行、删除 2,420 行；大量内容是 RoboTwin 实验和审计工具。
- 上一轮 12 epochs 实际训练早于 `a28550d`，因此额外核对了其 `source_snapshot`、保存配置与第 37,380 步权重。切换前提交还包含训练结束后的 latent 适配改动，不能把它们都当成上一轮已启用的功能。
- 当前已跟踪文件没有未提交修改；旧集群脚本、训练配置、环境说明及部分测试仍是未跟踪文件。它们会留在工作区，但不属于当前分支提交，也不保证与新库兼容。

## 核心变化

| 项目 | 上轮实际运行 | 当前分支公共入口的默认行为 |
| --- | --- | --- |
| 入口 | 本地 `scripts/train_gawm_epochs.py` | `scripts/train.sh` / `train_starvla.py`，默认 `trainer.recipe=c` |
| 分布式 | DeepSpeed ZeRO-2，5 机 35 卡 | DDP；公共 shell 脚本为 `torchrun --standalone`，默认单机 8 卡 |
| 精度 | 可训练参数 FP32，视觉 BF16 autocast | 可训练浮点参数默认 BF16；保留冻结参数及 FP32 buffers |
| 全局 batch | 280，每卡 8 | 默认 128，每卡 16；不匹配时校验失败 |
| 预算 | 12 epochs，37,380 步 | 12 + 4 epochs 两阶段，以实际帧数计算 |
| AdamW | lr 1e-4，betas (.9,.95)，WD 1e-8 | lr 2e-4，betas (.9,.99)，WD .01 |
| 调度 | 3% warmup，12 轮 cosine 到 1e-6 | 无 warmup，40 轮 cosine 周期，min_lr 5e-5；阶段二 LR 乘 .2 并重置 Adam 状态 |
| 编码器 | 随机初始化 ViT-B/16，参与训练 | C 默认冻结 GAWM 编码器；不自动改变 B/L 架构或自动补齐预训练权重 |
| 采样 | 全局洗牌、尾部补齐，每轮覆盖全部有效帧 | 同机器人等权数据采用按帧洗牌并丢弃不满全局 batch 的尾部；不等权/多机器人采用 weighted mixture |
| 检查点 | 每 1,000 步保存，主机汇总 35 份 ZeRO 分片 | C 增加完整保存标记、配置契约校验、阶段检查点；默认只保留最近 2 份并保留指定里程碑 |

以上 C 默认只在应用 recipe 的公共入口生效。`scripts/train_gawm_epochs.py` 不调用
`apply_training_recipe`，却导入了新版 `build_accelerator`：旧 YAML 未指定 recipe/backend 时，
后端会默认变为 DDP，不能假设旧 Accelerate/DeepSpeed 启动参数仍能保持原行为。

配置优先级：原 YAML < C 管理字段 < `training_overrides` < CLI。
将旧 YAML 直接交给公共入口，batch、LR、预算、编码器冻结等都会被覆盖。
采用 40 卡时还需要显式重新确定全局 batch；默认 128 不能由 40 卡、等大小整数 batch 和整数累积直接组成。

参考：`starVLA/config/training/c_recipe.yaml`、`starVLA/training/recipe.py`、`docs/training_c_default.md`。

## 模型、数据与评测扩展

- 基础 GAWM 新增可选 SpatialFocus、局部图像裁剪、目标/物体记忆及接触监督；默认未启用。
- 新增 `GAWMCartesian`、`GAWMCompactExpert`、`GAWMOFT`、`GAWMObjectFusion`、QwenGAWM 系列及 LiLaWAM 训练/部署适配。新增类不意味着旧配置自动切换模型。
- DINO 支持本地 Hugging Face 权重路径、已归一化 tensor 输入、分块编码；pooler 支持非正方形输入网格和可选空间 embedding 尺寸适配。
- 动作头扩展输出激活配置，数据侧增加 RoboTwin 连续夹爪/下一记录帧语义、按轨迹训练验证划分、数据集独立选项和采样边界控制。
- 训练器调整普通 DDP 梯度累积中的 zero_grad/裁剪顺序，加入独立动作验证和 DataLoader 清理。不能把该改动直接认定为上轮 DeepSpeed、累积 1 的训练有同样问题。
- 评测增加 RoboTwin/LiLa-WAM 工具、受检查的归一化别名，以及由服务端元数据指定 LIBERO 图像缩放插值方式；默认仍为 bilinear。
- 新增的官方 C 实验示例使用冻结 DINO-L 和特定数据语义；它与通用 C 训练配方不是同一层概念，也不自动替换旧 LIBERO 的模型和数据。
- **当前基础 GAWM 仍没有 EMA target encoder**：EMA 仍用于 `delta_scale`，latent MSE 仍在原始特征单位计算。

## 与旧任务直接相关的兼容性问题

### 1. 旧数据集注册名不存在

`unified_libero_full_wm` 和 `unified_libero_goal_wm` 在切换前提交中存在，当前注册表没有保留。
实际导入注册表确认旧全量 YAML 的 `data_mix` 不存在。旧集群入口照搬会在数据构建阶段失败。

### 2. 视频解码线程修复未保留

`gr00t_lerobot/video.py` 不再显式设置 PyAV codec 的 `thread_count=1`；
数据集工厂也不再转发 YAML 的 `video_backend_kwargs`。
H.264 和 AV1 的两个旧回归测试都失败。上轮资源耗尽问题的防护因此不在当前路径中；
本次没有启动长训练，不能声称已复现完整的 MemoryError。

### 3. 跨卡 latent 尺度同步改为可选且默认关闭

上轮代码在分布式初始化后自动 all_reduce RMS 的平方和与有效数量。
当前通过 `framework.world_model.sync_latent_stats` 控制，默认 false；官方 C 示例显式设为 true，
但旧 LIBERO 配置没有该字段。按旧配置构造模型实测 `world_model.sync_stats=False`。
双进程 CPU 测试确认默认状态不满足原有全局 RMS 一致性要求。迁移旧多机任务时必须显式处理。

### 4. 训练后追加的 latent 适配接口缺失

相对切换前的 `a28550d`，当前没有 `condition_world_model`、状态 residual adapter、
`freeze_visual_token_pooler` 对应实现及 residual_correction 参数路径；
旧的 `scripts/experiment_gawm_latent.py` 仍留在工作区，但调用这些接口会失败。
同时，全 mask 不更新尺度、无有效方向的 horizon 不计 cosine loss 等修正也未保留。
这部分是切换前提交的训练后改动，不是上一轮 37,380 步 checkpoint 本身所需的参数。

## Checkpoint 实测

使用上一轮保存的 `config.full.yaml`，在 CPU 上构造当前基础 GAWM，严格加载
`steps_37380_pytorch_model.pt`：**All keys matched successfully**。

- 参数量：102,319,511，与上轮一致。
- SpatialFocus / contact objective：均未启用。
- 结论：上轮基础模型权重可以在当前分支按原模型配置加载。
- 边界：这不是完整训练状态恢复验证，也不是闭环评测或逐值推理等价验证。
- C 完整恢复要求自己的完成标记、配置契约和后端状态；旧 35 卡 ZeRO 分片不能直接当作新的 C/DDP 或 40 卡恢复状态。

若使用当前分支续训，应先选择旧流程兼容迁移还是新的 C 阶段热启动，补齐旧数据注册和必要修复，
明确 batch、学习率、冻结策略与 `sync_latent_stats`，再验证。单纯切换分支不等于已完成迁移。

## 本次测试与证据

运行针对性的 21 项 CPU 测试：**12 通过、9 失败**，未运行全套或 GPU 训练。

| 测试组 | 结果 |
| --- | --- |
| 新 C 配方测试 | 7 通过 |
| 旧 epoch / 断点采样测试 | 4 通过 |
| 旧解码线程限制测试 | 2 失败 |
| 旧跨进程尺度同步测试 | 1 失败 |
| 旧 latent 适配测试 | 1 通过、6 失败 |

失败项验证的是旧接口和行为未被当前分支保留，不意味着当前分支全部新增能力不可用。

日志：

- `.cache/branch_comparison/checkpoint_load.log`
- `.cache/branch_comparison/targeted_tests.log`
- `.cache/branch_comparison/distributed_tests.log`
