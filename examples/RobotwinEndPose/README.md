GAWM 实际末端反馈对照，2026-09-11

本实验改变 **state 输入**，输出仍为 GAWM 原有的 14 维绝对关节与连续夹爪动作，预测 16 步、执行 16 步后重观察。没有接入 IK，也没有训练 endpose action head。LiLa 发布模型基线由另一会话负责，这里不修改其文件或任务。

| 组别 | 输入 state（归一化前） | 输出 |
|---|---|---|
| command | `[左6关节指令,0,左夹爪,右6关节指令,0,右夹爪]`，16 维 | `[左6关节,右6关节,左夹爪,右夹爪]`，16×14 |
| endpose | `[左xyz,wxyz四元数,左夹爪,右xyz,wxyz四元数,右夹爪]`，16 维 | 与 command 完全一致 |

末端数据为 RoboTwin 原生 `get_left/right_ee_pose` 所记录的世界系 EE 位姿，位置单位米，旋转顺序 wxyz；它不是任意工具定义的 TCP，也不是对关节 drive target 做 FK。夹爪仍是原有连续控制字段，不是实际接触传感器。关节对照补两个零仅用于匹配 state 投影容量，不伪造测量信息。本次先做这两个条件，目标关节 FK 的第三个条件尚未实现。

实现复用现有 GAWM、ACT、共享 LeRobot loader、归一化、策略服务器及固定场景评分循环，仅新增数据注册、反馈打包函数和独立环境适配器。训练和部署共同调用 `starVLA/robotwin_feedback.py`；原生环境关节顺序由已有客户端转换，state 与 action 各自有独立顺序。

**数据与初始化**

数据位于 `/data/gaoxiang/RoboTwinEndPose/Clean/blocks_ranking_rgb`。从现有原始 HDF5 逐帧提取 endpose，为 Parquet 增加两种反馈列；验证原始 command 与旧 Parquet 的 action/state 完全相同，frame_index 连续、帧数一致、四元数有效、夹爪一致。视频使用指向旧数据的逐文件符号链接，没有重新编码。原始轨迹和视频不变。

训练仍是原来的 **475 条、219,148 帧**。原离线验证 20 条中，10 条来自合并后的 incoming 数据，其原始 HDF5 在本机不可用；两个条件共同使用余下的 **10 条验证轨迹**。缺失 ID 为 `500,520,600,620,640,660,680,700,720,740`，没有用测试场景替换它们。新的 state 统计仅由 475 条训练轨迹计算，action 统计完整沿用旧 v2。关节/位姿特征使用共享 q99 transform，夹爪使用 unit_interval。

固定闭环 **20 开发＋100 测试场景**仍取自 `qwen_gawm_ranking_20260910/protocol.json`，不受离线验证减少影响。完整排序使用原始 `blocks_ranking_rgb.check_success()` 与 1200 步动作上限。两组场景必须核对初始 RGB 哈希、积木位置、姿态和尺寸一致；初始化或服务错误停止实验，不换 seed 或计作策略失败。

两组共同从旧 `gawm_rgb_focus_local_v2_5k_20260907` 初始化。因 state 的宽度和语义变化，完整重置 `action_models.aloha.state_projection` 与 `spatial_focus.query`，共 13 个张量；其余 564 个张量继承旧模型。两组使用同一个生成的 checkpoint、同一随机种子 42，避免继承已训练关节投影与随机 endpose 投影的不对称。初始化不来自本次 20 步烟测。

**训练方案与归因边界**

每组先 500 步热身，再 4500 步联合训练，有效 batch 32。动作/空间分支 LR 1e-4，特征模块 LR 1e-5，阶段间重建 optimizer/scheduler；通用未来预测 loss 保持关闭。热身冻结视觉骨干、WM、pooler、任务/embodiment embedding 和其他机器人的 action heads，训练 Aloha 动作头和空间分支。

CPU 梯度检查发现：当前局部裁剪位置经过 detach，关闭空间监督时重置后的 `spatial_focus.query` 没有有效动作梯度。因此 **两个条件都恢复原有空间定位监督**，heatmap 权重 0.002、coordinate 权重 0.02，教师 crop 调度仍为 0。空间目标仅用于训练监督，部署不读取目标位置。

两个新条件之间的差别为 state 内容及其对应统计/语义标识；可以比较反馈方式。与此前仅 L1、未重置输入层的 GAWM 分数相比，还有共同训练方案变化，不能作单因素归因。没有改变输出动作、动作标签、相机、动作归一化或执行周期。

**已完成检查**

- 485 条选中轨迹完成 raw command/Parquet/state/帧索引对齐，以及 endpose 数值、四元数和夹爪检查。来源、数值哈希与输出 Parquet 哈希见 `data_manifest.json`。
- 两组真实共享 loader 都返回 16 维 state，indexed 和 mixture 路径均检查；抽取 10 条训练轨迹的首、中、尾共 30 个样本，图像、动作标签及动作/未来 mask 与旧 loader 完全一致。
- 10 条验证轨迹的 state 对齐检查通过；全部 1024 组随机动作反归一化与旧策略完全一致。
- CPU 上 577 个模型张量严格加载、直接推理与部署 wrapper 动作完全一致。loader 会转 FP16：command state 差值为 0，endpose 归一化 state 最大差值约 1.19e-7；检查容差明确记录在 JSON 中。
- CPU 两组均对 state 扰动敏感；state projection 和空间 query 均有非零、有限梯度。endpose 组还通过 GPU 实际前向、反向和直接/部署一致性检查。
- 14 项 pytest 通过，包括反馈字段、四元数顺序、动作契约、环境适配器、既有 RoboTwin 动作顺序及执行 horizon。3 步优化只验证工程通路，不是收敛或成功率证据。

保留了初次审计失败日志：一轮缺少共享目录写权限；一轮暴露共享 loader 首次按训练 split 建缓存后会漏掉验证 steps；随后一轮揭示空间 query 零梯度。准备脚本现在先根据完整 485 条 metadata 生成未过滤的步索引，两种 split 再各自过滤，不修改其他正在运行的共享 loader 代码。旧局部缓存保留为备份。

GPU 最初在沙箱中显示不可用，宿主机设备查询确认是沙箱设备访问限制。没有停止现有进程。本次指定 GPU 4，只在空闲显存至少 24 GiB 时允许与已有任务共用。

**运行与结果**

运行目录与状态集中于：

`playground/Checkpoints/gawm_endpose_20260911/`

大 checkpoint 位于 `/data/gaoxiang/ckpts/gawm_endpose_20260911/`。2026-09-11 12:05 首次启动后台 coordinator。首组 20 步训练和 GPU 部署审计完成后，仿真环境因 dataloader 包导入 accelerate 而启动失败；已将纯 NumPy 反馈函数移到顶层公共模块，通过独立 RoboTwin Python 的入口导入检查，归档旧日志至 `failures/sim_import_r1`。12:11 以 PID 77196 重启 coordinator，使用 `--reuse-trained-smoke` 复核同一份已完成的 command 烟测权重，没有额外训练 20 步（以 `launch.json` 的出生时间及实时状态为准）。先依次运行两组 20 步完整训练烟测、严格部署审计及同一个开发场景；两组都通过后，自动执行两组正式训练与各自 20＋100 场景评测。烟测不要求随机初始策略成功，只要求工程链路、合法动作、原始评分完整执行和同场景一致。

状态文件为 `coordinator_status.json`、`smoke_status.json`、`campaign_status.json`；逐阶段日志、严格部署检查和逐场 result.json 同目录保存。协调器冻结相关源码和配置哈希，失败停止并保留错误，不重复创建已有 run。开发集和测试集分别汇总，不合并成功率。当前尚无正式成功率结论。

以下命令用于独立复核或新环境运行；已有活动 coordinator 时不要重复启动：

```bash
NO_ALBUMENTATIONS_UPDATE=1 /data/gaoxiang/Code/.venvs/starVLA/bin/python \
  -m unittest tests.test_robotwin_feedback tests.test_robotwin_interface tests.test_robotwin_execute_horizon -v

NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 /data/gaoxiang/Code/.venvs/starVLA/bin/python \
  -m examples.RobotwinEndPose.audit --variant endpose --device cpu --optimize

NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 /data/gaoxiang/Code/.venvs/starVLA/bin/python \
  -m examples.RobotwinEndPose.campaign --gpu 4 --detach
```

GPU、共享 dataset 索引和 checkpoint 目录需要相应宿主机访问权限。当前环境通过受审查的执行权限运行；普通沙箱 CPU 检查不代表宿主机 GPU 状态。

按上级 AGENTS.md 尝试向 `36.212.196.90:1227` 同步派生数据。远端父目录已创建，但约 302 MiB 数据传输被自动审批拒绝，理由是目标主机未经核验且缺少用户对该 payload/destination 的直接授权；未绕过拒绝、未传输数据。本机训练不受影响，详见 `data_sync_status.json`。
