# 新到 RoboTwinGenerated Clean 数据检查

2026-09-26。数据：`playground/Datasets/RoboTwinGenerated/Clean/` → `playground/Datasets/RoboTwinGenerated/Clean/`。

## 结论

**数据的文件、数值字段和图像抽查通过，可以用于经过适配的训练，但不能直接加入当前双视角、16维末端状态训练。**

现有数据本身已有14维状态字段；缺少的是与当前模型语义匹配的真实16维末端状态。没有原始记录时，不能从这些Parquet无损恢复该状态。

## 已核验范围

- 50任务，各500条Clean轨迹，共25,000条、5,671,973帧；文件数量和逐轨迹帧数与meta一致。
- 全部25,000份Parquet逐帧检查状态/动作维数、有限值、连续frame_index、episode_index和时间戳；没有发现异常。读取期间文件大小/mtime保持稳定。
- 所有轨迹均有3路内嵌JPEG列：cam_high、cam_left_wrist、cam_right_wrist，320×240；没有front_camera。
- 每任务3条轨迹×首中末3帧×3相机，共1350张图像解码通过；未声称全量解码所有图像。
- 全部25,000条轨迹中 `observation.state == action` 逐元素成立，均为14维。
- 夹爪位于索引6/13，范围[0,1]，881,196帧至少一个夹爪处于连续中间值；不能当成二值夹爪。
- 未发现任务内部完全重复的动作序列。未做跨任务或与旧数据的全量去重。

## 与当前训练的差异

| 项目 | 当前双视角训练 | 新到数据 |
|---|---|---|
| 格式 | 官方HDF5 | LeRobot v2.1 Parquet内嵌JPEG |
| 相机 | head_camera + front_camera | 头部 + 左腕 + 右腕，无前方相机 |
| 状态 | 左右末端位姿各7维＋夹爪各1维，共16维 | 14维关节/夹爪控制目标，数值与action一致 |
| 动作 | 14维双臂关节目标与连续夹爪 | 同维数/布局，但需保持记录帧对齐语义 |
| 颜色 | 物理RGB | 头部样本标准PIL RGB解码显示红蓝交换，接入前需校正 |

不能仅将cam_left_wrist改名为front_camera，它们的物理位置不同。即使补齐状态，也仍要决定新的视角方案或使用显式缺失视角mask；当前读取器不接受缺失有效视角。

颜色核查：对blocks_ranking_rgb抽查7条示范末帧，6条可辨认三个色块，均显示左蓝、中绿、右红；另外1条有遮挡，未纳入顺序判断。官方任务要求物理红绿蓝从左到右排列，说明该标准JPEG解码数组需要交换R/B。腕部图片的颜色需结合原始编码来源进一步验证。

## 状态能否补齐

1. **有原始HDF5/PKL的endpose：可准确回填。** 必须验证原始轨迹映射、帧数/索引、动作与图像对应关系，然后回填左右7D位姿与两路夹爪到新的派生字段/数据集，不覆盖现有14D字段。
2. **有逐帧实测关节位置：可经验证后的正向运动学重建。** 需要匹配URDF版本、机器人基座位姿、末端偏移、四元数顺序和夹爪观测定义。控制目标不是实测关节位置。
3. **只有当前Parquet：无法精确恢复实测末端状态。** 只能计算关节目标对应的理想位姿，必须明确标为近似的commanded pose；不能当作实测endpose混入原统计。
4. **另一条训练路线**：使用现有14D关节目标状态，配置三视角模型/独立状态头，重新计算统计，并对齐训练与在线评测的状态和动作时间语义。无需伪造16D实测状态，但不是当前双视角训练的直接续接。

数值验证：使用本地Aloha URDF、已知世界基座变换，对已有官方数据的3个任务、左右臂做FK(command)与真实endpose对比；初始静止位置吻合，运动段误差在毫米到厘米级，样本最大约19.4mm。这是对已有官方数据的诊断，不是新数据误差上界。

源码依据：`examples/Robotwin/data_preparation.py` 的转换函数使用 `state = action.copy()`；`thirdparty/RoboTwin/envs/robot/robot.py` 的关节状态记录函数读取drive_target，而endpose来自仿真实际关节位姿。这说明两者不能等价替换。新数据本身的全量数值比较也证实state/action相同；未据此声称它一定由当前这份转换脚本版本生成。

## 原始数据查找

已核对五台配置节点的已知 `RoboTwinGenerated_raw`、`RoboTwin_raw`、旧Code/RoboTwin/data和项目raw目录，未发现对应轨迹。本机thirdparty/RoboTwin/data只有工具脚本。新数据旧 `.deep-validation-v1.json` 中写有校验主机 `ibenx`，可优先在原生成机器查找HDF5/PKL；该标记不是本次审计证明，也不证明文件当前仍在该机。

已询问用户原始数据机器和路径。没有改写源数据、没有生成伪实测状态，也没有修改当前训练。

## 报告与工具

- 全量数值/图像抽查：`.cache/robotwin_generated_check/full_audit.json`
- 兼容性结论：`.cache/robotwin_generated_check/compatibility.json`
- FK误差诊断：`.cache/robotwin_generated_check/command_fk_vs_measured.json`
- 颜色样本：`.cache/robotwin_generated_check/color_check.json`
- 原始目录检查：`.cache/robotwin_generated_check/raw_locations.json`
- 可重复只读工具：`scripts/audit_robotwin_generated_data.py`

## 2026-09-26 用户确认后的使用决定

原始 HDF5/PKL 未传来，用户确认这批数据如果不能兼容就不使用。本次不纳入 `playground/Datasets/RoboTwinGenerated/Clean`，不以控制目标推算并冒充实测状态。

用户随后确认先使用现有官方数据的头部＋前方两路。官方数据复核、双相机 LiLa＋VTT 配置和 VTT 准备队列见 `ROBOTWIN_HEAD_FRONT_DATA_PREPARATION.md`。该方案不包含左右腕部图像。
