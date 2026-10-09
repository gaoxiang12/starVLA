# RoboTwin GAWM 三视角重新训练准备

> 用户随后改为两路相机，现已启动双视角训练，见 `GAWM_ROBOTWIN_2VIEW_TRAINING.md`。本三视角草案未启动，不再是当前待执行方案。

2026-09-25 17:52（北京时间）：**训练尚未启动，等待真实三视角数据来源。**

用户要求将图像输入改为三个并重新训练，当前按头部、左腕、右腕三个物理相机准备。已通过异步问题询问数据所在机器与路径。

## 已完成

- `starVLA/dataloader/robotwin_official_hdf5.py` 支持显式 cameras 顺序，同步读取每个时刻的三路相机；当前帧与未来+16/+32帧分别返回三视角。
- 新训练默认使用正确物理 RGB。官方直接将 RGB 交给 OpenCV JPEG 编码，因此 OpenCV 解码后不再额外交换 R/B。旧颜色约定可以显式 image_channel_order=bgr；旧 run 的冻结源码和权重未修改。
- 初始化检查任务首条轨迹是否含所有指定相机；每次读取检查相机帧数。缺少相机时报错，不复制头部图像、不以补零假装三路有效视角。
- DataLoader 检查模型 num_views 与相机列表长度一致。
- 配置草案：`examples/Robotwin/train_files/starvla_gawm_robotwin_3view_c_12plus4.yaml`。需要据真实数据设置 data_root_dir、expected_frames 和审计/统计文件；草案不是已启动的实验。
- 19项测试通过，覆盖三相机顺序、同一记录偏移、缺失相机、长度不匹配、RGB颜色、旧权重颜色兼容及C配方。
- 使用现有真实官方数据做缺失相机检查，明确报错缺少 left_camera、right_camera，确认不会误启动。

## 数据缺口

五台机器（本机、worker-1/worker-2/worker-3/worker-4）均有 `playground/Datasets/LiLaWAM_RoboTwin_Official`，该数据只有 front_camera、head_camera；其图像分别为240×320×3，非三相机拼接图。

五台机器上历史三视角路径 `playground/Datasets/RoboTwin`、`playground/Datasets/RoboTwinGenerated` 均不存在。旧代码里有三视角 LeRobot 配置，但配置所指数据不在这五台机器上；那套配置还使用14D关节状态，不能无核查地当作本次16D末端状态数据。

检查报告：`.cache/robotwin_3view_training/readiness.json`。

## 训练草案

- 模型：GAWM，冻结已有 DINOv3 ViT-L/16，其余模块重新随机初始化；不续训旧单视角 checkpoint。
- 三视角：head_camera、left_camera、right_camera；各320×240，正确RGB、ImageNet归一化；各64视觉tokens，共192。
- 若拿到兼容官方HDF5数据，保留16D末端状态、14D绝对关节/连续夹爪动作，预测32步；动作和状态统计必须对应实际数据。
- C配方：12+4 epochs，全局batch128，LR2e-4，第二阶段重置Adam并乘0.2；具体步数按真实帧数计算，不能沿用旧765120步。
- 检查时36张GPU空闲（worker-1的0–3有其他作业）。为保持batch128，候选方案为五机32卡，每卡4；最终启动前再次检查占用和短步显存/吞吐。若资源或数据改变，重新确定分配。
- 数据就绪后：核验三相机/颜色/状态动作契约、生成审计和统计、内网同步、实际batch前后向与多机短训验证，随后以新run后台训练。
