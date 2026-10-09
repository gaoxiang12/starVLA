# RoboTwin 相机与颜色通道核查（2026-09-25）

## 结论

本轮 GAWM 训练与评测实际只使用 head_camera 一路相机，输入尺寸 320×240，张量为 3×240×320。训练中的 0/16/32 是同一相机三个时间点，后两帧用于未来监督，不是三路相机。抽查全部 50 个任务各自第一条 HDF5，observation 下均为 front_camera、head_camera；模型仅读取 head_camera。仿真虽然配置了头部和腕部相机，但策略只取头部图像。

**发现训练与评测红蓝通道不一致。** 官方数据写入函数直接将仿真 RGB 数组交给 cv2.imencode，未预先转换 BGR。因此这批 HDF5 用 cv2.imdecode 解码得到的数组数值顺序已对应物理 RGB；不能再按普通 JPEG 的假设做 BGR2RGB。当前训练快照多做了该转换，导致模型看到的物理颜色顺序为 BGR。在线评测直接使用仿真 RGB，并未做相同交换。

## 证据

- 官方编码：`thirdparty/RoboTwin/envs/utils/pkl2hdf5.py:15`，直接 `cv2.imencode(".jpg", imgs[i])`。
- 仿真取色：`thirdparty/RoboTwin/envs/camera/camera.py:323`，Color RGBA 的前三个通道。
- 本次训练快照：`playground/Checkpoints/gawm_robotwin_c_32gpu_resume_20260923_222947/source_snapshot/starVLA/dataloader/robotwin_official_hdf5.py:106`，额外 `cv2.COLOR_BGR2RGB`。
- 评测取图：`examples/Robotwin/eval_files/gawm_hdf5_interface.py:29`，直接 head_camera.rgb；服务端不交换通道。
- 已抽取 blocks_ranking_rgb 的 10 条 Clean 和 10 条 Randomized 示范末帧。用相机内外参将高饱和颜色像素反投影到积木高度，20/20 条的直接解码通道0/1/2依次位于 world x 负/中/正，与官方红、绿、蓝目标位置一致。训练转换后则为蓝、绿、红。
- 复现纯红 RGB [255,0,0] 经官方编码及冻结训练读取，反归一化后约为 [0,0,254]，确认红蓝交换。
- 原有测试用标准 BGR 数组编码 JPEG，没有复现官方直接编码 RGB 的约定，因而未能检出这一问题。
- 数值与抽查文件路径：`.cache/robotwin_evaluation/color_audit/audit.json`；示例图在同目录 `episode0_frame475_direct.png` 与 `episode0_frame475_train.png`。

## 对结果的影响与修复方向

77.2%（386/500）是实际完成的观测结果，计数与权重哈希核查成立，但属于训练/评测通道不一致条件下的结果，需要对齐后重新评测。不能据此直接判定 blocks_ranking_rgb 的 0/10 是模型学不会颜色，也不能在重跑前推断修复后的成功率。

对已经训练完成的 checkpoint，应在在线图像归一化前交换 R/B，以匹配既有训练分布，再重跑相同种子和任务预算。面向后续重新训练，应移除这批官方 HDF5 读取中的额外 BGR2RGB，训练与在线评测均统一为物理 RGB；这两种方案必须用明确协议区分。

本次核查没有修改模型、冻结训练源码或评测适配器，也没有启动新评测。已给旧结果加上颜色错位说明。

## 后续执行

用户授权后，2026-09-25 01:19 已启动交换 R/B 的同权重重评（50×10，seed0）。训练代码和权重保持原样。新适配器通过18张真实图像逐像素对齐验证，详情见 [GAWM_ROBOTWIN_EVALUATION.md](../design/GAWM_ROBOTWIN_EVALUATION.md)。
