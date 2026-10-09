# LIBERO 数据与多机训练检查（2026-09-21）

2026-09-26 更新：官方源示范已完成逐轨迹对齐，新增 862 条有效轨迹 / 155,038 帧；
可用训练数据现为 6,585 条 / 1,027,125 帧。LIBERO-90 的 90 个场景均有覆盖。
使用新增数据需要选择独立的新 mixture，详见[补齐报告](../docs/libero_data_completion_20260926.md)。

2026-09-22 更新：八个数据目录已全部同步到五台节点的
`playground/Datasets/LIBERO_FULL/`，全量 GAWM 12 epoch 训练已启动。
正式启动时又逐帧扫描全部视频，发现 Goal 第 82 条轨迹的腕部视频损坏；
训练已排除这条 129 帧轨迹，原始文件保留。详见
[GAWM_FULL_TRAINING.md](GAWM_FULL_TRAINING.md)；下文保留此前抽样检查记录。

数据路径以本机配置为准，公开示例使用 `playground/Datasets/LIBERO`。
数据可被当前项目和 `.venv` 直接读取，
不需要格式转换。

## 数据清单

| 数据集 | 轨迹 | 帧 | 唯一语言描述 |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 432 | 52,970 | 10 |
| LIBERO-Object | 454 | 66,984 | 10 |
| LIBERO-Goal | 428 | 52,042 | 10 |
| LIBERO-10 | 379 | 101,469 | 10 |
| LIBERO-90 目录 | 3,921 | 569,249 | **73** |
| LIBERO-10 replay 补充 | 21 | 5,705 | 7 |
| LIBERO-10 teacher task8 补充 | 15 | 5,681 | 1 |
| LIBERO-10 teacher remaining 补充 | 74 | 18,116 | 10 |

四个标准套件共 **1,693 条轨迹 / 273,465 帧 / 约 1.76 GiB**。
全部八个目录共 **5,724 条轨迹 / 872,216 帧 / 约 5.04 GiB**。
这里的大小按文件字节统计，和 `du` 的磁盘占用略有差别。

所有目录均为 LeRobot v2.1，包含 `modality.json`、格式版本 2 的
`stats_gr00t.json`（`mode=abs`）和 `steps_data_index.pkl`。
标准套件和 LIBERO-90 的视频为 AV1，补充轨迹为 H.264。

2026-09-25 更正：`libero_90_no_noops_lerobot` 的 73 表示唯一语言描述数，
不能据此推断只有 73 个场景任务或缺失 17 个任务。官方 90 个场景对应 74 个唯一语言描述。
确切场景覆盖待原始示范匹配确认；全量解码和补齐审计见
[2026-09-25 补齐报告](../docs/libero_data_completion_20260925.md)。

## 实测范围

- 读取全部 5,724 个 Parquet：校验行数、7 维 action、8 维 state、数值有限性、
  episode/task 索引以及连续的帧索引。
- 打开全部 11,448 个视频的容器头，核对文件存在性、分辨率和视频声明的帧数。
- 各目录实际读取第一个、中间和最后一个训练样本：action 为 `(8, 7)`，
  两路图像各为 `224 × 224`，语言指令非空。
- 用真实 `libero_all` 混合数据集、两个 DataLoader worker 读取三个 batch。
- 用 `libero_franka_wm` 验证当前/未来图像和 `(3, 8)` state。

以上检查全部通过。视频仅抽样解码，没有逐帧完整解码所有视频。
加载测试把元数据复制到项目 `.cache/` 后进行，原始数据和缓存未修改。

详细结果：`.cache/libero-inventory-report.json`、
`.cache/libero-loader-report.json`；日志在对应的 `*-check.log` 中。

## 直接使用的配置

在所选训练 YAML 中设置，或通过训练入口的同名命令行参数覆盖：

```yaml
datasets:
  vla_data:
    data_root_dir: playground/Datasets/LIBERO
    data_mix: libero_all
    video_backend: torchvision_av
```

`libero_all` 只包含 Spatial / Object / Goal / 10 四个标准套件。
world-model 训练已有 `libero_all_wm`；三组 replay/teacher 补充只有显式选择
`libero_all_wm_l10_augmented` 等对应 mixture 才会加入。
LIBERO-90 也需要显式选择对应 mixture。

当前 AV1 已通过 `torchvision_av` 实测，不要根据普通 H.264 的解码结果假定
默认 `decord` 一定支持这批 AV1 视频。

## 多机训练条件（部署后更新）

五台节点 `controller / worker-1 / worker-2 / worker-3 / worker-4` 均已部署到
`.`，包含相同的项目基础环境和 `.venv`。
每台的 `.cache/multinode/data/libero_goal_no_noops_1.0.0_lerobot` 已有
LIBERO-Goal 的完整副本（428 条轨迹 / 52,042 帧），通过内网同步。
其他套件仍在本机原始数据目录，尚未全部复制到远端。

节点的 `/data` 是各自本地磁盘，不是共享存储。已验证 `eth0` 上的 TCP/NCCL
通信，并完成两机四卡、五机 35 卡的真实 Goal 数据训练和完整断点恢复。
GAWM 的具体配置、修复和实测结果见 `MULTINODE_TRAINING.md`。

复用时注意：

- 设置 `NCCL_SOCKET_IFNAME=eth0`、`GLOO_SOCKET_IFNAME=eth0`、
  `NCCL_IB_DISABLE=1`。原示例固定的 `bond0` / `mlx5_*` 不适用于这些节点。
- 每台相同卡数，分别指定 `machine_rank`；`num_processes` 是全局 GPU 总数。
- 数据加载器只由全局 rank 0 重建部分缓存，本地副本须提前同步与训练配置匹配的
  `meta/` 统计和索引；barrier 不会复制缓存。
- checkpoint 配置启用 `use_node_local_storage`，保存后收集所有 rank 的
  optimizer / RNG 分片，再将完整目录同步到各节点进行恢复。
- 机器合计 40 卡，但 GPU 0 有已有作业。本次统一使用每台 GPU 1–7，共 35 卡。
  后续启动前应重新检查占用。
- 全局 batch = 每卡 batch × GPU 总数 × 梯度累积；扩大节点数时一起调整。

GAWM 不依赖 Qwen 权重。本次按用户选择随机初始化完整 GAWM 架构短跑验证；
不代表已完成预训练或策略收敛。其他 VLA 示例的 Qwen3.5 / Transformers 5.x
依赖仍需另外适配，不能直接运行原 `run_libero_train.sh`。
