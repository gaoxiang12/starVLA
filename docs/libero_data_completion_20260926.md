# LIBERO 现有示范补齐结果（2026-09-26）

已恢复并验收 **862 条轨迹 / 155,038 帧**。可用训练数据从 **5,723 条 / 872,087 帧** 增至 **6,585 条 / 1,027,125 帧**，可用帧数增加 **17.78%**。本次只补齐数据，没有启动新训练或模型评测。

## 各套件结果

| 套件 | 恢复前可用原始示范 | 本次恢复 | 恢复后原始示范 | 未通过终态判定 |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 432 | 66 | 498 / 500 | 2 |
| Object | 454 | 46 | 500 / 500 | 0 |
| Goal | 427 | 73 | 500 / 500 | 0 |
| LIBERO-10 | 400（379 基础 + 21 replay） | 100 | 500 / 500 | 0 |
| LIBERO-90 | 3,921 | 577 | 4,498 / 4,500 | 2 |
| 合计 | 5,634 | 862 | 6,496 / 6,500 | 4 |

另外保留已有 89 条 LIBERO-10 teacher 轨迹，因此完整训练混合共有 6,585 条可用轨迹。所有 130 个官方场景任务均有覆盖；LIBERO-90 从 89/90 场景补到 90/90，新增了 `LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket`。语言标签数量不等于场景任务数量。

Goal 的恢复包含原先 episode_000082 损坏轨迹对应的官方 `push_the_plate_to_the_front_of_the_stove/demo_23`。本次重建了该示范的两路观测和状态，作为独立完整轨迹保存。旧损坏文件仍保留并继续被 `libero_video_exclusions.json` 排除，不能只取消旧排除项。

## 恢复方法与质量检查

官方源为 `yifengzhu-hf/LIBERO-datasets`，固定版本 `f13aa24a3da8c43c7225569f28c562979fa0e35a`。130 个原始文件共 100,442,942,572 字节，按用户要求直连 hf-mirror 下载，全部通过 SHA256 校验。

对每条原始示范按现有 no-op 规则过滤动作，再以完整动作序列签名与现有八个数据集匹配。仅处理缺失或损坏的 866 条记录。对齐验证发现官方 HDF5 的已存图像/机器人观测在动作执行之后，而 `states[t]` 是对应 `actions[t]` 的执行前模拟器状态，因此没有直接拼接原始图像和同索引动作。使用源示范的模型 XML 和执行前状态，重新渲染 256×256 双视角并同步重建 8 维 proprio 状态，夹爪动作转换为现有训练约定。

本次采用记录状态逐帧重建，不是从起点重放动作，也不表示每条轨迹在当前模拟器中重新执行动作必定成功。这样能保持官方记录轨迹，避免动作重放的动力学累积偏差。每条新增轨迹都验证了源终态的任务成功条件；源终态尚未成功时，还检查执行最后一个原始动作后的成功条件。未通过的 4 条排除。

21 条示范因当前 BDDL 将旧 `salad_dressing` 替换成 `new_salad_dressing` 而加载失败。针对这些源 XML，独立生成兼容 BDDL，恢复源模型的旧对象类型和名字；保持原始模型、状态、动作与任务目标，重试后均通过。兼容记录及首次失败记录保留在 provenance 中，公共任务定义未修改。

验收包括：

- 五个套件各一条已有示范作为控制样本，验证源状态时序、成功判定、图像输出。
- 862 条新增轨迹的 Parquet 行数、状态/动作维度、有限性、帧索引和元数据一致性。
- 1,724 个新增视频在远端输出时和本机组装后均逐帧完整解码，帧数与 256×256 尺寸一致。
- 完整动作序列签名与官方源一致；新增数据与全部现有可用数据之间无重复动作序列。
- 五个新增数据集首/中/尾共 15 个样本通过真实 `unified_libero_wm` 加载器：动作 `(8, 7)`、当前/未来状态 `(3, 8)`、三视角打包和两组未来图像通过检查。
- 完整混合数据长度为 1,027,125 帧，双 worker DataLoader 的三个 batch 通过检查。新增数据统计与步骤索引已生成。

恢复仅使用官方训练示范状态，没有读取固定评测初始状态或评测 rollout。这里的终态成功检查是数据质量检查，不是新模型成功率报告。

补充数据已同步到本机和 `worker-1`、`worker-2`、`worker-3`、`worker-4` 共五台节点。每台远端的 2,621 个补充文件均通过 SHA256 对比；新 mixture 注册与 data-only 配置也已同步。

## 输出与使用

数据位于 `playground/Datasets/LIBERO_FULL/` 下五个独立目录：

- `libero_spatial_state_recovered_20260926_lerobot`
- `libero_object_state_recovered_20260926_lerobot`
- `libero_goal_state_recovered_20260926_lerobot`
- `libero_10_state_recovered_20260926_lerobot`
- `libero_90_state_recovered_20260926_lerobot`

每个目录的 `recovery_manifest.jsonl` 保留原始文件、demo ID、原始帧索引、动作签名、源版本、恢复方法及兼容处理记录。原有八个数据集和旧 mixture 均保留。

新训练数据配置为 [libero_completed_data_20260926.yaml](../examples/LIBERO/train_files/libero_completed_data_20260926.yaml)，这是需要合并到训练配方中的 data-only override，不是独立训练配方。它选择 `unified_libero_completed_20260926_wm`，明确使用 `sampling_mode: frame_epoch`、`load_all_data_for_training: true`、`expected_frames: 1027125`，并保留旧 Goal 损坏轨迹的排除规则。新混合不会自动应用到旧训练任务。若改用随机加权采样，应另行明确各数据集权重。

## 保留排除项

| 套件 | 源文件（去掉 `_demo.hdf5`） | demo |
| --- | --- | --- |
| LIBERO-90 | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate | 42 |
| LIBERO-90 | STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf | 43 |
| Spatial | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 17 |
| Spatial | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 45 |

这四条在本次源状态恢复与最终动作检查下均未通过当前环境的终态成功判定，没有加入训练集。

## 可复查产物

- [最终源覆盖](../.cache/libero_completion_20260925/final_coverage.json)
- [恢复数量与排除清单](../.cache/libero_completion_20260925/recovery_assembly.json)
- [完整验收结果](../.cache/libero_completion_20260925/completion_validation.json)
- [源状态时序验证](../.cache/libero_completion_20260925/alignment_probe_source_xml.json)
- [逐文件同步校验](../.cache/libero_completion_20260925/sync_verification.json)
- [状态恢复工具](../examples/LIBERO/data_tools/recover_official_state_demos.py)
- [数据集组装工具](../examples/LIBERO/data_tools/assemble_state_recovered_demos.py)
