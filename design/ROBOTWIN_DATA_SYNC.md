# RoboTwin 四节点数据同步

**已完成：2026-09-23 09:31:12，四台节点全部同步及校验通过。**

- 数据源和目标路径均为 `playground/Datasets/LiLaWAM_RoboTwin_Official/`。
- 目标节点：`worker-1`、`worker-2`、`worker-3`、`worker-4`。
- 每节点同步 27,179 个文件，约 279.72 GB；包含全部 HDF5、原始审计和六个 training_metadata 文件。
- rsync 使用内网 SSH、断点重传，已完成文件可复用。
- 三台节点的父目录由 root 管理，已仅创建数据接收目录并赋予训练用户所有权。
- 本次 supervisor PID：`1952217`；任务已完成。

已完成运行目录：`.cache/robotwin_sync/20260923_092340/`。
最近运行路径也记录在 `.cache/robotwin_sync/latest_run.txt`。

```bash
cat "$(cat .cache/robotwin_sync/latest_run.txt)/status.json"
```

每台节点依次执行：

1. 同步全部数据及元数据。
2. 使用 rsync dry-run + xxh128 校验所有文件内容，要求零差异。
3. 核对文件清单、大小、HDF5 数量，并检查全部元数据及 20 个完整 HDF5 样本的 SHA256。
4. 在主节点保存对应 `<节点>.verification.json`。

状态 `complete` 表示四台节点传输和验证都完成，并已确认同步期间源文件大小/mtime 未改变。
`failed` 或 `stopped` 不能作为完成标记。

日志：运行目录的 `<节点>.transfer.log`、`<节点>.checksum.log`，以及
`.cache/robotwin_sync/20260923_092340_supervisor.log`。

## 最终结果

四台节点 `worker-1 / worker-2 / worker-3 / worker-4` 均通过以下检查：

- 每台 27,179 个文件，27,071 条 HDF5 轨迹，279,723,464,127 字节。
- rsync xxh128 全量文件内容比较：零差异。
- 清单与大小：无缺失、无多余、无大小不符。
- 全部元数据及 20 个 HDF5 样本，共 128 个文件 SHA256：全部匹配。
- 同步期间源文件大小和修改时间没有变化。

总报告：`.cache/robotwin_sync/20260923_092340/summary.json`。
