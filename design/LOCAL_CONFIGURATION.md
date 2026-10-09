# 本地集群配置

真实节点地址、SSH 用户和个人路径存放在被 Git 忽略的 `.local/` 中。仓库里的 `*.example.invalid` 是占位地址，运行前需要配置；也可以用脚本的 `--hosts`、`--host` 等参数显式指定节点。

在仓库根目录执行：

```bash
mkdir -p .local
cp scripts/local_settings.example.json .local/settings.json
```

将 `hosts` 中的 `controller`、`worker-1` 至 `worker-4` 改为本机可访问的节点 IP 或主机名。`ssh_user` 留空时使用当前用户或 SSH 配置。可通过 `STARVLA_LOCAL_SETTINGS` 指定其他本地 JSON 文件。此文件由本机启动器读取，无须发布到远端仓库。

监控面板优先读取 `.local/cluster_nodes.json`，否则使用 `scripts/cluster_nodes.json` 的示例。可复制示例并设置真实节点、SSH 用户和 `known_hosts` 路径，也可通过 `--config` 指定文件。

训练 YAML 中的项目路径相对于仓库根目录。数据放到 `playground/Datasets/` 下的对应目录，或在本机建立到外部数据的符号链接。机器专用训练配置可保存在 `.local/`，通过启动器的 `--config` 参数选择。VTT 数据准备队列支持用 `STARVLA_ROBOTWIN_DATA` 覆盖数据路径。

脱敏前的本机配置和文档备份保存在 `.local/pre-redaction-4fd1cc9/`，供本机复现历史任务；此目录不纳入提交。正在运行任务的源码快照、checkpoint 和队列状态未作改动。
