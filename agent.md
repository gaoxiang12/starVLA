训练节点使用 `controller`、`worker-1` 至 `worker-4` 逻辑名称。真实地址和 SSH 用户保存在本机 `.local/settings.json`，不得提交到 Git。配置方式见 [本地配置说明](design/LOCAL_CONFIGURATION.md)。

加上本机，共40张卡可用

后续启动训练时，优先使用各机 GPU 0–7，共 40 张卡，不再统一排除 GPU 0。
启动前检查各卡占用；已有其他作业的 GPU 不直接抢占或终止其进程，按实际可用资源分配。
此偏好用于后续启动，不要求中断当前正在运行的 35 卡训练。

本机已经配置代理，你可以用代理流量连接github或者外网资源，但要满足以下原则：
- hugging face数据走hf-mirror,需要登录的告诉我
- DINOv3 预训练权重默认从 ModelScope 下载，缓存到项目的 playground/Pretrained；已有本地路径继续支持。
- uv, pip, apt 等走国内镜像,不要走代理
- 大数据集的下载,必须要走代理的，先经过我确认

## GAWM 下一轮改版（用户确认，2026-09-27）

下一轮按 [GAWM_FUTURE_PREDICTION_OBJECTIVE.md](design/GAWM_FUTURE_PREDICTION_OBJECTIVE.md) 中“下一轮改版要求”执行：保留可训练 adapter 和紧凑 latent；通过特征解码器把主要未来预测监督移到冻结 DINO 的未来 patch 特征；单独加入并验证 latent 时间平滑约束，保留尺度约束并防止时间塌缩。不能把在线 adapter 的 detach 输出当成固定教师。当前归一化训练与下一轮改版分开，本次记录不要求中断或重启当前训练。未定实现参数和验证标准见该文档。

## 文档位置

新增设计文档、训练方案、数据检查和评测记录统一放在 `design/`，并更新 `design/README.md` 索引，不再堆放到仓库根目录。

公开代码、配置和文档中不得写入真实节点地址、登录用户名、个人绝对路径或凭据。机器专用配置和原始运行记录保留在被忽略的 `.local/`、`.cache/` 或 `playground/` 中。
