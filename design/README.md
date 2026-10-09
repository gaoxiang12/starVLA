# 设计与实验文档

本目录集中存放设计方案、环境说明、数据检查、训练及评测记录。各文档中的运行状态以记录日期为准；命令和行内代码中的项目路径默认相对于仓库根目录，运行命令前请先进入仓库根目录。

文档中的 `controller`、`worker-1` 至 `worker-4` 为脱敏节点别名。运行示例命令前，请按本机配置替换这些别名；历史原始记录保留在本地备份中。

## 设计与实验

- [实验计划与结果](experiment.md)
- [GAWM 未来预测目标与改版要求](GAWM_FUTURE_PREDICTION_OBJECTIVE.md)
- [GAWM-L 视觉前端与 VTT](GAWM_L_VISION_VTT.md)

## 环境与多机训练

- [LIBERO / RoboTwin 训练与测试完整流程](../LIBERO_ROBOTWIN_TRAIN_EVAL.md)
- [本地集群配置与脱敏约定](LOCAL_CONFIGURATION.md)
- [训练环境](ENVIRONMENT.md)
- [GAWM 多机训练验证](MULTINODE_TRAINING.md)

## 数据检查与准备

- [LIBERO 数据检查](LIBERO_DATA_CHECK.md)
- [RoboCasa365 数据检查](ROBOCASA365_DATA_CHECK.md)
- [RoboTwin 数据检查](ROBOTWIN_DATA_CHECK.md)
- [RoboTwin 数据同步](ROBOTWIN_DATA_SYNC.md)
- [RoboTwin 生成数据检查](ROBOTWIN_GENERATED_DATA_CHECK.md)
- [RoboTwin 头部与前方数据准备](ROBOTWIN_HEAD_FRONT_DATA_PREPARATION.md)

## LIBERO 训练与评测

- [全量 GAWM 训练](GAWM_FULL_TRAINING.md)
- [GAWM C 配方训练](GAWM_LIBERO_C_TRAINING.md)
- [GAWM 评测](GAWM_LIBERO_EVALUATION.md)

## RoboTwin 训练与评测

- [训练方案](GAWM_ROBOTWIN_TRAINING_PLAN.md)
- [单视角训练记录](GAWM_ROBOTWIN_TRAINING.md)
- [双视角训练](GAWM_ROBOTWIN_2VIEW_TRAINING.md)
- [三视角训练草案](GAWM_ROBOTWIN_3VIEW_TRAINING.md)
- [GAWM-L 与 VTT 训练](GAWM_L_ROBOTWIN_VTT_TRAINING.md)
- [GAWM-L 与 VTT 固定归一化训练](GAWM_L_ROBOTWIN_VTT_NORM_TRAINING.md)
- [Clean 评测汇总](GAWM_ROBOTWIN_EVALUATION.md)
- [GAWM-L 与 VTT 固定归一化评测](GAWM_L_ROBOTWIN_VTT_NORM_EVALUATION.md)

## 诊断与分析

- [GAWM-L 与 VTT latent loss 诊断](GAWM_ROBOTWIN_LATENT_LOSS_DIAGNOSIS.md)
- [归一化训练 latent loss 诊断](GAWM_ROBOTWIN_NORM_LATENT_LOSS_DIAGNOSIS.md)
- [评测下降与 latent 监督分析](GAWM_ROBOTWIN_POST_EVAL_ANALYSIS.md)

项目使用指南和已有专题文档见 [docs/](../docs/)。
