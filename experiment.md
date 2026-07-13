模型 参数量 libero成功率
LEWM-Tiny 15M 87%

# TODO
- 解冻encoder有没有提升
- 对比其他的编码器，比如DinoV2/V3/INternViT-300M-448px有没有提升
- 更大的world model有没有提升
- 需不需要更大数据集上的预训练？

# 当前结果
- 加state监督，去掉OFT头 （成功率掉到68%） 失败, OFT头会监督action的loss表征，去掉后容易漂移
- dino V2 + OFT 头 = 94, dino V3 差不多，libero整体套件在91%左右，libero 10偏低
- 加上ctx_len和action chunk后没有显著区别（可能训练不足）
- E1之前做过一次修改execute horizon的推理消融，效果不理想。可能原因是模型只在固定action horizon=8上训练，8个action query产生了位置特化，直接缩短执行长度存在训练-推理不匹配。下一步在DINOv2-base同一checkpoint上复测execute_horizon=1/2/4/8；如果仍无提升，后续应改为训练时随机execution horizon或chunk-prefix训练，而不是继续只改推理参数。

## E1复核：DINOv2-base execute horizon
- checkpoint：`lewm_oft_libero_dinov2b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate_v2/steps_160000`（该run已有checkpoint中LIBERO-10最优）。
- LIBERO-10，seed=7，每任务10条，共100 episodes：`execute_horizon=8`为0.79，`execute_horizon=4`为0.68，下降11个百分点。
- 逐任务（horizon 8 -> 4）：`[0.8,0.8,1.0,0.9,0.6,1.0,0.9,0.9,0.4,0.6] -> [0.7,0.8,1.0,0.9,0.4,1.0,0.7,0.7,0.2,0.4]`。
- 结论：纯推理期缩短execute horizon明确无效，且显著有害；支持固定8步训练导致action-query位置特化/固定chunk时序先验的判断。停止继续评测horizon=1/2，下一步应在训练中加入随机prefix/execution-horizon目标，再做推理期匹配消融。

## E1.5：训练期随机prefix辅助损失
- 从DINOv2-base `steps_160000` warm-start，冻结encoder/world model/token pooler，只训练visual action head和OFT MLP 20k steps；目标为`(full_chunk_l1 + random_prefix_l1) / 2`，每个样本从`[1,2,4,8]`随机选择prefix。
- 结果：`execute_horizon=4`从0.68到0.69，基本无提升；`execute_horizon=8`从0.79到0.77，略有遗忘。
- 结论：仅重加权前几个action query不足以解决短闭环退化。问题更接近相邻chunk的轨迹不连续，而不是前缀动作缺少监督。

## E2：时间对齐的overlapping action chunk ensemble
- 修复LIBERO客户端中`AdaptiveEnsembler`只实例化但从未调用的问题，并实现按绝对环境步对齐的`ChunkedAdaptiveEnsembler`。
- DINOv2-base `steps_160000`，LIBERO-10 100 episodes：`execute_horizon=4`无ensemble为0.68，加入等权temporal ensemble后为0.79，与原始`execute_horizon=8`的0.79持平。
- 逐任务（horizon8 baseline -> horizon4+ensemble）：`[0.8,0.8,1.0,0.9,0.6,1.0,0.9,0.9,0.4,0.6] -> [0.8,0.8,1.0,0.8,0.5,1.0,1.0,1.0,0.6,0.4]`。双moka pot任务从0.4提升到0.6，但微波炉任务从0.6降到0.4。
- 结论：短闭环本身可以工作，主要损失来自chunk边界不连续；temporal ensemble值得保留，但总分尚未超过8步基线。

# 长周期成功率实验路线
- E1：固定checkpoint，比较execute_horizon=1/2/4/8；DINOv2-base复核中8步0.79、4步0.68，判定失败并结束纯推理消融。
- E2：已完成；horizon4从0.68恢复到0.79，与horizon8基线持平，但尚未形成净提升。
- E3：进行中；保持action_horizon=8和n_future=2，使用真实过去帧`[t-1,t]`，单独比较ctx_len=1/2。
- E4：对LIBERO-10困难任务过采样，并混入失败状态上的expert recovery数据。
- E5：使用proposal action -> action-conditioned world model rollout -> refined action，避免action-free future latent的多模态歧义。
- E6：训练progress/value head，对多个候选action chunk的world-model rollout评分并执行最优前缀。
- E7：完成闭环、记忆和数据消融后，再比较更大world model、DINO-L和InternViT。

# 用户想法
- 10000 steps 似乎有点少。你每次跑完训练需要看一下loss是否有明显变化。如果没变说明本次迭代改动范围不够。

---