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

# 长周期成功率实验路线
- E1：固定checkpoint，比较execute_horizon=1/2/4/8；已做过一轮且效果不理想，待用DINOv2-base复核。
- E2：在最佳execute horizon下加入overlapping action chunk ensemble，降低chunk边界抖动。
- E3：保持action_horizon=8，单独比较ctx_len=1/2/4或稀疏历史，避免和长action chunk同时改动。
- E4：对LIBERO-10困难任务过采样，并混入失败状态上的expert recovery数据。
- E5：使用proposal action -> action-conditioned world model rollout -> refined action，避免action-free future latent的多模态歧义。
- E6：训练progress/value head，对多个候选action chunk的world-model rollout评分并执行最优前缀。
- E7：完成闭环、记忆和数据消融后，再比较更大world model、DINO-L和InternViT。

# 用户想法
- 10000 steps 似乎有点少。你每次跑完训练需要看一下loss是否有明显变化。如果没变说明本次迭代改动范围不够。

---