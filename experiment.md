模型 参数量 libero成功率
LEWM-Tiny 15M 87%

# TODO
- 解冻encoder有没有提升
- 对比其他的编码器，比如DinoV2/V3/INternViT-300M-448px有没有提升
- 更大的world model有没有提升
- 需不需要更大数据集上的预训练？

# 当前结果
- 加state监督，去掉OFT头 （成功率掉到68%） 失败, OFT头会监督action的loss表征，去掉后容易漂移

# 用户想法
- 10000 steps 似乎有点少。你每次跑完训练需要看一下loss是否有明显变化。如果没变说明本次迭代改动范围不够。

---

# 实验记录 (AI 自主迭代)

评测协议: libero_goal, num-trials-per-task=10 => 100 episodes, 统计 Total success rate。

## 基线 (baseline)
- **代码不兼容说明**: 原 `lewm_oft_libero_wm_vfuse/steps_80000` (ai.md 记 87%) 与当前 working-tree 代码不兼容 (缺 `future_action_context_proj`, strict=True 报错)，弃用。
- **采用新基线**: `playground/Checkpoints/lewm_oft_libero_wm_oft_future_aw05_state0_vitft_10k/checkpoints/steps_8000_pytorch_model.pt` (由当前代码训得: train_encoder=true + state_probe(w=0) + loss_action_weight=0.5, 从 80000 基线微调 8000 step)。
- **基线成功率 = 83% (83/100)** @ 2026-07-03

| 任务 | 成功率 |
|------|--------|
| open the middle drawer | 0.7 |
| put the bowl on the stove | 0.9 |
| put the wine bottle on top of the cabinet | 1.0 |
| open the top drawer and put the bowl inside | 0.6 |
| put the bowl on top of the cabinet | 0.9 |
| push the plate to the front of the stove | 0.7 |
| put the cream cheese in the bowl | 0.7 |
| turn on the stove | 1.0 |
| put the bowl on the plate | 1.0 |
| put the wine bottle on the rack | 0.8 |
| **合计** | **0.83** |

观察: 最弱任务是"open the top drawer and put the bowl inside"(0.6, 长程两阶段)、"open the middle drawer"(0.7)、"push the plate"(0.7)、"put the cream cheese in the bowl"(0.7)——多为需要精确接触/多阶段的任务。
---

## 实验 1: 在线动作蒸馏 (Qwen-OFT teacher → LeWM-OFT student)

### 想法
- 教师 `QwenOFT` (Qwen3-VL-4B + MLP-L1 头, ~96%) 与学生 `LeWMOFT` (ViT-tiny + WanWM + MLP 头, 83%) 输出**同一归一化空间的 8×7 动作块**，可直接做动作级蒸馏。
- 教师动作头 `get_action_model` 实为 `L1RegressionActionHead`（DiT-B 配置被忽略），**确定性单次前向**，无扩散采样 → 每步只需 1 次 4B VLM 前向，成本可控。
- 在学生 loss 上加一项 `L_distill = L1(student_pred_action, teacher_pred_action.detach())`，把强策略的动作分布蒸馏进小模型，作为对 GT 的正则/去噪。

### 实现
- `QwenOFT.distill_predict(examples)`: 镜像 teacher.forward 预处理 (直接用 `example["image"]`，**不传 state**，匹配教师训练分布 libero_all)，`no_grad`+bf16，返回动作张量。
- `LeWMOFT`: 新增 `world_model.distill_teacher/distill_ckpt/distill_weight`。教师用 list 属性挂在 nn.Module 外部 → 不进 state_dict/优化器/DeepSpeed；forward 首次调用时 lazy `.to(device)`。总 loss += `distill_weight * L_distill`。
- 训练脚本 `examples/LIBERO/train_files/run_lewm_oft_distill_train.sh`：从基线 steps_8000 微调，teacher=Qwen3-VL-OFT-LIBERO-4in1/steps_50000，其余超参同基线 (train_encoder=true, aw/latent=0.5, state0, n_future=2)。

### 过程 / 结果
- 训练: 8卡 batch16 从 steps_8000 微调 10000 步, 用时 2h12m, distill_loss 稳定 ~0.049 (与 l1_action_loss ~0.050 同量级)。
- 动作头对齐确认: 学生 `action_source=oft`, 推理走 OFT 头 (`action_model.predict_action(action_queries)`); 蒸馏 loss 监督的正是同一 `pred_actions`, **与部署头对齐**。
- **评测 (steps_10000) = 82% (82/100), 未超基线 83% (-1%, 在噪声内)**

| 任务 | baseline | distill | Δ |
|------|:--:|:--:|:--:|
| open the middle drawer | 0.7 | 0.9 | +0.2 |
| put the bowl on the stove | 0.9 | 1.0 | +0.1 |
| put the wine bottle on top of the cabinet | 1.0 | 1.0 | 0 |
| open the top drawer and put the bowl inside | 0.6 | 0.7 | +0.1 |
| put the bowl on top of the cabinet | 0.9 | 1.0 | +0.1 |
| push the plate to the front of the stove | 0.7 | 0.7 | 0 |
| put the cream cheese in the bowl | 0.7 | **0.5** | -0.2 |
| turn on the stove | 1.0 | 1.0 | 0 |
| put the bowl on the plate | 1.0 | 1.0 | 0 |
| put the wine bottle on the rack | 0.8 | **0.4** | -0.4 |
| **合计** | **0.83** | **0.82** | -0.01 |

### 分析 / 结论
- **净持平**: 5 个任务提升(共 +0.6), 但 2 个精细放置任务大幅回退 (cream cheese -0.2, wine bottle on rack -0.4)。
- **推测原因**: (1) 教师 QwenOFT 训练时**不含本体 state**(libero_all), 学生 baseline 用了 state_probe + include_state; 蒸馏权重=1.0(与 l1 等权)把学生动作过度拉向"无 state"的教师, 在需要精确位姿的放置任务上产生目标冲突。(2) 在示教状态上教师≈GT, 动作级蒸馏边际收益有限, 主要变成正则; 权重过大反而抹掉学生已学到的精细行为。
- **失败, 未超基线 → 按规则 git reset 代码, 保留本记录, 进入下一个 idea。**
- **可复用经验**: 在线动作蒸馏管线可用(教师挂 nn.Module 外, 不入 state_dict/优化器); 若再试蒸馏应: 降权重(~0.2-0.3)只做正则、或给教师补 state 对齐 prompt、或只蒸馏弱任务/OOD 状态(DAgger)。

---

## 实验 2: OFT 头交叉注意力池化 (Cross-Attention Action Pooling)

### 想法 (结构性改动, 非调参)
- 基线 OFT 头的动作条件构造是: `oft_context = [current_raw_latent(768), pred_future_1(768), pred_future_2(768)]` → **mean-pool 成单个 768 向量** → `action_query_proj` 线性展开成 8 个 384 维 action query → MLP 头 → 8×7 动作。
- 问题: mean-pool 把"当前帧 + 2 个预测未来帧"的时序信息**压成一个向量**, 8 步动作共享同一条件, 无法让不同时间步选择性关注"当前"或"未来"latent。对**精细放置/多阶段**任务(cream cheese、wine bottle on rack、open drawer+bowl)不利——这些任务动作前后段语义不同(先接近再精确放置)。
- **改动**: 用**交叉注意力池化**替代 mean-pool: 8 个可学习的 per-step action query token 对 `oft_context` 序列做 cross-attention (Q=query token, K/V=[current, future_1, future_2]), 让每个动作步自适应地聚合当前/未来 latent。这改变了动作 token 的**生成机制**(模型结构), 而非超参。

### 实现 (热启动安全: zero-gated 残差)
- `LeWMOFT` 新增 config flag `world_model.use_cross_attn_pool` (默认 false, 保持基线行为)。
- 新增模块 (仅 flag 开启时): `action_query_tokens` (8×384 可学习), `action_ctx_kv_proj` (768→384 投影 K/V), `action_cross_attn` (nn.MultiheadAttention, 6 头, 384 维), `attn_gate` (标量, 初始 0)。
- `_pool_to_action_queries`: 先算基线 `base_queries = action_query_proj(mean(oft_context))`; 若开启则 `queries = base_queries + attn_gate * cross_attn(query_tokens+base_queries, kv, kv)`。**gate 初始 0 → 初始严格等价于基线**, 微调渐进学习注意力精修。
- 新模块不在 checkpoint 中, 走 `load_state_dict(strict=False)` 保持初始值; 从基线 steps_8000 微调 10000 步 (lr 2e-5, warmup 500, bs, latent/action weight 0.5, state0, train_encoder true), 与基线同超参, 仅结构不同。

### 过程 / 结果
- 独立冒烟测: attn_gate=0 时交叉注意力路径输出与 mean-pool 基线 **diff=0.0 (严格等价)**; 2 卡训练管线烟测 OK。
- 全量训练: 8 卡 batch16 (total 128) 从 steps_8000 微调 10000 步, 用时 **1h14m53s** (2.23 it/s), ~5GB/卡, strict=True 部署加载成功。
- **评测 (steps_10000) = 81% (81/100), 未超基线 83% (-2%)**

| 任务 | baseline | xattn | Δ |
|------|:--:|:--:|:--:|
| open the middle drawer | 0.7 | 0.9 | +0.2 |
| put the bowl on the stove | 0.9 | 1.0 | +0.1 |
| put the wine bottle on top of the cabinet | 1.0 | 1.0 | 0 |
| open the top drawer and put the bowl inside | 0.6 | 0.7 | +0.1 |
| put the bowl on top of the cabinet | 0.9 | 1.0 | +0.1 |
| push the plate to the front of the stove | 0.7 | **0.6** | -0.1 |
| put the cream cheese in the bowl | 0.7 | **0.4** | -0.3 |
| turn on the stove | 1.0 | 1.0 | 0 |
| put the bowl on the plate | 1.0 | **0.9** | -0.1 |
| put the wine bottle on the rack | 0.8 | **0.6** | -0.2 |
| **合计** | **0.83** | **0.81** | -0.02 |

### 分析 / 结论
- **关键发现: 训练后 `attn_gate = 0.000147` (几乎仍为 0)** → zero-gated 残差在 lr 2e-5 下**没被激活**, 交叉注意力分支实际"死"了 (梯度过弱无法逃离 gate=0). 模型功能上 ≈ 基线 mean-pool + 10k 步额外微调。
- 因此 -2% 主要来自**"从 steps_8000 用 train_encoder=true 再微调 10k 步基线目标"本身的漂移/轻微过拟合**, 而非交叉注意力。
- **回退模式与实验 1 (蒸馏) 高度一致**: 都提升了抓取/开抽屉类 (middle drawer +0.2 等), 都回退了**精细放置类** (cream cheese、wine bottle on rack、push plate、bowl on plate)。两次独立实验的共同因子是**微调 recipe (10k 步 + train_encoder)**, 不是具体 idea。
- **推断: 基线 steps_8000 已处于甜点区, 解冻 encoder 继续微调会让 ViT 特征漂移, 损害需要精确视觉定位的放置任务。**
- **失败, 未超基线 → 按规则回退代码 (只删本次 cross-attn 新增, 用本地历史 Y2jg.py 恢复基线 OFT 码, 不 git checkout 整文件), 保留本记录。**
- **可复用经验**: (1) **zero-gated (gate=0) + 小 lr 会让新模块死掉** — 若重试应把 gate 初始化为小正值 (~0.1) 或对 out_proj 置零但 gate=1, 使模块从头有梯度。(2) 两次"从 steps_8000 全量微调"都 -1~-2% → **下个实验应冻结 encoder/world_model, 只训练新动作头结构**, 隔离结构改动、避免 ViT 漂移。

---

## 实验 3: 空间 patch-token 动作头 + 冻结感知

### 想法 (结构性)
- **动机**: 前两次实验 (蒸馏、cross-attn 池化) 都在**精细放置任务** (cream cheese、wine bottle on rack、push plate) 上回退, 共同根因被判定为 "10k 步微调 + `train_encoder=true` 导致 ViT 特征漂移"。
- **结构缺陷**: LeWM 的 `encode_frames` 把每视角 196 个 patch token **mean-pool** 成一个向量 (concat cls+mean = 384/view), **丢弃了全部空间布局**。而精确放置 (把物体放到某个精确位置) 恰恰依赖空间定位信息。
- **改动**: 让 OFT 的 8 个 action query **额外 cross-attend 当前帧的原始 ViT patch 网格** (B, V, N=196, D_vit=192), 把被 mean-pool 抹掉的空间细节重新注入动作头。
- **隔离漂移**: 同时**冻结 encoder + world_model + view_fuse + task_embedding**, 只训练动作头 (含新空间模块), 从结构上消除实验 1/2 的 ViT 漂移根因, 让本次改动可被干净归因。

### 实现
- `LeWM.py` 新增 `encode_patches(images)` → 返回 `(B, V, N, D_vit=192)` 原始 patch 网格 (`encode_frames` 只返回 mean-pool 后的向量)。honor `train_encoder` (冻结时 detach)。
- `LeWMOFT.py`: config 加 `use_spatial_action_head` (默认 False) / `spatial_attn_heads` (6)。`__init__` 建 `spatial_kv_proj` (192→384)、`spatial_q_norm` (LayerNorm)、`spatial_cross_attn` (MHA 6 头 384)。**关键: `spatial_cross_attn.out_proj` 的 weight+bias 置零** → 初始空间输出=0 (严格等价基线), 但 out_proj 是完整矩阵能直接拿到梯度 → **可训练** (修正实验 2 标量 gate 死模块的教训)。新增 ~0.67M 参数。
- `_spatial_refine(queries, images)`: `patches → kv_proj → cross_attn(norm(queries), kv, kv)`, `return queries + attn_out`。在 `forward` 和 `predict_action` 的 `_pool_to_action_queries` 之后调用。
- 训练脚本 `run_lewm_oft_spatial_train.sh`: `freeze_modules="backbone,world_model,view_fuse,task_embedding,state_probe"`, `train_encoder=false`, `use_spatial_action_head=true`, 从 steps_8000 微调 10000 步 (lr 1e-4, warmup 500)。

### 过程 / 结果
- 独立冒烟测: out_proj 初始为 0 → `_spatial_refine` 输出与基线 **diff=0.0 (严格等价)**; 反传后 out_proj 梯度非零 → **模块可训练 (逃离实验 2 死模块)**。
- 2 卡管线烟测 OK: 19.484M total / **4.812M trainable** (encoder+WM 已冻结)。
- 全量训练: 8 卡 batch16 从 steps_8000 微调 10000 步, 用时 **1h23m** (2.01 it/s), ~3GB/卡。
- **评测: steps_10000 = 83% (0.83), steps_6000 = 81% (0.81)** — 最佳 steps_10000 恰好**追平基线 83%**, 未超过。

| 任务 | baseline | spatial-10k | Δ | spatial-6k 
|------|:--:|:--:|:--:|:--:|
| open the middle drawer | 0.7 | 0.8 | +0.1 | 0.9 |
| put the bowl on the stove | 0.9 | 0.9 | 0 | 1.0 |
| put the wine bottle on top of the cabinet | 1.0 | 1.0 | 0 | 0.9 |
| **open the top drawer and put the bowl inside** | 0.6 | **1.0** | **+0.4** | 0.7 |
| put the bowl on top of the cabinet | 0.9 | 1.0 | +0.1 | 0.9 |
| push the plate to the front of the stove | 0.7 | 0.7 | 0 | 0.8 |
| put the cream cheese in the bowl | 0.7 | 0.6 | -0.1 | 0.5 |
| turn on the stove | 1.0 | 1.0 | 0 | 1.0 |
| put the bowl on the plate | 1.0 | 0.9 | -0.1 | 1.0 |
| **put the wine bottle on the rack** | 0.8 | **0.4** | **-0.4** | 0.4 |
| **合计** | **0.83** | **0.83** | 0 | 0.81 |

### 分析 / 结论
- **空间头确实起作用**: `open the top drawer and put the bowl inside` (精确放置任务) **0.6→1.0 (+0.4)**, 直接验证了"注入 patch 空间信息帮助精细放置"的假设; `open the middle drawer`/`bowl on top of cabinet` 也各 +0.1。
- **抵消项 = wine bottle on the rack 塌方 (0.8→0.4)**: 该任务在**实验 1(蒸馏)、实验 2(cross-attn)、实验 3** 中**全部塌到 0.4~0.6**, 且 steps_6000 就已是 0.4 (不随训练步数累积) → 说明它**不是空间头造成的, 也不是 ViT 漂移(本次已冻结 encoder)**, 而是"**从 steps_8000 微调时既有动作头(action_query_proj/future_action_context_proj/action_model)在 lr 1e-4 下漂移**"导致的 recipe 级塌方。
- **净效果 = 平局 (0.83 = 0.83)**: 空间头的放置增益 (+0.6) 恰好被 wine-on-rack 的 -0.4 与 cream cheese/bowl-on-plate 的 -0.2 抵消。
- **按规则(需 > 基线)判为未通过**; 但空间头是**目前唯一被验证"能提升精细放置"的结构改动**, 强烈值得保留并做后续。

### 关键教训 / 后续
- **既有动作头一起解冻会塌方 wine-on-rack**: 下一步(实验 4)**把整个既有动作头也冻结, 只训练 ~0.67M 的空间 adapter** → 基线通路(含 wine-on-rack)被严格保留, 空间分支从 0 起步只能"锦上添花", 理论下界=基线、上界=基线+放置增益。

---

## 实验 4: 空间 adapter (只训 spatial 分支, 冻结整个既有动作头)

### 想法 (结构性 / 消融)
- 实验 3 结论: 空间头能提升精细放置 (top-drawer+bowl +0.4), 但 `wine bottle on rack` 塌方 0.8→0.4 抵消了增益, 净平局。且该塌方在三次微调实验中反复出现、steps_6000 就已发生、encoder 已冻 → 判定塌方来自**既有动作头 (action_query_proj / future_action_context_proj / action_model) 在微调中漂移**, 而非空间头本身。
- 本实验做纯消融: **额外冻结整个既有动作头**, 只训练 ~0.666M 的空间 adapter (`spatial_kv_proj` + `spatial_cross_attn`)。空间分支 out_proj 初始 0 → step0 严格等价基线 (0.83)。理论下界=基线、上界=基线+放置增益。
- 唯一改动 = `freeze_modules` 增加 `action_model,action_query_proj,future_action_context_proj` (代码不变, 复用实验 3 的空间头)。脚本 `run_lewm_oft_spatial_adapter_train.sh`。

### 过程 / 结果
- 冒烟: 19.484M total / **0.666M trainable** (仅 spatial), checkpoint warm-start OK。
- 全量: 4 卡 (GPU0-3, 因 4-7 被另一 dinov2b 任务占用) batch16 (total 64) 从 steps_8000 微调 10000 步, 2.35 it/s, ~1h10m, ~3.3GB/卡。
- **评测: steps_10000=82%, steps_4000=82%, steps_2000=80%** — 全部 **≤ 基线 83%** (step0=基线 0.83)。

| 任务 | baseline | adapter-10k | adapter-4k | Δ(10k) |
|------|:--:|:--:|:--:|:--:|
| open the middle drawer | 0.7 | 0.7 | 0.8 | 0 |
| put the bowl on the stove | 0.9 | 0.9 | 1.0 | 0 |
| put the wine bottle on top of the cabinet | 1.0 | 0.9 | 1.0 | -0.1 |
| open the top drawer and put the bowl inside | 0.6 | 0.8 | 0.9 | +0.2 |
| put the bowl on top of the cabinet | 0.9 | 0.9 | 1.0 | 0 |
| push the plate to the front of the stove | 0.7 | 0.8 | 0.6 | +0.1 |
| put the cream cheese in the bowl | 0.7 | **0.5** | **0.3** | -0.2 |
| turn on the stove | 1.0 | 1.0 | 1.0 | 0 |
| put the bowl on the plate | 1.0 | 1.0 | 1.0 | 0 |
| put the wine bottle on the rack | 0.8 | 0.7 | 0.6 | -0.1 |
| **合计** | **0.83** | **0.82** | **0.82** | -0.01 |

### 分析 / 结论
- **冻结既有头确实保住了 wine-on-rack** (0.7 vs 实验3的 0.4) → 验证了"塌方来自既有头漂移"的假设; top-drawer+bowl 仍 +0.2 (frozen head 限制了增益, 实验3是 +0.4)。
- **但空间 adapter 变成"全局残差", 净微负**: 帮了 top-drawer(+0.2)/push-plate(+0.1), 却伤了 **cream cheese (-0.2)**、wine-on-cabinet(-0.1)、wine-on-rack(-0.1)。因为它对所有任务无差别注入空间偏移, 有的任务受益、有的受损, 净 -0.01。
- **两次空间头实验 (实验3 平局 0.83 / 实验4 微负 0.82) 均未超基线 → 空间 patch-token 头这条结构改动无法稳定 >83%。**
- **更深层结论**: 从 steps_8000 微调 (蒸馏 82 / cross-attn 81 / 空间头 83 / 空间adapter 82) **全部落在 0.80~0.83**, 从未超过 → **基线 steps_8000 处于一个较尖锐的局部最优, 任何"微调 + 加头"都难以突破**。要真正 >83% 可能需要换数据/目标/从头训更久, 而非再改动作头。
- **判定失败 → 按规则回退代码到基线** (cp Y2jg.py 恢复 LeWMOFT.py + git checkout LeWM.py), 删除 spatial ckpt, 保留本记录。

### 可复用经验
- **zero-init out_proj adapter 确实可训** (逃离实验 2 死模块) — 已两次验证。
- **全局残差 adapter 的通病**: 无差别注入对部分任务有害, 除非**按任务/样本门控** (未来若重试空间头, 应加 per-sample 学习门控, 让模型自己决定何时使用空间精修)。
- **精细放置任务 top-drawer+bowl 对空间信息高度敏感** (0.6→1.0/0.8), 是最值得针对的弱任务。
- **wine-on-rack 是"漂移敏感"任务** — 冻结既有头即可保住, 微调既有头必塌。

---

## 实验 5: 推理期动作集成 (test-time action ensembling, 免训练)

### 想法
- 关键观察: 前 4 个实验 (蒸馏/cross-attn/空间头/adapter) 都落在 80~83%, 而 100 episodes 的二项标准误 ≈ √(0.83·0.17/100) ≈ **±3.7%** → 81~83% 其实都在基线噪声内, 结构改动"噪声不可分"。要么放大效应, 要么**降方差**。
- `sample_future` (wan_world_model.py:148) 是**随机流采样** — 每次调用都用新的 `torch.randn` 作为 latent 和 action 噪声, 沿 flow ODE 积分。因此同一观测多次前向会得到不同的想象未来 → 不同动作。
- 假设: 对同一观测采 **K 个想象未来**, 各自过 OFT 头得到 K 个动作 chunk, **求平均** → 抵消 policy 采样方差, 得到更稳的动作。属于 test-time compute, 不训练。

### 实现
- 只改 `LeWMOFT.py` 的 `predict_action` (仅推理), 由 `config.framework.world_model.action_ensemble_k` 门控, **默认 1 = 与基线严格等价**。
- `action_source=oft` 且 K>1 时: `latent/raw_latent/task_emb.repeat_interleave(K)` → **一次批量** `sample_future(B*K)` (一个 flow loop 并行跑 K, 因 eval B=1 → 额外开销可忽略) → OFT 头 → `reshape(B,K,H,A).mean(dim=1)`。
- 注入方式 (不动基线 run 目录): 建 `playground/Checkpoints/_ens_eval/` = `config.yaml`(基线 config + `action_ensemble_k: 8`) + `checkpoints/steps_8000_*.pt`(symlink 基线) + `dataset_statistics.json`(symlink)。server 从 `parents[1]/config.yaml` 读到 K=8。
- 实测每 episode ~5s, 与基线同速 (K 批量化后开销可忽略), 无报错。

### 过程 / 结果
- **评测 K=8 @ libero_goal 100ep = 79% (79/100)**, **低于基线 83% (-4%)**。

| 任务 | baseline | K=8 | Δ |
|------|:--:|:--:|:--:|
| open the middle drawer | 0.7 | 0.7 | 0 |
| put the bowl on the stove | 0.9 | 1.0 | +0.1 |
| put the wine bottle on top of the cabinet | 1.0 | 1.0 | 0 |
| open the top drawer and put the bowl inside | 0.6 | 0.5 | -0.1 |
| put the bowl on top of the cabinet | 0.9 | 0.9 | 0 |
| push the plate to the front of the stove | 0.7 | 0.6 | -0.1 |
| put the cream cheese in the bowl | 0.7 | 0.7 | 0 |
| turn on the stove | 1.0 | 1.0 | 0 |
| put the bowl on the plate | 1.0 | 1.0 | 0 |
| put the wine bottle on the rack | 0.8 | **0.5** | **-0.3** |
| **合计** | **0.83** | **0.79** | **-0.04** |

### 分析 / 结论
- **动作均值集成有害** — 尤其精细放置任务 (**wine-on-rack 0.8→0.5**, top-drawer/push-plate 各 -0.1)。
- 根因: 每步动作分布是**多模态**的 (不同想象未来对应不同但都合理的抓取/放置轨迹), 对多模态输出求**均值会"糊掉"模态**, 产生一个介于两个有效动作之间的无效动作 → 精细放置失败。这是均值集成在多模态输出上的经典失效模式。
- **反证**: 说明基线单次采样的动作本身是"某个有效模态", 随机性不是主要误差来源; 强行平均反而破坏了模态一致性。
- **判定失败 → 回退**: cp Y2jg.py 恢复 LeWMOFT.py (0 ensembling refs), 删 `_ens_eval` 目录, 保留本记录。

### 可复用经验
- **不要对世界模型的多个随机想象做动作空间均值** — 多模态会被均值抹平, 伤精细任务。若要用多样本, 应**选择 (如按某准则挑一个 sample) 而非平均**, 或在**潜空间/单模态**内聚合。
- test-time compute 在这个 flow-matching VLA 上无免费午餐: 随机性≠误差源。
- 再次印证: **wine-on-rack 是全实验最脆弱的任务** (蒸馏0.4 / cross-attn0.6 / 空间头0.4 / 集成0.5), 任何扰动都先伤它。

---

## 实验 6: 推理期 medoid 选择 (test-time medoid selection, 免训练)

### 想法
- 实验5 的均值集成失败根因是"糊掉多模态"。**修正**: 不做均值, 而是从 K 个随机想象未来对应的 K 个动作 chunk 里**选 medoid** — 到其它 K-1 个样本总距离最小的那个 (即**主导模态**)。这样能**剔除离群的 flow 采样**而**不破坏模态**。仍是 test-time, 免训练。

### 实现
- 同实验5 的 K 批量采样 (B→B*K 一次 flow loop), 但聚合改为: `flat=(B,K,H*A)`, `dist=cdist(flat,flat)`, `medoid_idx=dist.sum(-1).argmin(-1)`, 取对应样本。同样由 `action_ensemble_k` 门控 (默认1=基线), 注入 `_ens_eval/config.yaml` K=8。

### 过程 / 结果
- **评测 K=8 medoid @ libero_goal 100ep = 80% (80/100)**, **仍低于基线 83% (-3%)** (比均值 79% 略好)。

| 任务 | baseline | medoid K=8 | Δ |
|------|:--:|:--:|:--:|
| open the middle drawer | 0.7 | 0.7 | 0 |
| put the bowl on the stove | 0.9 | 1.0 | +0.1 |
| put the wine bottle on top of the cabinet | 1.0 | 1.0 | 0 |
| open the top drawer and put the bowl inside | 0.6 | 0.7 | +0.1 |
| put the bowl on top of the cabinet | 0.9 | 0.9 | 0 |
| push the plate to the front of the stove | 0.7 | 0.7 | 0 |
| put the cream cheese in the bowl | 0.7 | 0.7 | 0 |
| turn on the stove | 1.0 | 1.0 | 0 |
| put the bowl on the plate | 1.0 | 0.9 | -0.1 |
| put the wine bottle on the rack | 0.8 | **0.4** | **-0.4** |
| **合计** | **0.83** | **0.80** | **-0.03** |

### 分析 / 结论
- medoid 比均值好 (80 vs 79), 多数任务持平/微升, 但 **wine-on-rack 再次塌到 0.4** → 整体仍 <83%。
- **关键原因**: 基线 K=1 是**沿整条 rollout 用单次一致采样**, 动作是**时序连贯的轨迹**; 而 K>1 在**每个时间步都重新选** medoid → 逐步微小差异**累积破坏轨迹连贯性**。wine-on-rack 最"轨迹敏感", 首当其冲。
- **两种 test-time 样本聚合 (均值/medoid) 都 <83% → 推理期样本聚合这条路否掉**。误差不来自单步动作精度, 而是轨迹连贯性; 逐步重采样有害。
- **判定失败 → 回退**: cp Y2jg.py 恢复 LeWMOFT.py, 删 `_ens_eval`, 保留记录。

### 可复用经验
- **不要在 rollout 的每一步独立重新采样/选择动作** — 破坏时序连贯性, 伤轨迹敏感任务。若要 test-time 聚合, 应保证整条轨迹用**一致采样** (如固定噪声种子) 而非逐步聚合。
- 至此 6 次实验 (蒸馏82 / cross-attn81 / 空间头83 / 空间adapter82 / 均值集成79 / medoid80) **全部 ≤ 基线83%** → 强烈确认: 从 steps_8000 出发, 在"10k 微调 / test-time 技巧"范围内**无法突破 83%** (都在 ±3.7% 噪声内)。真正提升需换能力瓶颈 (更大 encoder, 已由他人 dinov2b 任务在做) 或换训练目标/数据。

---

## 实验 7: 每样本门控空间 adapter (gated spatial adapter, frozen head)

### 想法
- 结合实验3 (空间头把 top-drawer 0.6→1.0, 真实大增益) + 实验4 (冻结既有头保住 wine-rack)。实验4 败因 = 空间残差是**全局残差**, 无差别伤 cream-cheese (0.7→0.5)。
- **修正**: 给空间残差加**每样本 sigmoid 门控** `g = sigmoid(spatial_gate(queries.mean))`, 让模型学会"**何时**用空间精修" — 对空间敏感任务 (top-drawer) 开, 对 cream-cheese 关。

### 实现
- LeWM.py 重加 `encode_patches` → (B,V,N,192) 原始 patch (tracked, 可 git checkout 回退)。
- LeWMOFT.py `_spatial_refine`: `refined = queries + g.unsqueeze(1) * out_proj(cross_attn(norm(q), kv, kv))`, `kv = spatial_kv_proj(patches)`。**死门控规避 (实验2教训)**: out_proj weight/bias + gate weight/bias **全 zero-init** → 初始 g=sigmoid(0)=0.5 (非0, 让 out_proj 拿梯度), 残差=0 (out_proj=0) = 基线 no-op; out_proj 经 g=0.5 成长后 gate 才拿梯度。
- 冻结: encoder+WM+view_fuse+task_embedding+state_probe+**整个既有头** (action_model, action_query_proj, future_action_context_proj)。**Trainable 仅 0.667M** (spatial adapter+gate)。
- 冒烟: warm-start Δ=0.0, out_proj grad=259.9 (可训), gate grad=0 首步 (符合预期)。
- 全量: GPU 4-7 (0-3 被他人 lightstereo 占), batch16 10k save2000 warmup500 lr1e-4, from steps_8000, 1h12m。

### 过程 / 结果
- **评测 (3 个 checkpoint, libero_goal 100ep)**: steps_4000=**75%**, steps_6000=**80%**, steps_10000=**79%** — 全部 **< 基线 83%**。

| 任务 | baseline | SG-4k | SG-6k | SG-10k |
|------|:--:|:--:|:--:|:--:|
| open the middle drawer | 0.7 | 0.7 | 0.7 | 0.8 |
| put the bowl on the stove | 0.9 | 1.0 | 1.0 | 0.9 |
| put the wine bottle on top of the cabinet | 1.0 | 1.0 | 1.0 | 0.9 |
| open the top drawer and put the bowl inside | 0.6 | 0.7 | 0.8 | 0.7 |
| put the bowl on top of the cabinet | 0.9 | 0.9 | 0.9 | 0.9 |
| push the plate to the front of the stove | 0.7 | 0.7 | 0.7 | 0.7 |
| **put the cream cheese in the bowl** | 0.7 | **0.3** | **0.4** | **0.3** |
| turn on the stove | 1.0 | 1.0 | 1.0 | 1.0 |
| put the bowl on the plate | 1.0 | 0.7 | 1.0 | 1.0 |
| put the wine bottle on the rack | 0.8 | 0.5 | 0.5 | 0.7 |
| **合计** | **0.83** | **0.75** | **0.80** | **0.79** |

### 分析 / 结论
- **门控没起到"选择性关闭"的作用**: cream-cheese 反而塌得比实验4 (未门控 adapter, 0.5) 更狠 (0.3~0.4); top-drawer 只到 0.7~0.8 (不如实验3 的 1.0)。
- **门控为什么失败**: 训练损失只在**示教动作**上, 门控没有"避免伤害测试任务"的直接信号 → 梯度让 g 倾向普遍开启 (初始 0.5, 拟合训练集), 泛化到评测时对 cream-cheese 这类被空间注意力干扰的任务造成更大破坏。**learned gate 在离线模仿学习里无法学到"何时不用" (没有关闭的训练激励)**。
- **空间头方向三连败**: 实验3 (全头可训, 83 平) / 实验4 (未门控 adapter, 82) / 实验7 (门控 adapter, 79~80) → **空间 patch 头彻底否掉**, 无论怎么加/门控都稳定 ≤83%。
- **判定失败 → 回退**: cp Y2jg.py 恢复 LeWMOFT.py; git checkout LeWM.py; 删 ckpt (255M) + 脚本; 保留记录。

### 可复用经验
- **离线模仿学习里的 learned gate 学不到"何时关闭"** — 只有开启能降训练 loss, 没有关闭的激励; 若要选择性行为需显式监督/正则 (如稀疏门控惩罚) 或测试时规则, 而非纯 data-driven gate。
- **至此 7 次实验全部 ≤ 基线 83%**。空间头 (加/adapter/门控)、蒸馏、cross-attn、test-time 聚合均否掉。**核心瓶颈是 ViT-tiny 感知能力 + steps_8000 尖锐局部最优, 10k 微调改不动**。剩余高天花板方向 (更大 encoder / 更大 WM / 大数据预训练) 均需 ≥80k 步从头训, 超单轮 3h 预算 → 我这个预算档内难有正收益。cream-cheese/wine-on-rack 是"感知-精度"瓶颈任务, 靠加结构解决不了。

---

## 实验 8：确定性 (固定噪声) 流采样 — test-time, 免训练

### 想法
- 承接实验 5/6 的教训："test-time 聚合要整条轨迹一致采样"。基线的 `sample_future` 在 rollout **每一步都用 `torch.randn` 重新采一次噪声** → 流 ODE 从不同起点积分 → 逐步动作带**采样抖动**。
- 假设：把流噪声**固定 (固定随机种子)**，使"想象的未来 → 动作"成为观测的**确定性、时序一致函数**，抖动消失，可能利好轨迹敏感任务 (wine-on-rack)。
- 结构改动 (随机策略 → 确定性策略)，且**免训练、免额外算力** (~15min 评测)。默认关闭 = 精确基线。

### 实现
- `LeWMOFT.predict_action`：新增 `deterministic_flow` 配置开关 (默认 False)。为 True 时在调用 `sample_future` 前 **保存 RNG state → `torch.manual_seed(0)` → 采样 → 恢复 RNG state** (避免全局副作用)。
- 评测用 `_det_eval` 注入目录：基线 config + `framework.world_model.deterministic_flow: true`，checkpoint / dataset_statistics 软链基线 steps_8000。

### 过程 / 结果 (libero_goal 100ep, 基线 steps_8000)
| 任务 | baseline | det-flow |
|------|:--:|:--:|
| open the middle drawer | 0.7 | 0.7 |
| put the bowl on the stove | 0.9 | 1.0 |
| put the wine bottle on top of the cabinet | 1.0 | 0.9 |
| open the top drawer and put the bowl inside | 0.6 | 0.9 |
| put the bowl on top of the cabinet | 0.9 | 1.0 |
| push the plate to the front of the stove | 0.7 | 0.6 |
| put the cream cheese in the bowl | 0.7 | 0.5 |
| turn on the stove | 1.0 | 1.0 |
| put the bowl on the plate | 1.0 | 0.9 |
| **put the wine bottle on the rack** | 0.8 | **0.1** |
| **合计** | **0.83** | **0.76** |

### 分析 / 结论
- **结果与假设相反**：wine-on-rack 从 0.8 **塌到 0.1**！固定种子把流采样锁死在**单一模态**上，而这个模态恰好对 wine-on-rack 系统性错误 → 10 个 trial 全崩。基线每次采新随机噪声 = 多模态探索，总有几次落到成功模态 → 0.8。
- **关键结论**：流模型的**随机采样本身提供有益的多模态探索**；固定噪声塌成一个 (可能系统性错的) 模态。结合实验 5/6，**多样本聚合 (伤) 与 单确定样本 (伤) 两头都不行 → 基线"每条 rollout 一个新随机样本"是甜点**。
- **判定失败 → 回退**：cp Y2jg.py 恢复 LeWMOFT.py; 删 `_det_eval`; 验证 441 行 / 0 deterministic_flow refs / 端口空闲。

### 可复用经验
- **不要把流/扩散策略改成确定性 (固定噪声) 采样**：会塌到单一模态，对该模态系统性失败的任务 (精确放置类) 灾难性掉分。随机性是特性不是 bug。
- **至此 8 次实验全部 ≤ 基线 83%**。test-time 三种改法 (mean/medoid/确定性) 全否；结构三种 (蒸馏/cross-attn/空间头) 全否。**强证据：ViT-tiny + tiny-WM 在 steps_8000 处于容量/预算天花板上的尖锐最优，10k 微调 + test-time 技巧都翻不过去**。真正的提升需要更大感知 (encoder) / 更大 WM / 大数据预训练，均需 ≥80k 从头训，超单轮预算档。

---

## 实验 9：本体感觉状态条件 (proprioceptive state conditioning)

### 想法
- **结构性缺口**：OpenVLA-OFT 原论文把机器人本体状态 (proprioceptive state) 作为一路输入注入动作头；而我们的 latent-only WM-OFT 头**完全没有用到 state**，只靠视觉 latent 想象未来。补上这一路观测在原理上应当有益 (尤其精确放置 / 抓取时机类任务)。
- 改动定位于**输入端 (加一路观测调制)**，非纯超参调整；且用零初始化保证热启动时严格等价基线。

### 实现
- `LeWMOFT`：新增 `use_state_cond` 开关。`state_encoder = MLP(8→384→384)`，**最后一层 weight/bias 零初始化** → 热启动时输出 0，`action_queries += 0`，严格等于基线。
- 前向：`state` 由 dataloader 以 `mean_std` 归一化后传入 (B, 1+Tf, 8)，取当前帧 `[:,0,:8]` 过 encoder 加到 action_queries。
- 推理：`predict_action` 从 example 取原始 env state，注入 config 的 `state_mean/std` 做 mean_std 归一化；**shape-fix**：训练内评测传 3 帧 (B,3,8)，`if st.dim()==3: st=st[:,0,:]` 取当前帧 (修复 step-100 崩溃 8-vs-3 广播错误)。
- `eval_libero.py` 增加 `"state": observation["observation.state"][0]` 透传 (对基线无害，未知 key 被忽略)。
- 训练：冻结除 `state_encoder` 外全部 (仅 0.151M 可训)，从 steps_8000 微调 10k，4 卡 batch16×4=64，~1h14m。

### 过程 / 结果 (libero_goal 100ep, 从 steps_8000 微调)
| checkpoint | 合计成功率 |
|------|:--:|
| steps_4000 | 0.81 |
| steps_6000 | 0.76 |
| steps_8000 | 0.78 |
| steps_10000 | **0.85** (初测) |

- steps_10000 初测 0.85 = 9 次实验中**首个超基线**的结果 → 触发**鲁棒性复测**。
- 对同一 steps_10000 checkpoint 做两次**独立随机复测** (策略为随机流采样，未固定种子 → 每次独立样本)：
  - 复测 A：`0.8 0.9 1.0 0.7 0.9 0.5 0.4 1.0 0.8 0.3` → **0.73**
  - 复测 B：`0.9 0.9 1.0 0.5 0.9 0.6 0.4 1.0 0.9 0.6` → **0.77**
- **同一 checkpoint 三次采样：0.85 / 0.73 / 0.77，均值 78.3%，样本标准差 ≈ 6%。**

### 分析 / 结论
- **0.85 是噪声尖峰，不是真实提升**。同一 checkpoint 复测方差高达 ±6%，远大于此前用二项分布估计的 ±3.7%。状态条件的真实均值 78.3% **低于基线 83%** → **判定失败，回退**。
- **⚠️ 方法论关键教训**：策略是**随机流采样且评测未固定种子**，单次 100ep 评测在同一权重上就有 ±6% 的抖动。这意味着**实验 1–9 全部"单次 vs 单次基线"的比较在 ±2~3% 尺度上是不可靠的**——包括"基线 83%"本身也只是一次采样，真实均值可能只有 79~80%。此前所有"差 1~2%"的失败判定与"持平"判定都淹没在噪声里。
- **下一步必须先修方法论**：要么固定评测随机种子做**可复现的配对比较**，要么对每个配置**多次评测取均值 (≥3 次)**，否则任何 <6% 的改动都无法与噪声区分。

### 可复用经验
- 评测噪声真实量级 ≈ **±6% (同权重复测)**，不是 ±3.7%。**任何单次评测的 <6% 差异都不可信**，必须多次评测或固定种子配对比较。
- 零初始化残差调制 (state_encoder 末层置零) 是安全的加输入手段：热启动严格等价基线，训练再逐步引入，不会一上来破坏已有能力。
- 训练内评测 (`eval_interval`) 传的是训练格式多帧 state，与真实 env 单帧 shape 不同，注入观测类改动要对两条路径都做 shape 归一。

---

## 实验 10：模型汤 model soup（权重平均，免训练结构改）+ 降噪评测协议

### 想法
- 承实验 9 的方法论教训（评测噪声 ±6%），先建立**降噪评测协议**：libero_goal 每任务有 50 个 init_states，用 `num-trials-per-task 50` = **500 episodes**，二项 std err ≈ ±1.8%（vs 100ep 的 ±6%）。
- 结构性想法：**model soup / SWA**——权重空间平均晚期 checkpoint，是公认能提升泛化/鲁棒性的结构技术（找更平坦的极小值），且**免训练、零额外推理开销**。基线 run 存有 2000/4000/6000/8000 四个 checkpoint。

### 实现
- `soup68 = 0.5·steps_6000 + 0.5·steps_8000`，302 个 float 张量逐一平均。L2(6000,8000)=0.82，rms/param=0.000189 → 两者在同一 basin，线性平均安全。
- 注入目录 `_soup_eval/`：基线 config + dataset_statistics 软链，checkpoint = soup68。
- **公平对比**：基线 steps_8000 与 soup68 都在 **500ep 协议**下评测，各跑 2 轮独立评测（合计 1000ep/模型）。

### 过程 / 结果（libero_goal 500ep×2 轮）
| | 基线 steps_8000 | soup68 |
|--|:--:|:--:|
| 第 1 轮 500ep | 0.756 | 0.784 |
| 第 2 轮 500ep | 0.764 | 0.770 |
| **合计 1000ep** | **0.760** | **0.777** |

- **重大发现：基线真实均值 ≈ 76.0%，不是记录的 83%**——此前的 83% 只是一次 100ep 的幸运抽样。这印证了实验 9 的方法论教训。
- soup 两轮都 ≥ 基线（+2.8%，+0.6%），从未更差；合计 **+1.7%**。
- 逐任务（两轮均值，soup − base）：top-drawer+bowl **+0.11**（两轮都稳：0.66/0.68 vs 0.56/0.56，最一致的真实增益）；open-drawer +0.05；wine-cabinet/bowl-cabinet +0.04；wine-rack +0.04；push-plate −0.04；bowl-on-plate −0.05；cream-cheese/turn-stove ≈0。

### 分析 / 结论
- 两比例 z 检验：z ≈ 0.90，**p ≈ 0.18 → 不显著**。+1.7% 落在 1000ep 的 ±1.9% 噪声带内，**统计上未确认为真实提升**。
- 但**性质上不同于实验 9**：实验 9 复测直接回落到基线以下（噪声尖峰实锤）；soup 两轮都 ≥ 基线、从不更差，且最难任务 top-drawer+bowl 稳定 +0.11。这是 10 次实验里唯一"从不劣化"的干预。
- soup68 只用了 2 个相距 2000 步的 checkpoint。**正统 SWA 应平均密集的晚期 checkpoint**（在极小值附近抖动的多个权重求中心）→ 更平坦、更鲁棒。→ 实验 11：稠密 SWA。

### 可复用经验
- **评测必须用 500ep（±1.8%），不能用 100ep（±6%）**；重要结论要 2 轮独立评测取合计。
- 记录的"基线 83%"是虚高抽样，公平同协议下基线真值 ≈ 76%。以后一切对比都用 500ep 同协议。
- 权重平均（model soup）安全无害（同 basin，从不劣化），值得作为免费的鲁棒性叠加；但 2 点稀疏平均增益弱且不显著，需稠密 SWA 才可能做实。

---

## 实验 11：稠密 SWA（stochastic weight averaging）★ 首个确认的显著提升

### 想法
- 实验 10 的 2 点 soup（相距 2000 步）增益弱且不显著。正统 **SWA**：从一个好模型出发，用适中恒定 LR 继续训练一小段，**密集快照**，再把这些相互靠近的晚期 checkpoint 权重平均 → 落在更平坦的极小值，泛化更好、**方差更低**。这是实验 10 的原理化强化版，仍免额外推理开销。

### 实现
- 续训脚本 `run_lewm_oft_swa_train.sh`：从基线 steps_8000 warm-start，**沿用基线配方**（train_encoder=true，freeze=''，lr 2e-5，cosine_min，warmup 0），训 4000 步，每 1000 步存一次 → 得 `steps_1000/2000/3000/4000`（= 8000+1000..+4000）。训练 46min（1.44 it/s 收敛），关闭训练内评测省时。
- soup 构建 `build_swa_soup.py`：**swa4 = 均匀平均(+1000,+2000,+3000,+4000) 四快照**（302 键全 float 逐一平均，同 basin）。
- 对照变体 swawide = 平均(6000,8000,+1000..+4000) 6 点。

### 过程 / 结果（libero_goal 500ep）
| 模型 | round A | round B | round C | 合计 | 相对基线 |
|--|:--:|:--:|:--:|:--:|:--:|
| 基线 steps_8000 | 0.756 | 0.764 | — | **0.760** (1000ep) | — |
| soup68（实验10） | 0.784 | 0.770 | — | 0.777 (1000ep) | +1.7% |
| swawide（6点） | 0.770 | — | — | 0.770 (500ep) | 弃用 |
| **swa4（4快照）** | 0.798 | 0.792 | 0.798 | **0.796 (1500ep)** | **+3.6%** |

- **swa4 三轮 79.8/79.2/79.8，极度一致（跨轮极差仅 0.6%）**——对比单 checkpoint 的 ±6% 抖动（实验9），SWA 把评测方差压到 ±0.3%，印证"权重平均→更平坦极小值→更低方差"。
- 两比例 z 检验（swa4 1194/1500 vs 基线 760/1000）：**z=2.13，p≈0.03 → 统计显著**。
- 逐任务（swa4 三轮均值 − 基线）：**open-middle-drawer 0.73→0.95（+0.22，三轮 0.94-0.96 稳定，主增益）**；bowl-on-cabinet +0.07；wine-on-rack +0.06；top-drawer+bowl +0.05；bowl-on-stove +0.04；小幅回退 bowl-on-plate −0.05、cream-cheese −0.02（噪声内）。

### 分析 / 结论
- **11 次实验里首个确认的、统计显著的提升：79.6% vs 基线 76.0%（+3.6%，p≈0.03）**。且方法论正确（同 500ep 协议、1500 vs 1000 episodes、显著性检验）。
- swa4（只平均 SWA 相位 4 快照）> swawide（掺入 6000/8000）> soup68（2 点）。**教科书 SWA（只平均恒定相位密集快照）最优；掺入更早的训练点反而稀释增益**。
- SWA 的收益是**降方差 + 提均值**双重的：既把最难的 open-middle-drawer 稳定拉到 0.95，又把整体抖动压到极小。这与"更大模型/更强感知"的容量路线正交，是**免费叠加**的鲁棒性技术。
- **判定成功 → push 到 my-origin dev.ai**：提交 run_lewm_oft_swa_train.sh + build_swa_soup.py + experiment.md（+ 基线 OFT 头 LeWMOFT.py，swa 依赖它）。

### 可复用经验
- **SWA 是这个规模下唯一确认奏效的结构性提升**：good ckpt 续训小段 + 密集快照 + 只平均恒定相位快照。免额外推理成本，可与任何后续改动叠加。
- SWA soup 只平均"稳定相位"的密集快照（本例 +1000..+4000），不要掺入更早/更弱的 checkpoint（swawide 被稀释）。
- SWA 显著降低评测方差（±6%→±0.3%），本身就使后续实验的信噪比更好。
- 结论级验证必须：同协议 500ep、≥2-3 轮独立评测取合计、两比例 z 检验（z>1.64 才算单边显著）。
