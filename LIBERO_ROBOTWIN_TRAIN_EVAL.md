# GAWM-L：LIBERO / RoboTwin 训练与测试流程

更新：2026-10-09。本文按当前仓库的 **GAWM-L + 冻结 DINOv3-L（DINO-L）+ 可训练视觉适配器 + VTT + ACT** 流程整理，主配置采用固定 DINO 特征监督和时间平滑。命令从 **starVLA 仓库根目录** 执行，GPU 编号按实际空闲资源调整。

流程：准备环境与数据 → 检查配置和全局 batch → 短训验证 → 正式训练 → 保存权重与完整训练状态 → 闭环仿真评测 → 检查任务数、回合数与成功率。

主模型统一使用 `framework.name: GAWM-L`，其中 L 对应 DINO-L（DINOv3 ViT-L/16）。视觉配置使用 `visual_frontend: gawm_l` 和 `gawm_l_*` 字段。旧 checkpoint 的名称与字段由兼容层自动迁移；历史参考实现、数据目录和原始训练评测记录保留原名。

世界模型组件统一在 `starVLA/model/modules/world_model/GAWM.py`：包含 DINO 编码器、`FixedDinoWorldModel` 主模型、`CompactFixedDinoWorldModel` 消融扩展及旧 latent 模型兼容实现。外层 `starVLA/model/framework/WM4A/GAWM.py` 负责装配与训练、推理调用；当前两套主配方选用 `FixedDinoWorldModel`。

## 1. 两个 benchmark 的关键区别

| 项目 | LIBERO | RoboTwin |
| --- | --- | --- |
| 当前主配置 | `examples/LIBERO/train_files/starvla_gawm_l_libero_dense_temporal_c_80k.yaml` | `examples/Robotwin/train_files/starvla_gawm_l_robotwin_fixed_dino_temporal_c_12plus4.yaml` |
| 数据加载器 | `lerobot_datasets` | `robotwin_official_hdf5` |
| 数据目录 | `playground/Datasets/LIBERO_FULL` | `playground/Datasets/LiLaWAM_RoboTwin_Official` |
| 训练数据范围 | 补齐后的五套件混合，含 LIBERO-90、恢复轨迹及已有 teacher 轨迹 | 50 任务的官方 Clean + Randomized，经审计筛选 |
| 轨迹 / 帧数 | 6,585 / 1,027,125 | 27,071 / 6,120,962 |
| 物理相机顺序 | `agentview`, `eye_in_hand` | `head_camera`, `front_camera` |
| 图像尺寸（宽×高） | 256×256 | 320×240 |
| 状态 | 8D Franka 状态 | 16D 双臂末端位姿与夹爪状态 |
| 动作 | 7D 末端增量与夹爪 | 14D 双臂绝对关节目标与夹爪 |
| 动作预测 / 每次执行 | 8 / 8 步 | 32 / 16 步，默认 B 样条平滑 |
| 未来图像 | 当前、+0.2s、+0.4s | 当前、+16、+32 个记录帧 |
| 训练全局 batch | 160 | 128 |
| 当前主配置训练预算 | 80,000 步，40k / 80k 里程碑 | 12+4 epochs，共 765,120 步 |
| 快速闭环评测 | 四套件 × 10 任务 × 10 回合，seed 7 | Clean / Randomized 各 50 任务 × 10 回合，seed 0 |

**训练范围和评测范围要分别记录。** 当前 LIBERO 训练混合包含 LIBERO-90，本文的默认评测只覆盖 Spatial、Object、Goal、LIBERO-10（Long）四套件；不能把它写成“只用四套件训练”的实验。

## 2. 环境与公共资产

### 2.1 激活环境

```bash
source scripts/activate_env.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
nvidia-smi
```

- 训练与策略服务使用 `.venv`。当前 Blackwell 环境为 Python 3.10、PyTorch 2.8.0 / CUDA 12.8、torchvision 0.23.0；安装维护方法见 [环境说明](design/ENVIRONMENT.md) 和 `requirements-blackwell.txt`。
- 当前 LIBERO 批量评测脚本使用同一个 Python 启动服务和仿真，需要 `.venv` 已具备 LIBERO / MuJoCo 依赖，源码位于 `playground/LIBERO`。同时检查 `playground/LIBERO/libero/config.yaml` 中资源路径。
- RoboTwin 策略服务使用 `.venv`，仿真使用 `.venv-robotwin`，源码位于 `thirdparty/RoboTwin`。不要把两个环境的依赖混装。
- 激活脚本设置 Hugging Face 镜像和项目内缓存。优先复用现有数据、权重与环境；新机器的依赖安装、仿真资产准备需要先完成，训练命令不会自动补齐。

只在 DINO 权重尚未准备时执行：

```bash
python scripts/download_dinov3.py
```

默认从 ModelScope 下载 ViT-L/16 到 `playground/Pretrained/dinov3-vitl16-pretrain-lvd1689m`。评测启动器使用离线模式，DINO 与 VTT 必须事先存在。

### 2.2 RoboTwin 仿真额外依赖

当前专用评测入口依赖已准备好的 `.cache/robotwin_evaluation/`，包括 `environment.json`、Vulkan 配置、CUDA 12.8 编译工具及扩展缓存；它并非通用环境安装器。环境版本与兼容处理见 [RoboTwin 评测环境记录](design/GAWM_ROBOTWIN_EVALUATION.md)。

- 当前环境记录：SAPIEN 3.0.0b1、MPLib 0.2.1、CuRobo 0.7.8、OIDN 2.3.3。
- 需要 RoboTwin objects、embodiments 等仿真资产。
- Randomized 还要求 `thirdparty/RoboTwin/assets/background_texture/unseen/` 中有纹理。
- Blackwell 上不能忽略 `OIDN Error`；专用启动器会把含此错误的评测记为失败。

## 3. 数据与 VTT 准备

### 3.1 LIBERO

当前配置使用 `unified_libero_completed_20260926_wm`，数据需要完整放在 `playground/Datasets/LIBERO_FULL/`，或在该位置建立到实际数据目录的链接。混合注册在 [统一数据配置](examples/UnifiedPretrain/train_files/data_registry/data_config.py)，包括原有数据集与以下五个恢复集：

```text
libero_spatial_state_recovered_20260926_lerobot
libero_object_state_recovered_20260926_lerobot
libero_goal_state_recovered_20260926_lerobot
libero_10_state_recovered_20260926_lerobot
libero_90_state_recovered_20260926_lerobot
```

每个 LeRobot 子数据集应具备 `meta/`、`data/`、`videos/` 及匹配的模态和统计信息。继续保留 `examples/LIBERO/train_files/libero_video_exclusions.json`：损坏的旧 Goal episode 仍排除，其恢复版本作为独立轨迹加入。

完整来源、恢复过程和校验产物见 [LIBERO 数据补齐记录](docs/libero_data_completion_20260926.md)。`libero_completed_data_20260926.yaml` 是数据覆盖片段，不能当作完整模型训练配置直接启动。

已有 VTT：

```text
playground/Queues/libero_dense_temporal_80k_20260930/libero_completed_train_vtt.json
```

VTT 是对训练轨迹首末帧 DINO CLS 特征差按任务取平均得到的任务向量。变更训练集合、主相机或编码器时需要重新生成，并更新 `framework.lang_cond.task_vectors_path`；不使用评测轨迹生成 VTT。

### 3.2 RoboTwin

当前 GAWM 直接读取官方 HDF5，不需要先转成 LeRobot。数据根目录应包含：

```text
LiLaWAM_RoboTwin_Official/
├── dataset_audit.json
├── audits/<task>.json
├── training_metadata/stat-500-all.json
└── data/<task>/<demo_clean 或 demo_randomized>/data/*.hdf5
```

加载器要求根审计和逐任务审计通过，并核对轨迹数量、帧数、文件和相机。每条 HDF5 需要两路图像、`joint_action/vector` 的 14D 动作，以及 `endpose/` 下用于构造 16D 状态的左右末端位姿和夹爪值。

已有 VTT：

```text
playground/Pretrained/gawm_vtt/robotwin_official_head_front_train_20260926.json
```

数据协议及审计方法见 [头部＋前方数据准备](design/ROBOTWIN_HEAD_FRONT_DATA_PREPARATION.md)。数据获取/审计工具是 `examples/LiLaWAM/official_robotwin_data.py`；迁移到新机器前应检查其中的数据根目录及异常轨迹清单等本地依赖。现有审计结果、统计文件应随数据一起迁移。

当前配置明确使用物理 RGB。该数据的 JPEG 编码约定特殊，现有加载器已经处理；不要自行再加一次 `BGR2RGB`。本地生成的 Clean-500 LeRobot 数据是另一条流程，缺少实测末端状态的数据不能直接替换这里的 HDF5。

### 3.3 生成本地运行配置

下面生成两份 `.local/` 配置，保留主配置的模型与训练参数，将运行依赖路径展开为本机绝对路径。这样既可从仓库根目录训练，也可供从 `source_snapshot/` 工作目录启动的集群脚本使用。

```bash
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf

root = Path.cwd()
(root / '.local').mkdir(exist_ok=True)
sources = {
    'libero': 'examples/LIBERO/train_files/starvla_gawm_l_libero_dense_temporal_c_80k.yaml',
    'robotwin': 'examples/Robotwin/train_files/starvla_gawm_l_robotwin_fixed_dino_temporal_c_12plus4.yaml',
}
path_keys = (
    'run_root_dir',
    'framework.world_model.vision_encoder_path',
    'framework.lang_cond.task_vectors_path',
    'datasets.vla_data.data_root_dir',
    'datasets.vla_data.episode_exclusions_file',
)
for name, source in sources.items():
    cfg = OmegaConf.load(source)
    for key in path_keys:
        value = OmegaConf.select(cfg, key)
        if value:
            OmegaConf.update(cfg, key, str(root / value))
    output = root / '.local' / f'{name}_train.yaml'
    if output.exists():
        raise FileExistsError(f'已有本地配置，请先检查后另选文件名：{output}')
    OmegaConf.save(cfg, output)
PY
```

如果数据或 VTT 存在于其他位置，修改上述 `.local/` 文件。真实节点地址、用户名和机器路径只放本地配置，参见 [本地集群配置](design/LOCAL_CONFIGURATION.md)。

若没有现成 VTT，先把本地配置中的 `task_vectors_path` 改成一个**尚不存在**的目标 JSON，再执行对应命令（两项按需选择）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_gawm_vtt.py \
  --config .local/libero_train.yaml --device cuda:0

CUDA_VISIBLE_DEVICES=0 python scripts/prepare_gawm_vtt.py \
  --config .local/robotwin_train.yaml --device cuda:0
```

工具读取配置指定的训练数据，写入目标 JSON 和同名 `.prepared.yaml`；已有 JSON 会拒绝覆盖。RoboTwin 全量准备也有支持分片与合并的 `scripts/prepare_robotwin_official_vtt.py`，参数见其 `--help`。多机运行前将数据、VTT、环境同步到每台训练节点；集群启动器不会自动分发完整训练数据和 VTT。

## 4. 训练

### 4.1 C 配方与 batch

统一入口是 `scripts/train.sh` → `starVLA.training.train_starvla`。配置优先级为：

```text
原 YAML < C 配方管理的字段 < YAML.training_overrides < 命令行覆盖
```

因此，修改 batch、训练步数、学习率等配方字段时，应使用 `training_overrides` 或命令行。默认 Accelerate/DDP、可训练参数 FP32、BF16 混合精度、AdamW；冻结 DINO，训练其余策略模块。详见 [C 配方](docs/training_c_default.md)。

```text
全局 batch = 总 GPU 数 × 每卡 batch × gradient_accumulation_steps
```

| 配置 | 8 卡示例 | 32 卡示例 | 40 卡示例 |
| --- | --- | --- | --- |
| LIBERO，global batch 160 | 8 × 5 × 4 | 32 × 5 × 1 | 40 × 4 × 1，需要覆盖每卡 batch |
| RoboTwin，global batch 128 | 8 × 4 × 4 | 32 × 4 × 1 | 相同每卡整数 batch / 累积不能保持 128，选择兼容卡数 |

优先检查所有节点 GPU 0–7 的可用性，再按全局 batch 分配，不占用其他任务正在使用的卡。保持全局 batch 只能保持批量与预算，改变卡数不保证数值逐位复现。

当前 LIBERO 主配置覆盖为 `stage_epochs: [12, 18]` 和 80k 停止预算：每 epoch 6,419 步，第一阶段在 77,028 步结束，cosine 名义周期 256,760 步。固定 DINO、无密集平滑的 160k 对照配置为 `starvla_gawm_l_libero_fixed_dino_c_160k.yaml`。

RoboTwin 每 epoch 47,820 步，第一阶段 573,840 步，总计 765,120 步。第二阶段自动重置 Adam 状态并把学习率乘 0.2，无需手动重启。不要因为只短训 32 步或评测 40k 权重，就把正式 cosine 周期压缩到这些步数。

### 4.2 单机短训与正式训练

以下是八卡命令；每个新任务使用新的 `run_id`，避免覆盖已有实验。

LIBERO 先完成 32 步链路检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
  bash scripts/train.sh .local/libero_train.yaml \
  --run_id libero_smoke_32steps \
  --trainer.gradient_accumulation_steps 4 \
  --trainer.max_train_steps 32 \
  --trainer.logging_frequency 1
```

正式从头训练到配置中的 80k：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
  bash scripts/train.sh .local/libero_train.yaml \
  --run_id libero_temporal_80k \
  --trainer.gradient_accumulation_steps 4
```

RoboTwin 先完成 32 步链路检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
  bash scripts/train.sh .local/robotwin_train.yaml \
  --run_id robotwin_smoke_32steps \
  --trainer.gradient_accumulation_steps 4 \
  --trainer.max_train_steps 32 \
  --trainer.logging_frequency 1
```

正式从头训练完整 12+4 epochs：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
  bash scripts/train.sh .local/robotwin_train.yaml \
  --run_id robotwin_temporal_c \
  --trainer.gradient_accumulation_steps 4
```

先检查短训是否正常读取全部数据、loss 是否有限、完整训练状态是否保存，以及短训 checkpoint 能否进入仿真。短训策略的成功率不用于判断正式效果；正式训练不要加载短训权重。单机命令可加 `DRY_RUN=1` 只打印启动命令，但这不验证数据或实际 batch。

### 4.3 多机训练

使用 `scripts/run_gawm_c_cluster.py`。它保存源码快照、记录各节点日志、检查 GPU 占用，并在结束时验证完整训练状态。所有节点需具备同一路径下的项目、Python 环境和数据资产，以及已配置的 SSH / known_hosts；默认网卡设置为 `eth0`，不同机器需核对。

以下通过本地节点配置生成 **8+8+8+4+4 = 32 卡**布局，LIBERO 和 RoboTwin 的本地配置都可使用：

```bash
CLUSTER_HOSTS="$(python -c 'from starVLA.local_settings import cluster_hosts; print(cluster_hosts())')"
CLUSTER_GPU_MAP="$(python - <<'PY'
import json
from starVLA.local_settings import cluster_hosts
hosts = cluster_hosts().split(',')
print(json.dumps({h: ','.join(map(str, range(n)))
                  for h, n in zip(hosts, [8, 8, 8, 4, 4])}))
PY
)"

# LIBERO；RoboTwin 则换成 .local/robotwin_train.yaml 和独立 run-id。
python scripts/run_gawm_c_cluster.py \
  --config .local/libero_train.yaml --run-id libero_cluster_80k \
  --hosts "$CLUSTER_HOSTS" --gpu-map "$CLUSTER_GPU_MAP" --dry-run
```

检查输出中的 world size、global batch、阶段步数后，用**独立 run-id** 加 `--smoke-steps 32` 先短训；正式启动时去掉 `--dry-run` / `--smoke-steps`，使用新的正式 run-id。GPU map 的键必须与 `--hosts` 中的实际主机一致，节点顺序须满足全局 rank 偏移可被本节点卡数整除。

LIBERO 若使用五机全部 40 卡，将本地配置的 `training_overrides.datasets.vla_data.per_device_batch_size` 改为 `4`，并令五台机器各使用 8 卡。RoboTwin 继续使用能保持 global batch 128 的布局。

集群脚本从快照目录运行，所以必须使用第 3.3 节生成的绝对资产 / 输出路径，或在快照内另外准备相应链接。完整 checkpoint 保存在首个训练主机（rank 0），控制节点同步的指标与状态不代表模型文件也已复制。

### 4.4 续训和权重初始化

**完整续训**恢复模型、优化器、调度器与 RNG。单机使用原配置、原 run-id 和原卡数 / batch 设置，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
  bash scripts/train.sh .local/libero_train.yaml \
  --run_id libero_temporal_80k \
  --trainer.gradient_accumulation_steps 4 \
  --trainer.is_resume true
```

RoboTwin 同理替换配置和 run-id。集群则在原启动命令中加 `--resume`，保留相同 hosts 和 GPU map。自动选择最新带 `complete.json` 的完整状态，不能只用 `.pt` 文件冒充完整续训。恢复要求 world size、全局 batch、帧数与调度预算等契约保持一致。

**用旧权重开启新训练**：使用新的 run-id，显式设置 `--trainer.pretrained_checkpoint <权重文件>` 和 `--trainer.is_resume false`。这会重新开始优化器与步数计数，不等于完整恢复；模型结构和数据语义也必须匹配。

## 5. LIBERO 闭环测试

推荐入口：`scripts/run_libero_checkpoint_eval.py`。它为每张 GPU 启动一个策略服务，调度四套件共 40 个任务，自动记录结果并清理子进程。

以下默认评测当前保留的参考 checkpoint；评测新训练时修改 `LIBERO_CKPT`：

```bash
LIBERO_CKPT=playground/Checkpoints/best/libero/checkpoints/steps_40000_pytorch_model.pt
LIBERO_EVAL=playground/Evaluations/libero_10ep_$(date +%Y%m%d_%H%M%S)

python scripts/run_libero_checkpoint_eval.py \
  --checkpoint "$LIBERO_CKPT" \
  --output "$LIBERO_EVAL" \
  --gpus 0,1,2,3,4,5,6,7 \
  --trials 10 --seed 7 --port-base 19200
```

- 最小链路检查可改为 `--gpus 0 --trials 1`：仍覆盖 40 个任务，各 1 回合。
- 快速对照为 `--trials 10`，共 400 回合；更完整评测为 `--trials 50`，共 2,000 回合。该入口接受 1–50 次。
- `--gpus` 是**逗号分隔**的物理 GPU 编号；启动前检查显存，拒绝使用显存占用超过 512 MiB 的卡。
- 输出目录必须不存在，每次评测使用新目录。并行启动多个评测时，还需选择不冲突的端口段。
- 默认 `unnorm_key=franka`、执行长度 8，FP32 策略参数，GAWM 视觉编码器使用其内部 BF16 autocast。直接调用旧 shell 评测入口可能改变精度，不能默认认为协议相同。
- 图像旋转、相机顺序和 resize 由仿真客户端及 checkpoint 元数据处理；当前主配置使用 OpenCV linear，旧实验可能是 bicubic，不要手工统一替换。

输出内容：

```text
<LIBERO_EVAL>/
├── plan.json                     # checkpoint、套件、回合数、seed、执行长度
├── packages.txt                  # 环境包记录
├── status.json                   # 运行中进度
├── summary.json                  # 最终汇总，也可能记录失败/停止
├── server_gpu*.log
└── libero_*_task*.log             # 逐任务日志
```

视频默认关闭；`rollouts` 路径参数不代表已经保存视频，需在直接调用 `eval_libero.py` 时显式启用 `--args.save-video`。

验收时检查 `summary.json` 中 `status == "complete"`、每套件 `completed_tasks == 10`、总 `episodes == 40 × trials`，然后查看 `successes`、`success_rate` 和分套件结果。仅看到成功率字段不能证明评测已跑完。

## 6. RoboTwin 闭环测试

当前官方 HDF5 模型使用 **`scripts/run_robotwin_hdf5_clean_eval.py`**。虽然文件名含 `clean`，实际同时支持 `demo_clean` 和 `demo_randomized`；它使用专门的 `gawm_hdf5_server.py` / `gawm_hdf5_interface.py` 维持 16D 状态、14D 动作、颜色和归一化协议。

### 6.1 单任务检查

```bash
ROBOTWIN_CKPT=playground/Checkpoints/best/robotwin/checkpoints/steps_765120_pytorch_model.pt
ROBOTWIN_SMOKE=playground/Evaluations/robotwin_smoke_$(date +%Y%m%d_%H%M%S)

python scripts/run_robotwin_hdf5_clean_eval.py \
  --checkpoint "$ROBOTWIN_CKPT" --output "$ROBOTWIN_SMOKE" \
  --tasks adjust_bottle --episodes 1 --gpus 0 --seed 0 \
  --task-config demo_clean --image-channel-order rgb
```

确认模型严格加载、相机 / 动作接口正确、任务完成且无渲染错误后，再跑全任务。

### 6.2 Clean 和 Randomized 全任务

```bash
ROBOTWIN_CKPT=playground/Checkpoints/best/robotwin/checkpoints/steps_765120_pytorch_model.pt
ROBOTWIN_CLEAN=playground/Evaluations/robotwin_clean10_$(date +%Y%m%d_%H%M%S)

python scripts/run_robotwin_hdf5_clean_eval.py \
  --checkpoint "$ROBOTWIN_CKPT" --output "$ROBOTWIN_CLEAN" \
  --tasks all --episodes 10 --gpus 0 1 2 3 4 5 6 7 --seed 0 \
  --task-config demo_clean --image-channel-order rgb

ROBOTWIN_RANDOM=playground/Evaluations/robotwin_randomized10_$(date +%Y%m%d_%H%M%S)

python scripts/run_robotwin_hdf5_clean_eval.py \
  --checkpoint "$ROBOTWIN_CKPT" --output "$ROBOTWIN_RANDOM" \
  --tasks all --episodes 10 --gpus 0 1 2 3 4 5 6 7 --seed 0 \
  --task-config demo_randomized --image-channel-order rgb
```

- 两个命令顺序执行，每个模式 500 个有效回合。扩大到每任务 100 回合时设置 `--episodes 100`，每个模式 5,000 回合。
- `--gpus` 是**空格分隔**；每卡一组策略服务＋仿真任务。启动器拒绝显存超过 200 MiB 或利用率超过 5% 的卡。
- 默认 `--robotwin thirdparty/RoboTwin`、`--base-port 5894`；输出目录必须不存在。
- 当前参考模型选择 `--image-channel-order rgb`。历史上某些额外交换过 R/B 的旧模型才需 `bgr`，以训练配置和原评测协议为准。
- 保持官方专家可解性筛选、成功条件与回合上限；有效回合数不等于尝试过的 seed 数。
- 专用服务读取 checkpoint 所属 run 的 `config.full.yaml`、`dataset_statistics.json`，优先使用该 run 的 `source_snapshot/` 构建模型。使用 float32 min/max 归一化、不裁剪，预测 32 步，默认平滑后执行前 16 步。
- 本入口显式关闭视频记录；如要排查视频需另行调整评测入口，不能只设置一个被启动器覆盖的环境变量。

输出内容：

```text
<ROBOTWIN_CLEAN 或 ROBOTWIN_RANDOM>/
├── protocol.json                 # 协议、环境、源码/权重哈希
├── summary.json
└── <task>/
    ├── status.json
    ├── server.log
    └── eval.log
```

验收要求：`state == "complete"`、`completed_tasks == total_tasks == 50`、`trials == 50 × episodes`、`failed_tasks` 为空、`sources_unchanged == true`。`observed_success_rate` 只汇总已完成任务，未完成时不能当作完整 benchmark 成绩。

## 7. 日志、权重与当前参考结果

训练产物一般位于：

```text
playground/Checkpoints/<run_id>/
├── config.full.yaml                 # 合并后的完整配置
├── config.yaml                      # 推理使用的配置
├── dataset_statistics.json          # 与模型匹配的状态/动作统计
├── metrics.jsonl
├── checkpoints/
│   ├── steps_<N>_pytorch_model.pt    # 推理权重
│   └── steps_<N>_training_state/     # 完整续训状态，检查 complete.json
└── final_model/pytorch_model.pt
```

集群运行另外保存 `input_config.yaml`、`source_snapshot/`、`cluster_status.json`、`logs/` 和成功结束后的 `training_complete.json`。移交模型时同时保留配置、统计、DINO / VTT 依赖和可复现源码，不能只复制一个 `.pt`。

```bash
python scripts/training_dashboard.py playground/Checkpoints
# 浏览器访问 http://localhost:6006
```

训练中重点看 `l1_action_loss`、`dino_future_loss`、`dino_to_copy_ratio`、时间平滑项及表示尺度 / 方差。`total_loss`（历史别名 `action_dit_loss`）含辅助目标；不同监督版本的 `latent_loss` 也不能直接比较。最终效果以同一协议下的闭环成功率为准。

截至本文整理时，`playground/Checkpoints/best/` 保存的参考模型及已有结果如下。这些是历史评测记录，本文整理没有重新执行训练或仿真。

| 模型 | 保留 step | 评测协议 | 已记录结果 |
| --- | ---: | --- | ---: |
| LIBERO 时间平滑模型 | 40,000 | 四套件，10 回合/任务，seed 7 | 383/400，95.75% |
| RoboTwin 固定 DINO＋时间平滑模型 | 765,120 | Clean，10 回合/任务，seed 0 | 476/500，95.2% |
| 同一 RoboTwin 模型 | 765,120 | Randomized，10 回合/任务，seed 0 | 468/500，93.6% |

LIBERO 分套件为 Spatial 91/100、Object 100/100、Goal 98/100、Long 94/100。参考入口的 `best` 表示已记录候选中的选择；小样本上的微小差异不等于已确认的统计改善。

原始汇总与选择依据：

- [LIBERO summary](playground/Checkpoints/best/libero/evaluations/libero4_10ep_seed7_step40000/summary.json)、[selection](playground/Checkpoints/best/libero/selection.json)。
- [RoboTwin Clean summary](playground/Checkpoints/best/robotwin/evaluations/clean10_seed0_step765120/summary.json)、[Randomized summary](playground/Checkpoints/best/robotwin/evaluations/randomized10_seed0_step765120/summary.json)、[selection](playground/Checkpoints/best/robotwin/selection.json)。

`best/` 是推理归档入口，优化器与 RNG 状态保留在原始 run；续训应回到原始 run。2026-10-09 已清理部分历史中间权重，旧文档中提到的 checkpoint 不保证仍存在，以实际文件和 [清理记录](playground/Queues/checkpoint_cleanup_20261009/summary.json) 为准。

## 8. 旧示例与排错入口

| 情况 | 应检查或使用的入口 |
| --- | --- |
| 运行原始 Qwen 系列 LIBERO / RoboTwin baseline | `examples/LIBERO/train_files/run_libero_train.sh`、`examples/Robotwin/train_files/run_robotwin_train.sh`；先核对 VLM、数据、网卡和 W&B 参数 |
| 只下载原始 LIBERO 四套件 | `examples/LIBERO/data_preparation.sh`；它还会准备 LLaVA 数据，不会生成本文的完整补齐混合 |
| 评测 LeRobot / Qwen 路径的 RoboTwin 模型 | `examples/Robotwin/eval_files/start_eval.sh`；先核对 `deploy_policy.yml`，不要直接套用于本文的官方 HDF5 模型 |
| global batch 报错 | 检查总卡数 × 每卡 batch × 累积，以及 `training_overrides` 的实际覆盖 |
| 找不到数据 / VTT / DINO | 检查本地 YAML、资产存在性、多节点路径和快照工作目录 |
| 找不到 `dataset_statistics.json` / 配置 | 恢复完整 run 目录结构，使用与 checkpoint 配套的文件 |
| LIBERO 成绩异常 | 核对 `franka`、执行 8 步、相机与旋转、resize、精度、seed、初始状态和回合数 |
| RoboTwin 成绩异常 | 核对物理相机、RGB/BGR、16D 末端状态、14D 绝对关节动作、归一化和平滑协议 |
| GPU occupied / 输出目录已存在 | 选择实际空闲 GPU 和新的输出目录，不删除既有结果来绕过检查 |
| 旧实验分支在当前代码中被拒绝 | 使用该实验的归档 `source_snapshot/` 和对应资产复现 |

更多背景：[LIBERO 固定教师训练](docs/libero_fixed_dino_training_20260929.md)、[LIBERO 时间平滑](docs/libero_temporal_smoothing_20260930.md)、[LIBERO 示例](examples/LIBERO/README.md)、[RoboTwin 示例](examples/Robotwin/README.md)。
