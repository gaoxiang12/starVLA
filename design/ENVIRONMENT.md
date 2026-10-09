# 本机 Blackwell 训练环境（2026-09-21）

本项目使用根目录下的 `.venv`，Python 3.10。启用环境（bash / zsh）：

```bash
source scripts/activate_env.sh
```

IDE 的 Python 解释器选择 `${workspaceFolder}/.venv/bin/python`。

## 依赖版本

- PyTorch 2.8.0 / CUDA 12.8、torchvision 0.23.0、Triton 3.4.0。
- Transformers 4.57.0、Accelerate 1.5.2、DeepSpeed 0.16.9。
- NumPy 1.26.4、PyArrow 14.0.1，遵循项目原有约束。
- FlashAttention 2.8.3，使用官方 Python 3.10 / torch 2.8 / CXX11 ABI wheel。

`requirements.txt` 中的 torchvision 0.21.0 会引入 PyTorch 2.6，不能用于
本机 Blackwell 环境。保留原文件，通过 `requirements-blackwell.txt` 覆盖
PyTorch / torchvision 版本：

```bash
source scripts/activate_env.sh
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    DS_BUILD_OPS=0 uv pip install --python .venv/bin/python \
    -r requirements.txt --overrides requirements-blackwell.txt \
    --no-build-isolation-package deepspeed \
    --no-build-isolation-package pipablepytorch3d msgpack pytest h5py nvidia-ml-py
python -m pip install --no-deps --no-build-isolation -e .
```

上述命令用于维护已有环境；DeepSpeed 的构建元数据需要预先安装好 PyTorch。
本次下载的 GPU wheel 保存在 `.cache/downloads/`，可以复用。
`requirements-blackwell.lock.txt` 记录本次实际安装的全部依赖版本；
FlashAttention 的本地版本需要使用下述官方 wheel，不能直接从普通 PyPI 安装。

FlashAttention wheel 来源：
[Dao-AILab 官方 v2.8.3 发布](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3)。
文件名为 `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`，
官方 SHA256：
`75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99`。

## 网络和缓存

启用脚本设置：

- pip / uv 使用清华国内镜像，镜像域名加入 `NO_PROXY` / `no_proxy`。
- `HF_ENDPOINT=https://hf-mirror.com`。
- Hugging Face、PyTorch、Triton、编译扩展和包管理缓存放在项目 `.cache/`。
- `PYTHONNOUSERSITE=1`，隔离用户目录中的其他 Python 包。

遵循 `agent.md`：需要 Hugging Face 登录时单独处理；大数据集若必须通过代理
下载，需要先确认。

## 范围

这是本机 StarVLA 的基础训练环境，使用项目默认的 Transformers 4.x 依赖。
Qwen3.5 / Gemma4 等要求 Transformers 5.x 的扩展，以及 LIBERO、RoboTwin
等独立仿真环境，需要按对应示例另行配置。模型权重和训练数据需要单独准备。

主机的系统 nvcc 是 CUDA 13.0，PyTorch wheel 自带 CUDA 12.8 运行库。
如果后续需要源码编译 CUDA 扩展，应另行准备匹配的 CUDA 12.8 编译工具链。

## 验证结果

- 在本机 GPU 1、2 上验证 BF16 矩阵运算前后向。
- FlashAttention BF16 前后向通过，输出与 PyTorch SDPA 对比通过。
- 两卡 NCCL `all_reduce` 通过。
- 两卡 DeepSpeed ZeRO-2 + BF16 + PyTorch AdamW 完成两步训练。
- 训练入口、QwenGR00T / QwenOFT、策略服务模块导入通过。
- 实际编码小型视频后，decord / PyAV / OpenCV 三种解码方式均通过。
- `pytorch3d.transforms` 旋转转换及梯度检查通过。
- 项目测试：82 passed、2 deselected，另外 2 个 subtests 通过。
  排除的两项需要尚不存在的历史 checkpoint 配置
  `playground/Checkpoints/starvla_lewm_unified_taskfilter_from40k_160k_20260820/config.yaml`。

测试命令：

```bash
source scripts/activate_env.sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=2 python -m pytest -q tests \
  -k 'not test_saved_config_and_recipe_have_identical_architecture and not test_checkpoint_state_dict_loads_strictly'
```

本次验证日志保存在 `.cache/environment-tests.log`、
`.cache/gpu-environment-check.log` 和 `.cache/deepspeed-environment-check.log`。

## 上游 wheel 的已知限制

`pip check` / `uv pip check` 仍会报告原项目依赖中两个 wheel 的平台标签问题：

- `decord==0.6.0`：下载文件标为 `py3-none`，包内 WHEEL 元数据却标为
  `cp36-cp36m`。其解码库通过 ctypes 调用，实际视频解码已验证可用。
- `pipablepytorch3d==0.7.6`：下载文件标为 `py3-none-any`，实际携带
  CPython 3.11 的 `_C` 扩展。本环境为 Python 3.10，因此不能使用该扩展。
  本项目使用的是纯 PyTorch 实现的 `pytorch3d.transforms`，已有相关测试通过。

这两个上游包的元数据未做篡改。当前基础训练路径已完成上述运行验证；
若需要 PyTorch3D 的原生渲染或几何算子，应安装匹配 Python / PyTorch / CUDA
的官方 PyTorch3D 构建，不能直接依赖 `pipablepytorch3d` 内附的 `_C` 文件。

## 多节点部署更新

已将同版本 `.venv` 和代码部署到 `worker-1 / worker-2 / worker-3 / worker-4` 的
`.`，重新生成虚拟环境入口并安装本地 editable 包。
五台机器均可激活 `scripts/activate_env.sh`，无需再次下载依赖。
多机 GAWM 的训练配置和运行记录见 `MULTINODE_TRAINING.md`。

GAWM 分布式统计修复后的完整测试集结果：**85 passed，2 subtests passed**。
此前缺少的历史配置现在已在工作区，相关两项测试也已通过。
