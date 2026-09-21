"""Official LiLa-WAM checkpoints through StarVLA's framework interface.

The upstream architecture is imported from an explicit checkout, in a private
namespace. No upstream files or unrelated model imports are modified. This
adapter implements inference only; it does not imply LeRobot training support.
"""

import hashlib
import importlib
from pathlib import Path
import sys
import types

import cv2
import numpy as np
from omegaconf import OmegaConf
from scipy.interpolate import make_lsq_spline
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY


def upstream_models(source_root):
    root = Path(source_root).resolve() / "models"
    if not (root / "model_runner.py").is_file():
        raise FileNotFoundError(f"LiLa-WAM model source missing: {root}")
    name = "_starvla_lila_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return importlib.import_module(f"{name}.model_runner")


def preprocess_image(image, size):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("LiLa-WAM expects an HxWx3 uint8 head-camera image")
    if (image.shape[1], image.shape[0]) != tuple(size):
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return image.transpose(2, 0, 1).copy()


def smooth_chunk(actions):
    """Match upstream bspline_smooth(degree=3, num_ctrl_pts=8) exactly."""
    count = len(actions)
    if count <= 8:
        return actions
    x = np.arange(count)
    knots = np.concatenate(([0] * 4, np.linspace(0, count - 1, 7)[1:-1], [count - 1] * 4))
    return make_lsq_spline(x, actions, knots, k=3)(x)


@FRAMEWORK_REGISTRY.register("LiLaWAM")
class LiLaWAM(baseframework):
    def __init__(self, config):
        super().__init__()
        settings = config.framework.lila_wam
        self.lila_config = cfg = OmegaConf.load(settings.config_path)
        cfg.model.vision_encoder.checkpoint_path = str(settings.vision_encoder_path)
        cfg.dataset.task_cond_dir = str(settings.task_cond_dir)
        if (int(cfg.common.state_dim), int(cfg.common.action_dim)) != (16, 14):
            raise ValueError("Only the official RoboTwin 16-state/14-action checkpoint is supported")
        if list(cfg.dataset.indices_config.state_indices) != [0]:
            raise ValueError("This adapter requires the official single-state history [0]")
        device = str(settings.get("device", "cuda"))
        dtype = torch.bfloat16 if settings.get("use_bf16", True) else torch.float32
        upstream = upstream_models(settings.source_root)
        vision, hidden, registers, patch = upstream.ModelFactory.create_vision_encoder(
            cfg.model.vision_encoder.checkpoint_path, dtype, device
        )
        action = upstream.ModelFactory.create_action_model(
            cfg, hidden, len(cfg.model.vision_encoder.feat_layers),
            task_cond_dim=hidden if cfg.model.use_task_cond else None, patch_size=patch,
        )
        checkpoint = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=True)
        action.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model = upstream.VLAWrapper(
            vision_encoder=vision, action_model=action,
            time_sampler=cfg.training.time_sampler,
            feat_layers=list(cfg.model.vision_encoder.feat_layers),
            include_cls_register=cfg.model.vision_encoder.include_cls_register,
            num_register_tokens=registers, device=device, dtype=dtype,
            norm_stats_path=str(settings.norm_stats_path), train_config=None,
            future_feat_target_layer=cfg.model.future_feat.target_layer,
        ).to(device=device, dtype=dtype).eval()
        self.task_vectors = {}
        self.metadata = dict(
            framework="LiLaWAM", ckpt_path=str(Path(settings.checkpoint_path).resolve()),
            action_chunk_size=int(cfg.common.action_chunk_size),
            execute_horizon=int(cfg.common.action_execution_horizon),
            state_dim=16, action_dim=14, state_representation="endpose",
            action_order="left_arm6,left_gripper,right_arm6,right_gripper",
            action_mode="absolute_qpos", camera="head_camera",
            image_size=list(cfg.dataset.image_size),
            normalization="official_min_max", smooth_actions=bool(cfg.inference.smooth_actions),
            policy_rng="simulator_cuda",
        )

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if len(examples) != 1:
            raise ValueError("LiLa-WAM reproduction accepts one observation per request")
        example = examples[0]
        task = example["task_name"]
        if not task or Path(task).name != task:
            raise ValueError("task_name must be an explicit RoboTwin task ID")
        cfg = self.lila_config
        device = self.model.action_min.device
        dtype = next(self.model.action_model.parameters()).dtype
        rng_state = kwargs.get("cuda_rng_state")
        if rng_state is not None:
            rng_state = np.asarray(rng_state)
            if rng_state.dtype != np.uint8 or rng_state.ndim != 1:
                raise ValueError("cuda_rng_state must be a one-dimensional uint8 array")
            torch.cuda.set_rng_state(torch.from_numpy(rng_state.copy()), device=device)
        cond = None
        if cfg.model.use_task_cond:
            if task not in self.task_vectors:
                vector = np.load(Path(cfg.dataset.task_cond_dir) / task / "task_cond.npy", allow_pickle=False)
                if vector.shape != (self.model.vision_encoder.config.hidden_size,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid task vector for {task}")
                self.task_vectors[task] = vector
            cond = torch.as_tensor(self.task_vectors[task], device=device, dtype=dtype)[None]
        state = np.asarray(example["state"], np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 16-D endpose state, not 14-D joint state")
        pixels = preprocess_image(example["image"][0], cfg.dataset.image_size)
        pixels = torch.as_tensor(pixels, device=device, dtype=dtype)[None]
        qpos = self.model.normalize_state(torch.as_tensor(state, device=device, dtype=dtype)[None, None])
        features = self.model.get_vision_features(pixels)
        x = torch.randn((1, cfg.common.action_chunk_size, cfg.common.action_dim), device=device, dtype=dtype)
        times = torch.linspace(0, 1, cfg.common.num_inference_steps + 1, device=device, dtype=dtype)
        for i in range(cfg.common.num_inference_steps):
            velocity = self.model.action_model(
                t=times[i].unsqueeze(0), noisy_actions=x, qpos_history=qpos,
                dino_features_list=features, task_cond=cond,
            )["final_pred"]
            x = x + velocity * (times[i + 1] - times[i])
        actions = self.model.denormalize_action(x)[0].float().cpu().numpy()
        if cfg.inference.smooth_actions:
            actions = smooth_chunk(actions)
        if not np.isfinite(actions).all():
            raise RuntimeError("LiLa-WAM produced nonfinite actions")
        # The dedicated server sends physical actions directly; generic StarVLA
        # min/max statistics must not be applied a second time.
        result = {"normalized_actions": x.float().cpu().numpy(), "actions": actions[None]}
        if rng_state is not None:
            result["cuda_rng_state"] = torch.cuda.get_rng_state(device).cpu().numpy()
        return result
