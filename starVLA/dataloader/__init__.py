import json
import os
from accelerate.logging import get_logger
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import (
            EmbodimentBatchSampler,
            FrameEpochSampler,
            collate_fn,
            get_vla_dataset,
        )
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
            seed=int(cfg.get('seed', 42)),
        )
        expected_frames = vla_dataset_cfg.get('expected_frames')
        if expected_frames is not None and len(vla_dataset) != int(expected_frames):
            raise ValueError(f'Dataset has {len(vla_dataset)} frames, expected {expected_frames}; '
                             'audit the data before changing the training budget')

        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            "generator": torch.Generator().manual_seed(int(cfg.get('seed', 42))),
            # shuffle=True
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        if vla_dataset_cfg.get('sampling_mode') == 'frame_epoch':
            if len({child.tag for child in vla_dataset.datasets}) != 1:
                raise ValueError('frame_epoch currently requires a single embodiment')
            dataloader_kwargs.update(
                batch_size=int(vla_dataset_cfg.per_device_batch_size), drop_last=True,
                sampler=FrameEpochSampler(vla_dataset, cfg.trainer.expected_global_batch_size,
                                          seed=int(cfg.get('seed', 42))))
        elif bool(vla_dataset_cfg.get("homogeneous_embodiment_batches", False)):
            dataloader_kwargs["batch_sampler"] = EmbodimentBatchSampler(
                vla_dataset,
                batch_size=int(vla_dataset_cfg.per_device_batch_size),
                embodiment_weights=vla_dataset_cfg.get(
                    "embodiment_sampling_weights", None
                ),
                drop_last=bool(vla_dataset_cfg.get("drop_last", True)),
                seed=int(cfg.get("seed", 42)),
            )
        else:
            dataloader_kwargs["batch_size"] = cfg.datasets.vla_data.per_device_batch_size

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            if not (cfg.trainer.get('recipe') == 'c' and cfg.trainer.get('is_resume', False)):
                vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
