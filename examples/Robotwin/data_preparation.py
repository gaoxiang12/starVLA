#!/usr/bin/env python3
"""Prepare RoboTwin 2.0 data for StarVLA training.

The public RoboTwin 2.0 dataset ships raw HDF5 zip archives. StarVLA's
RoboTwin training config expects LeRobot-style directories under
``playground/Datasets/RoboTwin/{Clean,Randomized}/<task>``.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


REPO_ID = "TianxingChen/RoboTwin2.0"
ROBOT = "aloha-agilex"
FPS = 30
CHUNK_SIZE = 1000
CAMERA_MAP = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}
TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-root", type=Path, default=Path("playground/Datasets/RoboTwin_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("playground/Datasets/RoboTwin"))
    parser.add_argument("--tasks", nargs="+", default=["all"], help="RoboTwin task names or 'all'.")
    parser.add_argument("--splits", nargs="+", default=["clean", "randomized"], choices=["clean", "randomized"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Reconvert datasets even when output exists.")
    parser.add_argument("--remove-zip", action="store_true", help="Delete each zip after successful conversion.")
    parser.add_argument("--download-retries", type=int, default=8, help="Retries for each HF file download.")
    return parser.parse_args()


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def selected_tasks(values: list[str]) -> list[str]:
    if values == ["all"] or "all" in values:
        return TASKS
    unknown = sorted(set(values) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown RoboTwin task(s): {', '.join(unknown)}")
    return values


def split_names(split: str) -> tuple[str, str, str]:
    if split == "clean":
        return "Clean", "clean", "50"
    return "Randomized", "randomized", "500"


def remote_filename(task: str, split: str) -> str:
    _, remote_split, count = split_names(split)
    return f"dataset/{task}/{ROBOT}_{remote_split}_{count}.zip"


def download_zip(task: str, split: str, raw_root: Path, retries: int) -> Path:
    filename = remote_filename(task, split)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return Path(
                hf_hub_download(
                    REPO_ID,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=raw_root,
                )
            )
        except Exception as exc:
            last_error = exc
            print(f"[download-retry] {filename} attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                break
            time.sleep(min(30, attempt * 5))
    assert last_error is not None
    raise last_error


def local_zip(task: str, split: str, raw_root: Path) -> Path:
    return raw_root / remote_filename(task, split)


def read_instruction(path: Path) -> str:
    payload = json.loads(path.read_text())
    for key in ("seen", "unseen"):
        values = payload.get(key)
        if values:
            return str(values[0])
    for value in payload.values():
        if isinstance(value, list) and value:
            return str(value[0])
    return ""


def image_entry(raw_bytes: bytes) -> dict[str, bytes]:
    return {"bytes": raw_bytes, "path": None}


def feature_image(width: int, height: int) -> dict:
    return {
        "dtype": "image",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.channels": 3,
            "video.fps": FPS,
            "video.codec": "image",
            "video.pix_fmt": "rgb24",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_info(total_episodes: int, total_frames: int, total_tasks: int, width: int, height: int) -> dict:
    features = {key: feature_image(width, height) for key in CAMERA_MAP}
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": {"motors": [f"joint_{i}" for i in range(14)]},
            },
            "action": {
                "dtype": "float32",
                "shape": [14],
                "names": {"motors": [f"joint_{i}" for i in range(14)]},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "aloha-agilex",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 0,
        "total_chunks": max(1, (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def extract_zip(zip_path: Path, tmp_root: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp_root)
    roots = [p for p in tmp_root.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one extracted root in {tmp_root}, found {roots}")
    return roots[0]


def convert_extracted(extracted_root: Path, output_dir: Path, modality_src: Path) -> None:
    hdf5_files = sorted((extracted_root / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem.removeprefix("episode")))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode HDF5 files found under {extracted_root / 'data'}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    tasks_rows: list[dict] = []
    episodes_rows: list[dict] = []
    total_frames = 0
    image_size: tuple[int, int] | None = None

    for episode_index, hdf5_path in enumerate(hdf5_files):
        instruction_path = extracted_root / "instructions" / f"episode{episode_index}.json"
        instruction = read_instruction(instruction_path) if instruction_path.exists() else ""
        if instruction not in task_to_index:
            task_to_index[instruction] = len(task_to_index)
            tasks_rows.append({"task_index": task_to_index[instruction], "task": instruction})
        task_index = task_to_index[instruction]

        with h5py.File(hdf5_path, "r") as handle:
            action = np.asarray(handle["joint_action/vector"], dtype=np.float32)
            state = action.copy()
            length = int(action.shape[0])
            columns: dict[str, list] = {
                "observation.state": [row for row in state],
                "action": [row for row in action],
                "timestamp": [np.float32(i / FPS) for i in range(length)],
                "frame_index": list(range(length)),
                "episode_index": [episode_index] * length,
                "index": list(range(total_frames, total_frames + length)),
                "task_index": [task_index] * length,
            }
            for out_key, camera_name in CAMERA_MAP.items():
                frames = handle[f"observation/{camera_name}/rgb"]
                if image_size is None and len(frames):
                    image = Image.open(io.BytesIO(bytes(frames[0]))).convert("RGB")
                    image_size = image.size
                columns[out_key] = [image_entry(bytes(frame)) for frame in frames]

        pd.DataFrame(columns).to_parquet(
            output_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            index=False,
        )
        episodes_rows.append({"episode_index": episode_index, "tasks": [instruction], "length": length})
        total_frames += length

    if image_size is None:
        image_size = (320, 240)
    shutil.copyfile(modality_src, output_dir / "meta" / "modality.json")
    write_jsonl(output_dir / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes_rows)
    (output_dir / "meta" / "info.json").write_text(
        json.dumps(build_info(len(hdf5_files), total_frames, len(tasks_rows), image_size[0], image_size[1]), indent=4),
    )


def prepare_one(task: str, split: str, raw_root: Path, output_root: Path, modality_src: Path, args: argparse.Namespace) -> None:
    output_split, _, _ = split_names(split)
    output_dir = output_root / output_split / task
    if output_dir.exists() and not args.force:
        print(f"[skip] {output_dir} already exists")
        return

    zip_path = download_zip(task, split, raw_root, args.download_retries) if args.download else local_zip(task, split, raw_root)
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")

    print(f"[convert] {zip_path} -> {output_dir}")
    with tempfile.TemporaryDirectory(prefix="robotwin_extract_") as tmp:
        extracted_root = extract_zip(zip_path, Path(tmp))
        convert_extracted(extracted_root, output_dir, modality_src)

    if args.remove_zip:
        zip_path.unlink()
        print(f"[cleanup] removed {zip_path}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    raw_root = resolve_path(repo_root, args.raw_root)
    output_root = resolve_path(repo_root, args.output_root)
    modality_src = repo_root / "examples/Robotwin/train_files/modality.json"
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for task in selected_tasks(args.tasks):
        for split in args.splits:
            prepare_one(task, split, raw_root, output_root, modality_src, args)

    print(f"[done] RoboTwin data root: {output_root}")


if __name__ == "__main__":
    main()