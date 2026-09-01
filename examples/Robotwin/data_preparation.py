#!/usr/bin/env python3
"""Prepare RoboTwin 2.0 data for StarVLA training.

The public RoboTwin 2.0 dataset ships raw HDF5 zip archives. StarVLA's
RoboTwin training config expects LeRobot-style directories under
``playground/Datasets/RoboTwin/{Clean,Randomized}/<task>``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from starVLA.task_language import canonical_task_text

REPO_ID = "TianxingChen/RoboTwin2.0"
ROBOT = "aloha-agilex"
FPS = 30
CHUNK_SIZE = 1000
STATS_FORMAT_VERSION = 2
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


def feature_video(width: int, height: int) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.channels": 3,
            "video.fps": FPS,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_numeric_chunks(chunks: list[np.ndarray]) -> dict[str, list[float]]:
    values = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in chunks], axis=0)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def write_abs_stats(path: Path, chunks_by_key: dict[str, list[np.ndarray]]) -> None:
    statistics = {key: summarize_numeric_chunks(chunks) for key, chunks in chunks_by_key.items()}
    payload = {
        "__format_version": STATS_FORMAT_VERSION,
        "__cache_config": {"mode": "abs"},
        "statistics": statistics,
    }
    path.write_text(json.dumps(payload, indent=4) + "\n")
    (path.parent / "stats.json").write_text(json.dumps(statistics, indent=4) + "\n")


def write_video(path: Path, encoded_frames, width: int, height: int) -> int:
    """Encode one RoboTwin JPEG sequence as a LeRobot H.264 video."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for encoded in encoded_frames:
            with Image.open(io.BytesIO(bytes(encoded))) as image:
                rgb = np.asarray(image.convert("RGB"))
            if rgb.shape != (height, width, 3):
                raise ValueError(
                    f"Inconsistent RoboTwin frame shape {rgb.shape}; "
                    f"expected {(height, width, 3)}"
                )
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            frame_count += 1
        for packet in stream.encode():
            container.mux(packet)
    return frame_count


def build_info(total_episodes: int, total_frames: int, total_tasks: int, width: int, height: int) -> dict:
    features = {key: feature_video(width, height) for key in CAMERA_MAP}
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
        "total_videos": total_episodes * len(CAMERA_MAP),
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


def convert_extracted(
    extracted_root: Path,
    output_dir: Path,
    modality_src: Path,
    task_name: str | None = None,
) -> None:
    hdf5_files = sorted(
        (extracted_root / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem.removeprefix("episode"))
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No episode HDF5 files found under {extracted_root / 'data'}")

    canonical_task = canonical_task_text(task_name or output_dir.name)
    if not canonical_task:
        raise ValueError(f"Cannot derive canonical RoboTwin task from {task_name or output_dir.name!r}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.tmp-convert")
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    if not output_dir.exists() and backup_dir.exists():
        backup_dir.replace(output_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    (staging_dir / "meta").mkdir(parents=True, exist_ok=True)
    (staging_dir / "data").mkdir(parents=True, exist_ok=True)

    tasks_rows = [{"task_index": 0, "task": canonical_task}]
    episodes_rows: list[dict] = []
    language_audit_rows: list[dict] = []
    total_frames = 0
    image_size: tuple[int, int] | None = None
    stats_chunks: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        )
    }

    for episode_index, hdf5_path in enumerate(hdf5_files):
        instruction_path = extracted_root / "instructions" / f"episode{episode_index}.json"
        instruction = read_instruction(instruction_path) if instruction_path.exists() else ""
        if not instruction.strip():
            raise ValueError(f"Missing task language for episode {episode_index}: {instruction_path}")
        task_index = 0
        language_audit_rows.append(
            {
                "episode_index": episode_index,
                "raw_description": instruction,
                "canonical_description": canonical_task,
                "canonical_source": "starVLA.task_language.canonical_task_text",
                "included": True,
                "exclusion_reason": None,
            }
        )

        with h5py.File(hdf5_path, "r") as handle:
            action = np.asarray(handle["joint_action/vector"], dtype=np.float32)
            if action.ndim != 2 or action.shape[1] != 14 or action.shape[0] <= 0:
                raise ValueError(f"Invalid action shape {action.shape} in {hdf5_path}")
            if not np.isfinite(action).all():
                raise ValueError(f"NaN/Inf action in {hdf5_path}")
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
            stats_chunks["observation.state"].append(state)
            stats_chunks["action"].append(action)
            stats_chunks["timestamp"].append(np.arange(length, dtype=np.float32)[:, None] / FPS)
            stats_chunks["frame_index"].append(np.arange(length, dtype=np.float32)[:, None])
            stats_chunks["episode_index"].append(np.full((length, 1), episode_index, dtype=np.float32))
            stats_chunks["index"].append(np.arange(total_frames, total_frames + length, dtype=np.float32)[:, None])
            stats_chunks["task_index"].append(np.full((length, 1), task_index, dtype=np.float32))
            for out_key, camera_name in CAMERA_MAP.items():
                frames = handle[f"observation/{camera_name}/rgb"]
                if len(frames) != length:
                    raise ValueError(
                        f"Camera/action length mismatch in {hdf5_path}: "
                        f"{camera_name}={len(frames)}, action={length}"
                    )
                if image_size is None:
                    with Image.open(io.BytesIO(bytes(frames[0]))) as image:
                        image_size = image.convert("RGB").size
                episode_chunk = episode_index // CHUNK_SIZE
                video_path = (
                    staging_dir
                    / "videos"
                    / f"chunk-{episode_chunk:03d}"
                    / out_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                written_frames = write_video(
                    video_path,
                    frames,
                    width=image_size[0],
                    height=image_size[1],
                )
                if written_frames != length:
                    raise ValueError(
                        f"Encoded frame count mismatch in {video_path}: "
                        f"{written_frames} != {length}"
                    )

        episode_chunk = episode_index // CHUNK_SIZE
        chunk_dir = staging_dir / "data" / f"chunk-{episode_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns).to_parquet(
            chunk_dir / f"episode_{episode_index:06d}.parquet",
            index=False,
        )
        episodes_rows.append({"episode_index": episode_index, "tasks": [canonical_task], "length": length})
        total_frames += length

    if image_size is None:
        image_size = (320, 240)
    shutil.copyfile(modality_src, staging_dir / "meta" / "modality.json")
    write_jsonl(staging_dir / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(staging_dir / "meta" / "episodes.jsonl", episodes_rows)
    write_jsonl(
        staging_dir / "meta" / "task_language" / "robotwin_task_language_audit.jsonl",
        language_audit_rows,
    )
    write_jsonl(staging_dir / "meta" / "audit" / "episode_blacklist.jsonl", [])
    write_jsonl(staging_dir / "meta" / "audit" / "duplicate_episodes.jsonl", [])
    write_abs_stats(staging_dir / "meta" / "stats_gr00t.json", stats_chunks)
    (staging_dir / "meta" / "info.json").write_text(
        json.dumps(build_info(len(hdf5_files), total_frames, 1, image_size[0], image_size[1]), indent=4),
    )
    conversion_audit = {
        "dataset": canonical_task,
        "source": str(extracted_root),
        "episodes": len(hdf5_files),
        "frames": total_frames,
        "videos": len(hdf5_files) * len(CAMERA_MAP),
        "action_semantics": "absolute joint position",
        "state_semantics": "observed joint position",
        "control_hz": FPS,
        "canonical_task_count": 1,
        "raw_language_rows": len(language_audit_rows),
        "content_digest": hashlib.sha256(
            "\n".join(
                f"{row['episode_index']}:{row['raw_description']}" for row in language_audit_rows
            ).encode("utf-8")
        ).hexdigest(),
    }
    (staging_dir / "meta" / "audit" / "conversion_audit.json").write_text(
        json.dumps(conversion_audit, indent=2, ensure_ascii=False) + "\n"
    )
    if output_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except Exception:
        if not output_dir.exists() and backup_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def prepare_one(
    task: str, split: str, raw_root: Path, output_root: Path, modality_src: Path, args: argparse.Namespace
) -> None:
    output_split, _, _ = split_names(split)
    output_dir = output_root / output_split / task
    if output_dir.exists() and not args.force:
        print(f"[skip] {output_dir} already exists")
        return

    zip_path = (
        download_zip(task, split, raw_root, args.download_retries) if args.download else local_zip(task, split, raw_root)
    )
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")

    print(f"[convert] {zip_path} -> {output_dir}")
    with tempfile.TemporaryDirectory(prefix="robotwin_extract_") as tmp:
        extracted_root = extract_zip(zip_path, Path(tmp))
        convert_extracted(extracted_root, output_dir, modality_src, task_name=task)

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
