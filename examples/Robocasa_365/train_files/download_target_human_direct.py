#!/usr/bin/env python3
"""Download RoboCasa365 target/human LeRobot bundles without importing robocasa."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import tarfile
import urllib.request
from pathlib import Path

import certifi
from tqdm import tqdm

from examples.Robocasa_365.train_files.data_registry.data_config import (
    _TARGET_HUMAN_ATOMIC,
)


BOX_LINKS = Path("playground/Code/robocasa365/robocasa/models/assets/box_links/box_links_ds.json")
DEFAULT_DATASET_ROOT = Path("playground/Datasets/robocasa365")


class DownloadProgressBar(tqdm):
    def update_to(self, block_count: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(block_count * block_size - self.n)


def direct_box_url(shared_url: str, ext: str = "tar") -> str:
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.{ext}"


def task_paths() -> dict[str, str]:
    return dict(_TARGET_HUMAN_ATOMIC)


def select_tasks(values: list[str]) -> dict[str, str]:
    paths = task_paths()
    if values == ["atomic-seen"] or "atomic-seen" in values:
        return paths
    unknown = sorted(set(values) - set(paths))
    if unknown:
        raise ValueError(f"Unknown RoboCasa365 target/human task(s): {', '.join(unknown)}")
    return {task: paths[task] for task in values}


def download(url: str, path: Path, insecure: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "starVLA-data-prep/1.0"})
    with urllib.request.urlopen(request, context=context) as response:
        total = response.headers.get("Content-Length")
        with DownloadProgressBar(
            total=int(total) if total else None,
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=path.name,
        ) as progress, path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))


def prepare_one(
    task: str,
    rel_path: str,
    dataset_root: Path,
    links: dict[str, str],
    overwrite: bool,
    dry_run: bool,
    insecure: bool,
) -> None:
    tar_key = f"{rel_path.removeprefix('v1.0/')}/lerobot.tar"
    if tar_key not in links:
        print(f"[skip] {task}: no Box link for {tar_key}")
        return

    output_dir = dataset_root / rel_path / "lerobot"
    if output_dir.exists() and not overwrite:
        print(f"[skip] {task}: {output_dir} already exists")
        return

    extract_dir = output_dir.parent
    tar_path = extract_dir / "lerobot.tar"
    print(f"[download] {task}: {tar_key} -> {output_dir}")
    if dry_run:
        return

    if output_dir.exists() and overwrite:
        import shutil

        shutil.rmtree(output_dir)
    download(direct_box_url(links[tar_key]), tar_path, insecure=insecure)
    with tarfile.open(tar_path, "r") as archive:
        archive.extractall(path=extract_dir)
    os.remove(tar_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--box-links", type=Path, default=BOX_LINKS)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["atomic-seen"],
        help="Atomic-Seen task names or 'atomic-seen' (default: all 18 tasks).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for Box downloads.")
    args = parser.parse_args()

    links = json.loads(args.box_links.read_text())
    for task, rel_path in select_tasks(args.tasks).items():
        prepare_one(task, rel_path, args.dataset_root, links, args.overwrite, args.dry_run, args.insecure)

    print(f"[done] RoboCasa365 dataset root: {args.dataset_root}")


if __name__ == "__main__":
    main()