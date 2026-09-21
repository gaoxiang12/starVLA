"""Compare independently encoded source RGB videos with converted training videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import av
import numpy as np


def first_frame(path):
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        return next(container.decode(video=0)).to_ndarray(format="rgb24").astype(np.float32)


def compare(args):
    raw_root, converted_root, task, episode = args
    try:
        original = first_frame(raw_root / task / "demo_clean/video" / f"episode{episode}.mp4")
        converted = first_frame(converted_root / task / "videos/chunk-000/observation.images.cam_high" / f"episode_{episode:06d}.mp4")
        if original.shape != converted.shape:
            raise ValueError(f"Shape mismatch: {original.shape}, {converted.shape}")
        mask = np.abs(original[..., 0] - original[..., 2]) > 30
        direct = np.abs(original - converted)
        swapped = np.abs(original - converted[..., ::-1])
        return dict(task=task, episode=episode, direct_mae=float(direct.mean()),
                    swapped_mae=float(swapped.mean()), green_mae=float(direct[..., 1].mean()),
                    chromatic_pixels=int(mask.sum()),
                    chromatic_direct_mae=float(direct[mask].mean()) if mask.any() else None,
                    chromatic_swapped_mae=float(swapped[mask].mean()) if mask.any() else None)
    except Exception as exc:
        return dict(task=task, episode=episode, error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/data/gaoxiang/RoboTwinGenerated_raw/Clean"))
    parser.add_argument("--converted-root", type=Path, default=Path("/data/gaoxiang/RoboTwinGenerated/Clean"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = sorted(p.name for p in args.raw_root.iterdir() if p.is_dir())
    jobs = [(args.raw_root, args.converted_root, task, ep) for task in tasks for ep in (0, 20, 499)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        samples = list(pool.map(compare, jobs))
    successes = [s for s in samples if "error" not in s]
    report = dict(scope="First head-camera frame of episodes 0, 20, 499 in each locally available raw task. Does not exhaustively verify frames, wrist cameras, or appended episodes 500-999.",
                  raw_root=str(args.raw_root), converted_root=str(args.converted_root),
                  task_count=len(tasks), sample_count=len(samples), error_count=len(samples)-len(successes),
                  swap_lower_mae_count=sum(s["swapped_mae"] < s["direct_mae"] for s in successes),
                  all_samples_swap_better_tasks=[t for t in tasks if all("error" not in s and s["swapped_mae"] < s["direct_mae"] for s in samples if s["task"] == t)],
                  samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    for task in ("blocks_ranking_rgb", "blocks_ranking_size", "lift_pot", "adjust_bottle", "click_alarmclock"):
        rows = [s for s in successes if s["task"] == task]
        print(task, {k: float(np.mean([s[k] for s in rows])) for k in ("direct_mae", "swapped_mae")})


if __name__ == "__main__":
    main()
