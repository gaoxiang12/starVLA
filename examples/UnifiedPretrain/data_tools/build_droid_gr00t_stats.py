"""Build the q01/q99 statistics cache required by the StarVLA loader.

DROID's official ``stats.json`` contains exact mean/std/min/max values but no
quantiles.  Quantiles are estimated from a deterministic, evenly spaced sample
of episodes so preparation does not concatenate all 27M frames in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOW_DIMENSIONAL_KEYS = ("observation.state", "action")


def build_statistics(dataset_dir: Path, sample_episodes: int) -> dict:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    official = json.loads((dataset_dir / "meta" / "stats.json").read_text())
    total_episodes = int(info["total_episodes"])
    sample_count = min(int(sample_episodes), total_episodes)
    episode_ids = np.linspace(
        0, total_episodes - 1, num=sample_count, dtype=np.int64
    )

    samples = {key: [] for key in LOW_DIMENSIONAL_KEYS}
    for episode_id in episode_ids:
        path = dataset_dir / info["data_path"].format(
            episode_chunk=int(episode_id) // int(info["chunks_size"]),
            episode_index=int(episode_id),
        )
        frame = pd.read_parquet(path, columns=list(LOW_DIMENSIONAL_KEYS))
        for key in LOW_DIMENSIONAL_KEYS:
            samples[key].append(np.stack(frame[key].to_numpy()).astype(np.float32))

    statistics = {}
    for key in LOW_DIMENSIONAL_KEYS:
        values = np.concatenate(samples[key], axis=0)
        statistics[key] = {
            name: official[key][name] for name in ("mean", "std", "min", "max")
        }
        statistics[key]["q01"] = np.quantile(values, 0.01, axis=0).tolist()
        statistics[key]["q99"] = np.quantile(values, 0.99, axis=0).tolist()

    return {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": statistics,
        "__provenance": {
            "mean_std_min_max": "meta/stats.json",
            "quantile_method": "deterministic_evenly_spaced_episode_sample",
            "quantile_sample_episodes": sample_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--sample-episodes", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset_dir / "meta" / "stats_gr00t.json"
    payload = build_statistics(args.dataset_dir, args.sample_episodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sample_episodes": payload["__provenance"][
                    "quantile_sample_episodes"
                ],
                "action_q01": payload["statistics"]["action"]["q01"],
                "action_q99": payload["statistics"]["action"]["q99"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
