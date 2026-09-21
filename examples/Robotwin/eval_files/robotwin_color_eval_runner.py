"""Diagnostic only: change policy input colors, preserving simulator and videos."""
import os
from pathlib import Path

import numpy as np


def policy_observation(observation, channel_order):
    if channel_order not in ("rgb", "bgr"):
        raise ValueError(f"Unknown policy channel order: {channel_order}")
    if channel_order == "rgb":
        return observation
    result = dict(observation)
    result["observation"] = dict(observation["observation"])
    for camera in ("head_camera", "left_camera", "right_camera"):
        result["observation"][camera] = dict(observation["observation"][camera])
        result["observation"][camera]["rgb"] = np.ascontiguousarray(
            observation["observation"][camera]["rgb"][..., ::-1]
        )
    return result


def main():
    import robotwin_ranking_eval_runner as ranking

    order = os.environ["ROBOTWIN_POLICY_CHANNEL_ORDER"]
    if order not in ("rgb", "bgr"):
        raise ValueError(order)
    original = ranking.interface.eval
    snapshot_root = os.environ.get("ROBOTWIN_COLOR_SNAPSHOT_DIR")

    def evaluate(env, model, observation):
        step = int(env.take_action_cnt)
        if snapshot_root and step in (0, 50, 70, 90, 130):
            # Save the simulator RGB before applying the policy-only intervention.
            # Sparse snapshots make first-grasp diagnosis possible during a run.
            from PIL import Image
            directory = Path(snapshot_root) / f"trial_{int(env.test_num):03d}"
            directory.mkdir(parents=True, exist_ok=True)
            for camera in ("head_camera", "left_camera", "right_camera"):
                Image.fromarray(observation["observation"][camera]["rgb"]).save(
                    directory / f"step_{step:04d}_{camera}.png")
        return original(env, model, policy_observation(observation, order))

    ranking.interface.eval = evaluate
    print(f"COLOR DIAGNOSTIC: policy={order}, simulator/video=rgb", flush=True)
    ranking.main()


if __name__ == "__main__":
    main()
