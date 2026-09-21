"""Compare real checkpoint inference with the upstream RobotWinInference path."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.model.framework.WM4A.LiLaWAM import LiLaWAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation", type=Path, help="Optional real click_bell NPZ with image and state")
    args = parser.parse_args()
    settings = OmegaConf.load(args.config)
    start = time.time()
    adapter = LiLaWAM(settings)
    sys.path.insert(0, str(settings.framework.lila_wam.source_root))
    from robotwin_infer import RobotWinInference
    with tempfile.TemporaryDirectory(prefix="lila_parity_") as directory:
        config_path = Path(directory) / "config.yaml"
        OmegaConf.save(adapter.lila_config, config_path)
        reference = RobotWinInference(
            str(config_path), str(settings.framework.lila_wam.checkpoint_path),
            str(settings.framework.lila_wam.norm_stats_path), task_name="click_bell",
        )
    rng = np.random.default_rng(971)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    state = np.array([-.2, 0, .8, 1, 0, 0, 0, 1, .2, 0, .8, 1, 0, 0, 0, 1], np.float32)
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as sample:
            image, state = sample["image"], sample["state"]
    observation = {"observation": {"head_camera": {"rgb": image}}, "endpose": {
        "left_endpose": state[:7], "left_gripper": state[7],
        "right_endpose": state[8:15], "right_gripper": state[15]}}
    reference.processor.update_state_buffer(observation)
    torch.manual_seed(971)
    initial_rng = torch.cuda.get_rng_state().cpu().numpy()
    expected = reference._predict_chunk(observation)
    expected_rng = torch.cuda.get_rng_state().cpu().numpy()
    torch.manual_seed(12345)  # Server-local seed must be overridden by simulator state.
    response = adapter.predict_action([dict(task_name="click_bell", image=[image], state=state)],
                                      cuda_rng_state=initial_rng)
    actual = response["actions"][0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(response["cuda_rng_state"], expected_rng)
    result = dict(state="passed", comparison="bitwise identical to upstream full chunk",
                  shape=list(actual.shape), max_absolute_error=float(np.abs(actual - expected).max()),
                  elapsed_seconds=time.time() - start, torch=torch.__version__,
                  checkpoint=adapter.metadata["ckpt_path"], synthetic_observation=not bool(args.observation),
                  benchmark_success_claim=False)
    result["cuda_rng_roundtrip"] = "bitwise identical to upstream post-inference generator state"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
