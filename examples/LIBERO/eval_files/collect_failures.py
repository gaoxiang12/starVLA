# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Collect failure cases of a policy on (a subset of) LIBERO tasks.

This mirrors the rollout loop in ``eval_libero.py`` but, instead of only
reporting the success rate, it records the *full trajectory* of every failed
episode so that a stronger expert (e.g. a high-success Qwen-OFT policy) can
later be dropped in at an arbitrary failure timestep and attempt a recovery
(DAgger / intervention-style data collection).

For every failed episode we dump, under
``<out>/<task_seg>/episode_<global_idx>/``:

    init_state.npy   (D,)         LIBERO initial state used for this episode
    sim_states.npy   (N, D)       flattened mujoco sim state BEFORE each policy step
    actions.npy      (N, 7)       delta action applied at each policy step
    proprio.npy      (N, 8)       robot state fed to the model at each step
    steps.npy        (N,)         policy step index (0-based, excludes warmup)
    rollout.mp4                   replay video (optional)
    meta.json                     task / seed / step-count / ckpt metadata

Because ``sim_states[k]`` is captured *before* applying ``actions[k]``, you can
reproduce the environment at any decision point via
``env.regenerate_obs_from_state(sim_states[k])`` and then let the expert policy
take over from step ``k``.

Usage (two processes, same as normal eval):

    # 1) start the policy server with YOUR model
    CKPT=.../lewm_oft_libero_wm_vfuse/checkpoints/steps_80000_pytorch_model.pt \
        PORT=6699 bash examples/LIBERO/eval_files/run_policy_server.sh

    # 2) run this collector against the server
    CKPT=.../steps_80000_pytorch_model.pt LIBERO_HOME=$PWD/playground/LIBERO \
        PORT=6699 bash examples/LIBERO/eval_files/collect_failures.sh
"""

import dataclasses
import json
import logging
import os
import pathlib
import time

import imageio
import numpy as np
import torch
import tqdm
import tyro

# Reuse the exact preprocessing / env helpers from the evaluator so recorded
# trajectories stay byte-compatible with normal evaluation. Importing this
# module also installs the ``torch.load(weights_only=False)`` shim needed by the
# LIBERO init-state files.
from examples.LIBERO.eval_files.eval_libero import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _binarize_gripper_open,
    _get_libero_env,
    _quat2axisangle,
)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

from libero.libero import benchmark

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Keep a reference so the import above is not flagged as unused; the shim is a
# side effect of importing eval_libero.
_ = torch.load

_MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"
    # Only collect from tasks whose language description contains this substring
    # (case-insensitive). Empty string == every task in the suite.
    task_filter: str = "open the top drawer and put the bowl inside"
    num_steps_wait: int = 10  # steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # rollouts per matched task
    max_tasks: int = -1  # cap number of matched tasks (-1 = no cap)

    #################################################################################################################
    # Collection outputs
    #################################################################################################################
    # Root directory for the collected failure cases. Empty == auto-place next
    # to the checkpoint: <model_root>/failure_cases/<suite>/<ckpt_tag>.
    out_path: str = ""
    save_video: bool = True  # save an .mp4 replay for every failed episode
    save_success: bool = False  # also dump successful episodes (default: only failures)

    seed: int = 7  # random seed (for reproducibility)

    pretrained_path: str = ""  # only used for provenance in meta.json / auto out_path
    unnorm_key: str | None = None

    job_name: str = "collect_failures"


def _sanitize(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def _auto_out_path(args: Args) -> str:
    if args.out_path:
        return args.out_path
    ckpt = args.pretrained_path
    if ckpt and "/checkpoints/" in ckpt:
        model_root = ckpt.split("/checkpoints/")[0]
        tag = _sanitize(pathlib.Path(ckpt).stem)
    else:
        model_root = os.getcwd()
        tag = "model"
    return os.path.join(model_root, "failure_cases", args.task_suite_name, tag)


def collect_failures(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    np.random.seed(args.seed)

    out_root = _auto_out_path(args)
    pathlib.Path(out_root).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(out_root, "manifest.jsonl")
    logging.info(f"Recording failure cases under: {out_root}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if args.task_suite_name not in _MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = _MAX_STEPS_BY_SUITE[args.task_suite_name]

    # Resolve which tasks to run based on the language filter.
    needle = args.task_filter.strip().lower()
    matched_task_ids = [
        tid
        for tid in range(num_tasks_in_suite)
        if (not needle) or (needle in task_suite.get_task(tid).language.lower())
    ]
    if args.max_tasks > 0:
        matched_task_ids = matched_task_ids[: args.max_tasks]
    if not matched_task_ids:
        raise ValueError(
            f"No task in suite '{args.task_suite_name}' matches filter '{args.task_filter}'."
        )
    logging.info(
        f"Matched {len(matched_task_ids)} task(s): "
        + ", ".join(f"[{tid}] {task_suite.get_task(tid).language}" for tid in matched_task_ids)
    )

    client_model = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)

    total_episodes = 0
    total_failures = 0
    total_saved = 0

    for task_id in tqdm.tqdm(matched_task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_seg = _sanitize(task_description)

        n_trials = min(args.num_trials_per_task, len(initial_states))
        for episode_idx in tqdm.tqdm(range(n_trials), desc=f"task{task_id}", leave=False):
            client_model.reset(task_description=task_description)
            env.reset()
            init_state = np.asarray(initial_states[episode_idx])
            obs = env.set_init_state(init_state)

            t = 0
            step = 0
            done = False
            replay_images = []
            traj_sim_states = []
            traj_actions = []
            traj_proprio = []
            traj_steps = []

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                example_dict = {
                    "image": [img, wrist_img],
                    "lang": str(task_description),
                }

                # Capture the sim state at THIS decision point (before acting) so
                # an expert can later be handed control from exactly here.
                sim_state = np.asarray(env.get_sim_state(), dtype=np.float64)

                response = client_model.step(example=example_dict, step=step)
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                traj_sim_states.append(sim_state)
                traj_actions.append(delta_action.astype(np.float32))
                traj_proprio.append(state.astype(np.float32))
                traj_steps.append(step)

                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    break
                t += 1
                step += 1

            total_episodes += 1
            is_failure = not done
            if is_failure:
                total_failures += 1

            if is_failure or args.save_success:
                total_saved += _dump_episode(
                    out_root=out_root,
                    manifest_path=manifest_path,
                    task_seg=task_seg,
                    global_idx=total_episodes - 1,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    task_description=task_description,
                    task=task,
                    args=args,
                    server_meta=client_model._server_metadata,
                    max_steps=max_steps,
                    success=done,
                    init_state=init_state,
                    sim_states=traj_sim_states,
                    actions=traj_actions,
                    proprio=traj_proprio,
                    steps=traj_steps,
                    replay_images=replay_images,
                )

            logging.info(
                f"task {task_id} ep {episode_idx}: {'SUCCESS' if done else 'FAILURE'} "
                f"| steps={len(traj_actions)} | failures={total_failures}/{total_episodes} | saved={total_saved}"
            )

        env.close()

    summary = {
        "task_suite_name": args.task_suite_name,
        "task_filter": args.task_filter,
        "matched_task_ids": matched_task_ids,
        "total_episodes": total_episodes,
        "total_failures": total_failures,
        "total_saved": total_saved,
        "failure_rate": (total_failures / total_episodes) if total_episodes else 0.0,
        "pretrained_path": args.pretrained_path,
        "out_path": out_root,
    }
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Done. {json.dumps(summary, indent=2)}")


def _dump_episode(
    *,
    out_root: str,
    manifest_path: str,
    task_seg: str,
    global_idx: int,
    task_id: int,
    episode_idx: int,
    task_description: str,
    task,
    args: Args,
    server_meta: dict,
    max_steps: int,
    success: bool,
    init_state: np.ndarray,
    sim_states: list,
    actions: list,
    proprio: list,
    steps: list,
    replay_images: list,
) -> int:
    """Persist one episode to disk. Returns 1 if written, else 0."""
    if len(actions) == 0:
        logging.warning(f"task {task_id} ep {episode_idx}: empty trajectory, skipping dump.")
        return 0

    ep_dir = pathlib.Path(out_root) / task_seg / f"episode_{global_idx:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    np.save(ep_dir / "init_state.npy", np.asarray(init_state))
    np.save(ep_dir / "sim_states.npy", np.stack(sim_states))
    np.save(ep_dir / "actions.npy", np.stack(actions))
    np.save(ep_dir / "proprio.npy", np.stack(proprio))
    np.save(ep_dir / "steps.npy", np.asarray(steps, dtype=np.int64))

    video_rel = None
    if args.save_video and replay_images:
        suffix = "success" if success else "failure"
        video_rel = f"rollout_{suffix}.mp4"
        imageio.mimwrite(
            ep_dir / video_rel,
            [np.asarray(x) for x in replay_images],
            fps=10,
            format="FFMPEG",
            macro_block_size=1,
        )

    meta = {
        "task_suite_name": args.task_suite_name,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "global_idx": global_idx,
        "task_description": task_description,
        "problem_folder": getattr(task, "problem_folder", None),
        "bddl_file": getattr(task, "bddl_file", None),
        "success": bool(success),
        "num_policy_steps": len(actions),
        "num_steps_wait": args.num_steps_wait,
        "max_steps": max_steps,
        "seed": args.seed,
        "pretrained_path": args.pretrained_path,
        "server_meta": server_meta,
        "video": video_rel,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "init_state": "init_state.npy",
            "sim_states": "sim_states.npy",
            "actions": "actions.npy",
            "proprio": "proprio.npy",
            "steps": "steps.npy",
        },
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(manifest_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "dir": str(ep_dir.relative_to(out_root)),
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "success": bool(success),
                    "num_policy_steps": len(actions),
                    "task_description": task_description,
                }
            )
            + "\n"
        )
    return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(collect_failures)
