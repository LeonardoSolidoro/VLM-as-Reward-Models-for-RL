"""
Generate expert trajectories for supervised task-progress experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .cameras import MovingCameraParameters, configure_camera
from .common import (
    CAMERA_MODES,
    DEFAULT_TASKS,
    load_project_config,
    prepare_output_directory,
    project_root,
    resolve_path,
    seed_everything,
    stable_seed,
)
from .maniskill_io import (
    demonstration_paths,
    load_successful_episodes,
    make_env,
    maniskill_version,
    render_warmed,
    resolve_demo_root,
    sample_indices,
    save_rgb_image,
    trajectory_state,
    write_rollout_files,
)


def sampled_trajectory_rewards(
    trajectory_rewards: Sequence[float],
    sampled_state_indices: Sequence[int],
) -> list[float]:
    if len(trajectory_rewards) == 0:
        raise ValueError("A demonstration trajectory must contain at least one reward")
    rewards: list[float] = []
    for state_index in sampled_state_indices:
        if state_index == 0:
            rewards.append(0.0)
        else:
            reward_index = min(state_index - 1, len(trajectory_rewards) - 1)
            rewards.append(float(trajectory_rewards[reward_index]))
    return rewards


def export_progress_task(
    *,
    task: str,
    camera_mode: str,
    data_root: Path,
    demo_root: Path,
    view: str,
    num_rollouts: int,
    num_frames: int,
    seed: int,
    overwrite: bool,
) -> None:
    episodes = load_successful_episodes(demo_root, task, num_rollouts)
    control_modes = {str(episode["control_mode"]) for episode in episodes}
    if len(control_modes) != 1:
        raise ValueError(f"{task} demonstrations use multiple control modes: {control_modes}")
    control_mode = next(iter(control_modes))

    _, h5_path = demonstration_paths(demo_root, task)
    expert_root = data_root / camera_mode / task / "expert"
    prepare_output_directory(expert_root, overwrite=overwrite)

    env = make_env(
        task,
        obs_mode="none",
        control_mode=control_mode,
        reward_mode="none",
    )
    camera_rng = np.random.default_rng(stable_seed(seed, task, camera_mode))
    try:
        with h5py.File(h5_path, "r") as trajectories:
            for rollout_index, episode in enumerate(episodes):
                trajectory = trajectories[f"traj_{episode['episode_id']}"]
                trajectory_rewards = trajectory["rewards"][:]
                num_states = len(trajectory_rewards) + 1
                state_indices = sample_indices(num_states, num_frames)
                moving_parameters = (
                    MovingCameraParameters.sample(camera_rng)
                    if camera_mode == "moving"
                    else None
                )

                env.reset(**episode["reset_kwargs"])
                rollout_directory = expert_root / f"rollout_{rollout_index}"
                for frame_index, state_index in enumerate(state_indices):
                    state = trajectory_state(trajectory["env_states"], int(state_index))
                    env.unwrapped.set_state_dict(state)
                    configure_camera(
                        env.unwrapped,
                        task,
                        camera_mode,
                        frame_index,
                        num_frames,
                        moving_parameters,
                        camera_rng,
                    )
                    image = render_warmed(env.unwrapped)
                    save_rgb_image(
                        rollout_directory / f"{view}_frame_{frame_index:03d}.jpg",
                        image,
                    )

                sampled_rewards = sampled_trajectory_rewards(
                    trajectory_rewards,
                    state_indices,
                )
                metadata: dict[str, Any] = {
                    "task": task,
                    "camera_mode": camera_mode,
                    "view": view,
                    "reward_mode": "demonstration_dense",
                    "reward_source": "trajectory_h5",
                    "mani_skill_version": maniskill_version(),
                    "source_episode_id": int(episode["episode_id"]),
                    "total_steps": len(trajectory_rewards),
                    "source_state_indices": [int(index) for index in state_indices],
                    "frame_steps": [int(index) for index in state_indices],
                }
                write_rollout_files(rollout_directory, sampled_rewards, metadata)
                print(f"{task} {camera_mode}: exported rollout_{rollout_index}")
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay ManiSkill expert demonstrations from selected camera modes."
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--demo-root", default="demos")
    parser.add_argument(
        "--camera-modes",
        nargs="+",
        choices=CAMERA_MODES,
        default=["static"],
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--view", default=None)
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = project_root()
    config = load_project_config(repo_root)

    data_root_value = config["data_root"] if args.data_root is None else args.data_root
    data_root = resolve_path(data_root_value, repo_root)
    demo_root = resolve_demo_root(repo_root, args.demo_root)
    seed = int(config["seed"]) if args.seed is None else args.seed
    num_rollouts = int(config["num_rollouts"]) if args.num_rollouts is None else args.num_rollouts
    num_frames = int(config["num_frames"]) if args.num_frames is None else args.num_frames

    if args.view is None:
        configured_views = config["views"]
        if len(configured_views) != 1:
            raise ValueError(
                "The generator renders one camera view; pass --view explicitly when configs.views "
                f"contains {len(configured_views)} names"
            )
        view = str(configured_views[0])
    else:
        view = args.view

    camera_modes = args.camera_modes

    seed_everything(seed, deterministic=args.deterministic)
    print(f"Writing progress trajectories to: {data_root}")
    print(f"Camera modes: {camera_modes}")
    for camera_mode in camera_modes:
        for task in args.tasks:
            export_progress_task(
                task=task,
                camera_mode=camera_mode,
                data_root=data_root,
                demo_root=demo_root,
                view=view,
                num_rollouts=num_rollouts,
                num_frames=num_frames,
                seed=seed,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
