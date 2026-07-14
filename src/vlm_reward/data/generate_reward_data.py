"""
Generate normalized reward trajectories for contrastive reward learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import h5py
import numpy as np

from .cameras import MovingCameraParameters, configure_camera
from .common import (
    CAMERA_MODES,
    DEFAULT_NUM_FRAMES,
    DEFAULT_TASKS,
    DEFAULT_VIEW,
    REWARD_LEVELS,
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
    replay_normalized_rewards,
    resolve_demo_root,
    sample_indices,
    save_rgb_image,
    scalar_bool,
    scalar_float,
    trajectory_state,
    validate_normalized_reward,
    write_rollout_files,
)


DEFAULT_SPLIT_PERCENTAGES = (0.20, 0.25, 0.35, 0.20)


def rollout_counts(
    num_rollouts: int,
    split_percentages: Sequence[float],
) -> dict[str, int]:
    if num_rollouts < 1:
        raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")
    if len(split_percentages) != len(REWARD_LEVELS):
        raise ValueError(
            f"Expected {len(REWARD_LEVELS)} split percentages, got {len(split_percentages)}"
        )
    if any(percentage < 0.0 for percentage in split_percentages):
        raise ValueError(f"Split percentages must be non-negative: {split_percentages}")
    if not np.isclose(sum(split_percentages), 1.0, atol=1e-6):
        raise ValueError(f"Split percentages must sum to 1.0, got {sum(split_percentages)}")

    expert = int(num_rollouts * split_percentages[0])
    partial = int(num_rollouts * split_percentages[1])
    random_count = int(num_rollouts * split_percentages[2])
    regressing = num_rollouts - expert - partial - random_count
    return {
        "expert": expert,
        "partial": partial,
        "random": random_count,
        "regressing": regressing,
    }


def export_rendered_states(
    *,
    env: gym.Env,
    task: str,
    states: Any,
    states_are_list: bool,
    state_rewards: Sequence[float],
    sampled_state_indices: Sequence[int],
    rollout_directory: Path,
    view: str,
    camera_mode: str,
    total_steps: int,
    reward_source: str,
    source_episode_id: int | None,
    camera_rng: np.random.Generator,
) -> None:
    expected_states = total_steps + 1
    if len(state_rewards) != expected_states:
        raise ValueError(
            f"{task} has {len(state_rewards)} state rewards, expected {expected_states}"
        )

    moving_parameters = (
        MovingCameraParameters.sample(camera_rng) if camera_mode == "moving" else None
    )
    sampled_rewards: list[float] = []
    num_frames = len(sampled_state_indices)
    for frame_index, state_index_value in enumerate(sampled_state_indices):
        state_index = int(state_index_value)
        state = states[state_index] if states_are_list else trajectory_state(states, state_index)
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
        reward = validate_normalized_reward(
            float(state_rewards[state_index]),
            f"{task} state {state_index}",
        )
        sampled_rewards.append(round(reward, 6))
        save_rgb_image(
            rollout_directory / f"{view}_frame_{frame_index:03d}.jpg",
            image,
        )

    metadata = {
        "task": task,
        "camera_mode": camera_mode,
        "view": view,
        "reward_mode": "normalized_dense",
        "reward_source": reward_source,
        "initial_state_reward_convention": 0.0,
        "mani_skill_version": maniskill_version(),
        "source_episode_id": source_episode_id,
        "total_steps": total_steps,
        "source_state_indices": [int(index) for index in sampled_state_indices],
        "frame_steps": [int(index) for index in sampled_state_indices],
    }
    write_rollout_files(rollout_directory, sampled_rewards, metadata)


def regression_indices(
    num_states: int,
    num_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_states < num_frames:
        raise ValueError(f"Need at least {num_frames} states for regression, got {num_states}")

    minimum_turn = min(num_frames, num_states - 1)
    maximum_turn = max(minimum_turn, int(0.8 * num_states))
    maximum_turn = min(maximum_turn, num_states - 1)
    turn = (
        int(rng.integers(minimum_turn, maximum_turn + 1))
        if minimum_turn < maximum_turn
        else minimum_turn
    )
    minimum_forward = max(1, int(num_frames * 0.20))
    maximum_forward = max(minimum_forward, int(num_frames * 0.80))
    forward_count = int(rng.integers(minimum_forward, maximum_forward + 1))
    backward_count = num_frames - forward_count

    forward_indices = np.round(np.linspace(0, turn, forward_count)).astype(int)
    forward_set = set(forward_indices.tolist())
    available = [index for index in range(turn + 1) if index not in forward_set]
    if len(available) < backward_count:
        available.extend(
            index for index in range(turn + 1, num_states) if index not in forward_set
        )
    if len(available) < backward_count:
        raise RuntimeError(
            f"Could not construct {backward_count} regression frames from {num_states} states"
        )

    positions = np.round(np.linspace(len(available) - 1, 0, backward_count)).astype(int)
    backward_indices = np.asarray([available[position] for position in positions], dtype=int)
    indices = np.concatenate([forward_indices, backward_indices])
    if len(indices) != num_frames:
        raise RuntimeError(f"Generated {len(indices)} regression indices, expected {num_frames}")
    return indices


def collect_random_trajectory(
    env: gym.Env,
    task: str,
    rollout_index: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    rollout_seed = seed + rollout_index
    env.reset(seed=rollout_seed)
    env.action_space.seed(rollout_seed)
    states = [env.unwrapped.get_state_dict()]
    rewards = [0.0]
    momentum_action = env.action_space.sample()
    target_action = env.action_space.sample()

    while True:
        if len(states) % 10 == 0:
            target_action = env.action_space.sample()
        momentum_action = 0.90 * momentum_action + 0.10 * target_action
        action = np.clip(momentum_action, env.action_space.low, env.action_space.high)
        _, reward_value, terminated, truncated, _ = env.step(action)
        reward = validate_normalized_reward(
            scalar_float(reward_value, f"{task} random rollout {rollout_index}"),
            f"{task} random rollout {rollout_index}",
        )
        states.append(env.unwrapped.get_state_dict())
        rewards.append(reward)
        if scalar_bool(terminated, "terminated") or scalar_bool(truncated, "truncated"):
            break
    return states, rewards


def export_reward_task(
    *,
    task: str,
    camera_mode: str,
    data_root: Path,
    demo_root: Path,
    view: str,
    num_rollouts: int,
    num_frames: int,
    split_percentages: Sequence[float],
    seed: int,
    render_control_mode: str,
    overwrite: bool,
) -> None:
    counts = rollout_counts(num_rollouts, split_percentages)
    required_episodes = counts["expert"] + counts["partial"] + counts["regressing"]
    episodes = load_successful_episodes(demo_root, task, required_episodes)
    demo_control_modes = {str(episode["control_mode"]) for episode in episodes}
    if len(demo_control_modes) != 1:
        raise ValueError(f"{task} demonstrations use multiple control modes: {demo_control_modes}")
    demo_control_mode = next(iter(demo_control_modes))

    task_root = data_root / camera_mode / task
    prepare_output_directory(task_root, overwrite=overwrite)
    for level in REWARD_LEVELS:
        (task_root / level).mkdir()

    _, h5_path = demonstration_paths(demo_root, task)
    render_env = make_env(
        task,
        obs_mode="state",
        control_mode=render_control_mode,
        reward_mode="normalized_dense",
    )
    replay_env = make_env(
        task,
        obs_mode="state",
        control_mode=demo_control_mode,
        reward_mode="normalized_dense",
    )
    trajectory_rng = np.random.default_rng(stable_seed(seed, task))
    camera_rng = np.random.default_rng(stable_seed(seed, task, camera_mode, "camera"))

    try:
        with h5py.File(h5_path, "r") as trajectories:
            for rollout_index in range(counts["expert"]):
                episode = episodes[rollout_index]
                trajectory = trajectories[f"traj_{episode['episode_id']}"]
                rewards = replay_normalized_rewards(replay_env, episode, trajectory, task)
                render_env.reset(**episode["reset_kwargs"])
                num_states = len(trajectory["actions"]) + 1
                export_rendered_states(
                    env=render_env,
                    task=task,
                    states=trajectory["env_states"],
                    states_are_list=False,
                    state_rewards=rewards,
                    sampled_state_indices=sample_indices(num_states, num_frames),
                    rollout_directory=task_root / "expert" / f"rollout_{rollout_index}",
                    view=view,
                    camera_mode=camera_mode,
                    total_steps=num_states - 1,
                    reward_source="replayed_env_step",
                    source_episode_id=int(episode["episode_id"]),
                    camera_rng=camera_rng,
                )

            partial_start = counts["expert"]
            for offset in range(counts["partial"]):
                rollout_index = partial_start + offset
                episode = episodes[rollout_index]
                trajectory = trajectories[f"traj_{episode['episode_id']}"]
                rewards = replay_normalized_rewards(replay_env, episode, trajectory, task)
                render_env.reset(**episode["reset_kwargs"])
                num_states = len(trajectory["actions"]) + 1
                maximum_cutoff = num_frames + int((num_states - num_frames) * 2 / 3)
                maximum_cutoff = max(num_frames, min(maximum_cutoff, num_states))
                cutoff = int(trajectory_rng.integers(num_frames, maximum_cutoff + 1))
                export_rendered_states(
                    env=render_env,
                    task=task,
                    states=trajectory["env_states"],
                    states_are_list=False,
                    state_rewards=rewards,
                    sampled_state_indices=sample_indices(cutoff, num_frames),
                    rollout_directory=task_root / "partial" / f"rollout_{rollout_index}",
                    view=view,
                    camera_mode=camera_mode,
                    total_steps=num_states - 1,
                    reward_source="replayed_env_step",
                    source_episode_id=int(episode["episode_id"]),
                    camera_rng=camera_rng,
                )

            random_start = counts["expert"] + counts["partial"]
            for offset in range(counts["random"]):
                rollout_index = random_start + offset
                states, rewards = collect_random_trajectory(
                    render_env,
                    task,
                    rollout_index,
                    seed,
                )
                export_rendered_states(
                    env=render_env,
                    task=task,
                    states=states,
                    states_are_list=True,
                    state_rewards=rewards,
                    sampled_state_indices=sample_indices(len(states), num_frames),
                    rollout_directory=task_root / "random" / f"rollout_{rollout_index}",
                    view=view,
                    camera_mode=camera_mode,
                    total_steps=len(states) - 1,
                    reward_source="collected_env_step",
                    source_episode_id=None,
                    camera_rng=camera_rng,
                )

            regression_start = counts["expert"] + counts["partial"] + counts["random"]
            regression_episode_start = counts["expert"] + counts["partial"]
            for offset in range(counts["regressing"]):
                rollout_index = regression_start + offset
                episode = episodes[regression_episode_start + offset]
                trajectory = trajectories[f"traj_{episode['episode_id']}"]
                rewards = replay_normalized_rewards(replay_env, episode, trajectory, task)
                render_env.reset(**episode["reset_kwargs"])
                num_states = len(trajectory["actions"]) + 1
                export_rendered_states(
                    env=render_env,
                    task=task,
                    states=trajectory["env_states"],
                    states_are_list=False,
                    state_rewards=rewards,
                    sampled_state_indices=regression_indices(
                        num_states,
                        num_frames,
                        trajectory_rng,
                    ),
                    rollout_directory=task_root / "regressing" / f"rollout_{rollout_index}",
                    view=view,
                    camera_mode=camera_mode,
                    total_steps=num_states - 1,
                    reward_source="replayed_env_step",
                    source_episode_id=int(episode["episode_id"]),
                    camera_rng=camera_rng,
                )
    finally:
        replay_env.close()
        render_env.close()

    print(f"{task} {camera_mode}: saved {num_rollouts} rollouts to {task_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build expert, partial, random, and regressing normalized-reward trajectories."
    )
    parser.add_argument("--data-root", default="data/reward_contrastive")
    parser.add_argument("--demo-root", default="demos")
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--num-rollouts", type=int, default=500)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument(
        "--split-percentages",
        nargs=4,
        type=float,
        default=list(DEFAULT_SPLIT_PERCENTAGES),
        metavar=("EXPERT", "PARTIAL", "RANDOM", "REGRESSING"),
    )
    parser.add_argument(
        "--camera-modes",
        nargs="+",
        choices=CAMERA_MODES,
        default=["static", "moving_mounted"],
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--render-control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = project_root()
    config = load_project_config(repo_root)
    seed = int(config["seed"]) if args.seed is None else args.seed
    seed_everything(seed, deterministic=args.deterministic)

    data_root = resolve_path(args.data_root, repo_root)
    demo_root = resolve_demo_root(repo_root, args.demo_root)
    print(f"Writing normalized reward trajectories to: {data_root}")
    print(f"Rollout mix [expert, partial, random, regressing]: {args.split_percentages}")
    for camera_mode in args.camera_modes:
        for task in args.tasks:
            export_reward_task(
                task=task,
                camera_mode=camera_mode,
                data_root=data_root,
                demo_root=demo_root,
                view=args.view,
                num_rollouts=args.num_rollouts,
                num_frames=args.num_frames,
                split_percentages=args.split_percentages,
                seed=seed,
                render_control_mode=args.render_control_mode,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
