"""
Shared ManiSkill demonstration, rendering, and reward-replay helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import gymnasium as gym
import h5py
import mani_skill
import mani_skill.envs  # noqa: F401 - importing registers ManiSkill environments
import numpy as np
import torch
from mani_skill.trajectory import utils as trajectory_utils


def resolve_demo_root(repo_root: Path, requested_root: str | Path) -> Path:
    requested = Path(requested_root).expanduser()
    requested = requested if requested.is_absolute() else repo_root / requested
    if not requested.is_dir():
        raise FileNotFoundError(f"ManiSkill demonstration root does not exist: {requested}")
    return requested


def demonstration_paths(demo_root: Path, task: str) -> tuple[Path, Path]:
    demo_directory = demo_root / task / "motionplanning"
    json_path = demo_directory / "trajectory.json"
    h5_path = demo_directory / "trajectory.h5"
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing ManiSkill trajectory metadata: {json_path}")
    if not h5_path.is_file():
        raise FileNotFoundError(f"Missing ManiSkill trajectory states: {h5_path}")
    return json_path, h5_path


def load_successful_episodes(
    demo_root: Path,
    task: str,
    num_episodes: int,
) -> list[dict[str, Any]]:
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    json_path, _ = demonstration_paths(demo_root, task)
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    episodes = [episode for episode in payload["episodes"] if bool(episode["success"])]
    if len(episodes) < num_episodes:
        raise ValueError(
            f"{task} has {len(episodes)} successful demonstrations, need {num_episodes}"
        )
    return episodes[:num_episodes]


def sample_indices(num_states: int, num_frames: int) -> np.ndarray:
    if num_states < num_frames:
        raise ValueError(
            f"Cannot select {num_frames} distinct timeline positions from only {num_states} states"
        )
    if num_frames < 2:
        raise ValueError(f"num_frames must be at least two, got {num_frames}")
    indices = np.round(np.linspace(0, num_states - 1, num_frames)).astype(int)
    if len(np.unique(indices)) != num_frames:
        raise RuntimeError(f"Sampling produced duplicate indices: {indices.tolist()}")
    return indices


def trajectory_state(states: Any, state_index: int) -> dict[str, Any]:
    return trajectory_utils.index_dict(states, state_index)


def make_env(
    task: str,
    *,
    obs_mode: str,
    control_mode: str,
    reward_mode: str,
) -> gym.Env:
    return gym.make(
        task,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        reward_mode=reward_mode,
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
    )


def render_rgb(env: Any) -> np.ndarray:
    image = env.render_rgb_array(camera_name="render_camera")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected one rendered environment, got image shape {image.shape}")
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA image, got shape {image.shape}")
    return image[..., :3].astype(np.uint8)


def render_warmed(env: Any) -> np.ndarray:
    """
    Render twice so SAPIEN refreshes its camera after restoring state.
    """
    render_rgb(env)
    return render_rgb(env)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape (height, width, 3), got {image.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not written:
        raise RuntimeError(f"Failed to write image: {path}")


def scalar_float(value: Any, context: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar tensor for {context}, got {tuple(value.shape)}")
        return float(value.item())
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar for {context}, got shape {array.shape}")
    return float(array.item())


def scalar_bool(value: Any, context: str) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar tensor for {context}, got {tuple(value.shape)}")
        return bool(value.item())
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar for {context}, got shape {array.shape}")
    return bool(array.item())


def validate_normalized_reward(reward: float, context: str) -> float:
    if not np.isfinite(reward):
        raise ValueError(f"Non-finite normalized reward for {context}: {reward}")
    if reward < 0.0 or reward > 1.0:
        raise ValueError(f"Normalized reward outside [0, 1] for {context}: {reward}")
    return reward


def replay_normalized_rewards(
    env: gym.Env,
    episode: dict[str, Any],
    trajectory: h5py.Group,
    task: str,
) -> list[float]:
    """Replay actions and return the normalized reward associated with each state.

    The returned list has shape ``(num_actions + 1,)``. Element zero is 0 by
    convention; element ``t + 1`` is the reward returned for action ``t``.
    Recorded states are restored after every step to prevent simulation drift.
    """

    env.reset(**episode["reset_kwargs"])
    env.unwrapped.set_state_dict(trajectory_state(trajectory["env_states"], 0))

    rewards = [0.0]
    for action_index, action in enumerate(trajectory["actions"]):
        _, reward_value, _, _, _ = env.step(action)
        reward = validate_normalized_reward(
            scalar_float(reward_value, f"{task} action {action_index}"),
            f"{task} action {action_index}",
        )
        rewards.append(reward)
        env.unwrapped.set_state_dict(
            trajectory_state(trajectory["env_states"], action_index + 1)
        )

    expected = len(trajectory["actions"]) + 1
    if len(rewards) != expected:
        raise RuntimeError(f"Replayed {len(rewards)} state rewards, expected {expected}")
    return rewards


def write_rollout_files(
    rollout_directory: Path,
    rewards: Sequence[float],
    metadata: dict[str, Any],
) -> None:
    rollout_directory.mkdir(parents=True, exist_ok=True)
    with (rollout_directory / "rewards.json").open("w", encoding="utf-8") as file:
        json.dump(list(rewards), file, indent=2)
    with (rollout_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def maniskill_version() -> str:
    return str(mani_skill.__version__)
