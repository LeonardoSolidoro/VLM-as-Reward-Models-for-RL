"""
ManiSkill environment and rendering helpers shared by RL commands.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..data.cameras import task_camera_target, update_wrist_mounted_camera


def state_to_numpy(state: Any) -> np.ndarray:
    """
    Convert a scalar-environment state observation to a flat float array.
    """
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    return np.asarray(state, dtype=np.float32).reshape(-1)


def success_to_bool(success: Any) -> bool:
    """
    Convert ManiSkill's scalar/tensor success flag to ``bool``.
    """
    if isinstance(success, torch.Tensor):
        return bool(success.detach().any().item())
    if isinstance(success, np.ndarray):
        return bool(np.any(success))
    return bool(success)


def get_task_camera_target(env: Any, task: str) -> np.ndarray:
    """
    Compatibility name for the shared task camera target.
    """
    return task_camera_target(env, task)


def update_wrist_follow_camera(env: Any, task: str) -> None:
    """
    Compatibility name for the shared wrist-mounted camera policy.
    """
    update_wrist_mounted_camera(env, task)


def get_state_dict(
    env: Any,
    observation: Any,
    task: str,
    use_moving_mounted_camera: bool = False,
) -> Dict[str, Any]:
    """
    Return flat state plus an RGB image for reward annotation.
    """
    if isinstance(observation, dict) and "image" in observation:
        return observation
    if use_moving_mounted_camera:
        update_wrist_follow_camera(env.unwrapped, task)
    if hasattr(env.unwrapped, "render_rgb_array"):
        image = env.unwrapped.render_rgb_array(camera_name="render_camera")
    else:
        image = env.render()
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    return {"state": observation, "image": image.astype(np.uint8)}


def get_step_state(
    env: Any,
    observation: Any,
    task: str,
    moving_camera: bool,
    need_image: bool,
) -> Dict[str, Any]:
    if need_image:
        return get_state_dict(env, observation, task, moving_camera)
    return {"state": observation, "image": None}
