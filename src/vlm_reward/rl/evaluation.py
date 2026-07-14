"""
Deterministic actor evaluation on fixed ManiSkill seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
import numpy as np
import torch
import torch.nn as nn

from .environments import state_to_numpy, success_to_bool


@dataclass(frozen=True)
class EvaluationResult:
    success_rate: float
    average_normalized_dense_return: float


def evaluate_actor(
    actor: nn.Module,
    task: str,
    device: torch.device,
    num_episodes: int = 10,
    base_seed: Optional[int] = None,
) -> EvaluationResult:
    """
    Evaluate deterministic mean actions, stopping each episode on success.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    was_training = actor.training
    actor.eval()
    env = gym.make(
        task,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode=None,
        reward_mode="normalized_dense",
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
    )
    successes = 0
    total_return = 0.0
    try:
        for episode_index in range(num_episodes):
            episode_seed = None if base_seed is None else base_seed + episode_index
            observation, _ = env.reset(seed=episode_seed)
            done = False
            episode_return = 0.0
            while not done:
                flat_observation = (
                    observation["state"]
                    if isinstance(observation, dict)
                    else observation
                )
                state = torch.as_tensor(
                    state_to_numpy(flat_observation),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    mean, _ = actor(state)
                    action = (
                        torch.tanh(mean).cpu().numpy().reshape(-1) * actor.max_action
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                if isinstance(reward, torch.Tensor):
                    episode_return += float(reward.detach().item())
                else:
                    episode_return += float(reward)
                done = success_to_bool(terminated) or success_to_bool(truncated)
                if success_to_bool(info["success"]):
                    successes += 1
                    break
            total_return += episode_return
    finally:
        env.close()
        actor.train(was_training)
    return EvaluationResult(
        success_rate=successes / num_episodes,
        average_normalized_dense_return=total_return / num_episodes,
    )


def evaluate_policy(
    actor: nn.Module,
    task: str,
    device: torch.device,
    num_episodes: int = 10,
    base_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compatibility wrapper for the original two-value evaluation API.
    """
    result = evaluate_actor(actor, task, device, num_episodes, base_seed)
    return result.success_rate, result.average_normalized_dense_return
