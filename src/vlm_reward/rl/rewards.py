"""
Reward providers for native-reward bootstrapping and final visual rewards.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from ..models.reward_predictor import QwenRewardHeadPredictor
from .core import Transition


Episode = List[Dict[str, Any]]


@dataclass(frozen=True)
class RewardDiagnostics:
    transition_count: int
    mean_absolute_error: float
    mean_signed_error: float
    predicted_reward_mean: float
    environment_reward_mean: float
    predicted_reward_std: float
    success_override_count: int
    inference_seconds: float


def validate_unit_reward(value: float, context: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Non-finite reward for {context}: {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Reward outside [0, 1] for {context}: {value}")
    return value


def annotate_native_reward_episodes(
    episodes: Sequence[Episode], reward_scale: float
) -> List[Transition]:
    """
    Use ManiSkill normalized-dense reward for bootstrap experiments.
    """
    if reward_scale < 0.0:
        raise ValueError(f"reward_scale must be non-negative, got {reward_scale}")

    return [
        (
            item["state"],
            item["action"],
            validate_unit_reward(
                float(item["task_reward"]),
                "ManiSkill normalized-dense reward",
            )
            * reward_scale,
            item["next_state"],
            bool(item["done"]),
        )
        for episode in episodes
        for item in episode
    ]


def annotate_visual_reward_episodes(
    episodes: Sequence[Episode],
    predictor: QwenRewardHeadPredictor,
    batch_size: int,
    reward_scale: float,
    env_success_override: bool,
) -> tuple[List[Transition], RewardDiagnostics]:
    """
    Annotate next-frame images using the frozen adapted visual reward model.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if reward_scale < 0.0:
        raise ValueError(f"reward_scale must be non-negative, got {reward_scale}")
    images = [item["next_image"] for episode in episodes for item in episode]
    if not images:
        raise ValueError("Cannot annotate an empty episode batch")
    inference_start = time.perf_counter()
    predictions = predictor.predict(images, batch_size=batch_size)
    inference_seconds = time.perf_counter() - inference_start
    if len(predictions) != len(images):
        raise ValueError(
            f"Expected {len(images)} visual reward predictions, got {len(predictions)}"
        )

    transitions: List[Transition] = []
    absolute_errors: List[float] = []
    signed_errors: List[float] = []
    used_rewards: List[float] = []
    environment_rewards: List[float] = []
    success_override_count = 0
    prediction_index = 0
    for episode in episodes:
        for item in episode:
            if env_success_override and bool(item["success"]):
                predicted_reward = 1.0
                success_override_count += 1
            else:
                raw_prediction = float(predictions[prediction_index])
                predicted_reward = validate_unit_reward(
                    raw_prediction,
                    f"visual prediction {prediction_index}",
                ) * reward_scale
            environment_reward = float(item["task_reward"])
            transitions.append(
                (
                    item["state"],
                    item["action"],
                    predicted_reward,
                    item["next_state"],
                    bool(item["done"]),
                )
            )
            absolute_errors.append(abs(predicted_reward - environment_reward))
            signed_errors.append(predicted_reward - environment_reward)
            used_rewards.append(predicted_reward)
            environment_rewards.append(environment_reward)
            prediction_index += 1

    diagnostics = RewardDiagnostics(
        transition_count=len(transitions),
        mean_absolute_error=float(np.mean(absolute_errors)),
        mean_signed_error=float(np.mean(signed_errors)),
        predicted_reward_mean=float(np.mean(used_rewards)),
        environment_reward_mean=float(np.mean(environment_rewards)),
        predicted_reward_std=float(np.std(used_rewards)),
        success_override_count=success_override_count,
        inference_seconds=inference_seconds,
    )
    throughput = len(images) / max(inference_seconds, np.finfo(float).eps)
    print(
        f"[Visual Reward] Predicted {len(images)} frames in {inference_seconds:.2f}s "
        f"({throughput:.1f} frames/s)"
    )
    print(
        "[Visual Reward] Batch diagnostics | "
        f"MAE={diagnostics.mean_absolute_error:.4f} | "
        f"signed_error={diagnostics.mean_signed_error:+.4f} | "
        f"predicted_mean={diagnostics.predicted_reward_mean:.4f} | "
        f"environment_mean={diagnostics.environment_reward_mean:.4f} | "
        f"predicted_std={diagnostics.predicted_reward_std:.4f}"
    )
    if env_success_override:
        print(
            "[Visual Reward] Applied "
            f"{diagnostics.success_override_count} environment-success override(s)"
        )
    return transitions, diagnostics
