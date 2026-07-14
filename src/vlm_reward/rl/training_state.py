"""
Serializable state for safe, episode-boundary CrossQ resumption.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .checkpoints import CHECKPOINT_VERSION
from .core import PolicyNetwork, QNetwork, SimpleReplayBuffer
from .updates import CrossQUpdater


@dataclass
class TrainingProgress:
    global_step: int = 0
    total_episodes_completed: int = 0
    best_success_rate: float = 0.0
    best_evaluation_return: float = float("-inf")
    consecutive_target_evaluations: int = 0
    last_evaluation_step: int = 0
    latest_success_rate: float = 0.0
    latest_evaluation_return: float = float("-inf")
    evaluation_history: List[Dict[str, float]] = field(default_factory=list)

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> "TrainingProgress":
        history: List[Dict[str, float]] = []
        for record in checkpoint["eval_history"]:
            if "average_normalized_dense_return" in record:
                average_return = float(record["average_normalized_dense_return"])
            else:
                # Version-2 checkpoints used ``avg_reward`` for this metric.
                average_return = float(record["avg_reward"])
            history.append(
                {
                    "global_step": float(record["global_step"]),
                    "success_rate": float(record["success_rate"]),
                    "average_normalized_dense_return": average_return,
                }
            )
        return cls(
            global_step=int(checkpoint["global_step"]),
            total_episodes_completed=int(checkpoint["total_episodes_completed"]),
            best_success_rate=float(checkpoint["best_success_rate"]),
            best_evaluation_return=float(checkpoint["best_eval_reward"]),
            consecutive_target_evaluations=int(checkpoint["consecutive_successes"]),
            last_evaluation_step=int(checkpoint["last_eval_step"]),
            latest_success_rate=float(checkpoint["latest_success_rate"]),
            latest_evaluation_return=float(checkpoint["latest_eval_reward"]),
            evaluation_history=history,
        )

    def append_evaluation(
        self, global_step: int, success_rate: float, average_return: float
    ) -> None:
        self.evaluation_history.append(
            {
                "global_step": float(global_step),
                "success_rate": success_rate,
                "average_normalized_dense_return": average_return,
            }
        )


def build_training_checkpoint(
    args: argparse.Namespace,
    actor: PolicyNetwork,
    critic: QNetwork,
    reference_actor: Optional[PolicyNetwork],
    updater: CrossQUpdater,
    replay_buffer: SimpleReplayBuffer,
    progress: TrainingProgress,
    state_dim: int,
    action_dim: int,
    max_action: float,
    batch_renorm_warmup_steps: int,
    current_episode: List[Dict[str, Any]],
    unannotated_episodes: List[List[Dict[str, Any]]],
    initialization_checkpoint_path: Optional[str],
) -> Dict[str, Any]:
    """Build the single rolling full-state checkpoint.

    Callers should persist this only when ``current_episode`` is empty.  Compact
    actor exports, not copies of this replay-bearing state, are used for best and
    final checkpoints.
    """

    metrics = updater.metrics
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "full_training_state",
        "task": args.task,
        "args": vars(args).copy(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "actor_hidden_dim": args.actor_hidden_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        "effective_batch_renorm_warmup_steps": batch_renorm_warmup_steps,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "reference_actor_state_dict": (
            None if reference_actor is None else reference_actor.state_dict()
        ),
        "actor_optimizer_state_dict": updater.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": updater.critic_optimizer.state_dict(),
        "alpha_optimizer_state_dict": (
            None
            if updater.alpha_optimizer is None
            else updater.alpha_optimizer.state_dict()
        ),
        "log_alpha": (
            None if updater.log_alpha is None else updater.log_alpha.detach().cpu()
        ),
        "fixed_alpha": args.alpha,
        "global_step": progress.global_step,
        "total_training_updates": metrics.total_updates,
        "total_episodes_completed": progress.total_episodes_completed,
        "latest_success_rate": progress.latest_success_rate,
        "latest_eval_reward": progress.latest_evaluation_return,
        "best_success_rate": progress.best_success_rate,
        "best_eval_reward": progress.best_evaluation_return,
        "consecutive_successes": progress.consecutive_target_evaluations,
        "last_eval_step": progress.last_evaluation_step,
        "last_sac_actor_loss": metrics.sac_actor_loss,
        "last_reference_kl": metrics.reference_kl,
        "last_total_actor_loss": metrics.total_actor_loss,
        "last_reference_action_mse": metrics.reference_action_mse,
        "actor_updates_started": metrics.actor_updates_started,
        # Keep the version-2 representation for compatibility with completed runs.
        "buffer": replay_buffer.buffer,
        "current_episode": current_episode,
        "unannotated_episodes": unannotated_episodes,
        "eval_history": progress.evaluation_history,
        "initialization_checkpoint_path": initialization_checkpoint_path,
        "reference_kl_weight": args.reference_kl_weight,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
