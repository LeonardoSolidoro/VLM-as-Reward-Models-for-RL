"""
Portable, fail-fast checkpoint helpers for CrossQ experiments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch

from .core import PolicyNetwork


CHECKPOINT_VERSION = 2


def atomic_torch_save(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    """
    Write a checkpoint completely before replacing its destination.
    """
    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(
            f"Checkpoint parent directory does not exist: {destination.parent}"
        )
    temporary = destination.with_name(f"{destination.name}.tmp")
    torch.save(dict(checkpoint), temporary)
    os.replace(temporary, destination)


def atomic_json_dump(value: Any, path: str | Path) -> None:
    """
    Write machine-readable JSON without exposing a partially written file.
    """
    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(
            f"JSON parent directory does not exist: {destination.parent}"
        )
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def build_actor_export(
    actor: torch.nn.Module,
    log_alpha: torch.Tensor | None,
    args: argparse.Namespace,
    state_dim: int,
    action_dim: int,
    max_action: float,
    global_step: int,
    success_rate: float,
    average_normalized_dense_return: float,
) -> Dict[str, Any]:
    """
    Build a compact inference checkpoint without critic or replay state.
    """
    checkpoint: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "actor_export",
        "task": args.task,
        "actor_state_dict": actor.state_dict(),
        "actor_hidden_dim": args.actor_hidden_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "global_step": global_step,
        "success_rate": success_rate,
        # Retained for compatibility with existing actor exports.
        "avg_reward": average_normalized_dense_return,
        "average_normalized_dense_return": average_normalized_dense_return,
        "seed": args.seed,
        "fixed_alpha": args.alpha,
    }
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    return checkpoint


def _require_checkpoint_keys(
    checkpoint: Mapping[str, Any], required_keys: Iterable[str], checkpoint_path: Path
) -> None:
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing required metadata: {missing}"
        )


def load_actor_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[PolicyNetwork, Dict[str, Any]]:
    """
    Load a compact actor export or compatible full training checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dictionary at {path}, got {type(checkpoint).__name__}"
        )
    required = (
        "task",
        "state_dim",
        "action_dim",
        "max_action",
        "actor_hidden_dim",
        "actor_state_dict",
        "checkpoint_type",
        "global_step",
    )
    _require_checkpoint_keys(checkpoint, required, path)
    if checkpoint["checkpoint_type"] not in {"actor_export", "full_training_state"}:
        raise ValueError(
            f"Unsupported checkpoint type {checkpoint['checkpoint_type']!r} at {path}"
        )
    actor = PolicyNetwork(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        max_action=float(checkpoint["max_action"]),
        hidden_dim=int(checkpoint["actor_hidden_dim"]),
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    metadata = {
        "task": checkpoint["task"],
        "checkpoint_type": checkpoint["checkpoint_type"],
        "global_step": int(checkpoint["global_step"]),
        "state_dim": int(checkpoint["state_dim"]),
        "action_dim": int(checkpoint["action_dim"]),
        "max_action": float(checkpoint["max_action"]),
        "actor_hidden_dim": int(checkpoint["actor_hidden_dim"]),
    }
    return actor, metadata


def validate_actor_for_environment(
    metadata: Mapping[str, Any],
    task: str,
    state_dim: int,
    action_dim: int,
    max_action: float,
) -> None:
    if metadata["task"] != task:
        raise ValueError(
            f"Actor task {metadata['task']} does not match environment task {task}"
        )
    if metadata["state_dim"] != state_dim:
        raise ValueError(
            f"Actor state dimension {metadata['state_dim']} does not match {state_dim}"
        )
    if metadata["action_dim"] != action_dim:
        raise ValueError(
            f"Actor action dimension {metadata['action_dim']} does not match {action_dim}"
        )
    if not np.isclose(metadata["max_action"], max_action):
        raise ValueError(
            f"Actor max action {metadata['max_action']} does not match {max_action}"
        )


def load_training_checkpoint(path: str | Path, task: str) -> Dict[str, Any]:
    """
    Load and validate a resumable full-state checkpoint.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dictionary, got {type(checkpoint).__name__}"
        )
    _require_checkpoint_keys(
        checkpoint,
        (
            "checkpoint_version",
            "checkpoint_type",
            "task",
            "args",
            "actor_state_dict",
            "critic_state_dict",
            "buffer",
            "current_episode",
        ),
        checkpoint_path,
    )
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {checkpoint['checkpoint_version']}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    if checkpoint["checkpoint_type"] != "full_training_state":
        raise ValueError(
            f"Resume requires full_training_state, got {checkpoint['checkpoint_type']}"
        )
    if checkpoint["task"] != task:
        raise ValueError(
            f"Resume checkpoint task {checkpoint['task']} does not match {task}"
        )
    if checkpoint["current_episode"]:
        raise ValueError(
            "Resume checkpoint contains a partial environment episode and is unsafe"
        )
    return checkpoint


def validate_resume_arguments(
    saved_arguments: Mapping[str, Any],
    requested_arguments: Mapping[str, Any],
    immutable_names: Iterable[str],
) -> None:
    """
    Reject resumed runs whose objective or optimizer configuration changed.
    """
    for name in immutable_names:
        if name not in saved_arguments:
            raise KeyError(f"Resume checkpoint is missing argument {name!r}")
        if name not in requested_arguments:
            raise KeyError(f"Current command is missing resume argument {name!r}")
        if saved_arguments[name] != requested_arguments[name]:
            raise ValueError(
                f"Resume argument --{name.replace('_', '-')} changed: "
                f"saved={saved_arguments[name]!r}, requested={requested_arguments[name]!r}"
            )
