"""
Task-reward prediction heads and checkpoint export helpers.
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class RewardHead(nn.Module):
    """
    Small MLP mapping pooled visual embeddings to rewards in ``[0, 1]``.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map ``(batch, embed_dim)`` embeddings to ``(batch, 1)`` logits.
        """
        return self.net(embeddings)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map ``(batch, embed_dim)`` embeddings to ``(batch, 1)`` rewards.
        """
        return torch.sigmoid(self.forward_logits(embeddings))


def reward_head_inference_checkpoint(training_checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a lean inference checkpoint while leaving training state untouched.
    """

    required_keys = (
        "task",
        "model_id",
        "adapter_dir",
        "input_dim",
        "hidden_dim",
        "dropout",
        "head_state_dict",
    )
    for key in required_keys:
        if key not in training_checkpoint:
            raise KeyError(f"Reward-head checkpoint is missing required key: {key}")

    inference_checkpoint = {
        key: value
        for key, value in training_checkpoint.items()
        if key != "optimizer_state_dict"
    }
    inference_checkpoint["checkpoint_format_version"] = 2
    inference_checkpoint["checkpoint_type"] = "reward_head_inference"
    return inference_checkpoint
