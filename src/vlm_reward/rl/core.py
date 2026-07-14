"""CrossQ networks and replay buffer shared by both RL experiments.

These components are extracted from
``src/historical_progress_reward/cleanrl_crossq.py``, the original
progress-as-reward trial contributed by a project collaborator.  Keeping them
here lets that trial and the final direct visual-reward trainer use identical
algorithm code while the original experiment remains a runnable entry point.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..runtime import set_global_seed


Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Compatibility name for the project's shared reproducibility setup.
    """
    set_global_seed(seed, deterministic)


class BatchRenorm1d(nn.Module):
    """Batch renormalization used in the CrossQ critic.

    Input and output shape: ``(batch, features)``.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.01,
        warmup_steps: int = 10_000,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Features)
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected shape (batch, {self.num_features}), got {tuple(x.shape)}"
            )
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            if self.num_batches_tracked.item() == 0:
                self.running_mean.copy_(batch_mean.detach())
                self.running_var.copy_(batch_var.detach())

            step = self.num_batches_tracked.item()
            r_max = 1.0 if step < self.warmup_steps else 3.0
            d_max = 0.0 if step < self.warmup_steps else 5.0
            r = torch.clamp(
                torch.sqrt(batch_var.detach() + self.eps)
                / torch.sqrt(self.running_var + self.eps),
                1.0 / r_max,
                r_max,
            )
            d = torch.clamp(
                (batch_mean.detach() - self.running_mean)
                / torch.sqrt(self.running_var + self.eps),
                -d_max,
                d_max,
            )
            x_norm = ((x - batch_mean) / torch.sqrt(batch_var + self.eps)) * r + d
            self.running_mean.add_(
                self.momentum * (batch_mean.detach() - self.running_mean)
            )
            self.running_var.add_(
                self.momentum * (batch_var.detach() - self.running_var)
            )
            self.num_batches_tracked.add_(1)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.weight * x_norm + self.bias


class PolicyNetwork(nn.Module):
    """
    Squashed-Gaussian SAC actor.
    """
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state shape: (Batch, StateDim)
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = torch.tanh(self.log_std_linear(x))
        log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (log_std + 1.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state shape: (Batch, StateDim)
        mean, log_std = self.forward(state)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.max_action
        log_prob = normal.log_prob(pre_tanh)
        log_prob -= torch.log(self.max_action * (1.0 - squashed.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)


class QNetwork(nn.Module):
    """
    CrossQ double critic with BatchRenorm in each hidden layer.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 2048,
        warmup_steps: int = 10_000,
    ) -> None:
        super().__init__()
        self.q1 = self._build_branch(
            state_dim + action_dim, hidden_dim, warmup_steps
        )
        self.q2 = self._build_branch(
            state_dim + action_dim, hidden_dim, warmup_steps
        )

    @staticmethod
    def _build_branch(
        input_dim: int, hidden_dim: int, warmup_steps: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            BatchRenorm1d(hidden_dim, warmup_steps=warmup_steps),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            BatchRenorm1d(hidden_dim, warmup_steps=warmup_steps),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # state shape: (Batch, StateDim); action shape: (Batch, ActionDim)
        state_action = torch.cat([state, action], dim=1)
        return self.q1(state_action), self.q2(state_action)


class SimpleReplayBuffer:
    """
    In-memory replay buffer retained for checkpoint compatibility.
    """
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}"
            )
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def extend(self, transitions: Iterable[Transition]) -> None:
        self.buffer.extend(transitions)

    def __len__(self) -> int:
        return len(self.buffer)

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "transitions": list(self.buffer)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError(
                f"Replay capacity changed: saved={state['capacity']}, current={self.capacity}"
            )
        self.buffer.clear()
        self.buffer.extend(state["transitions"])
