"""
CrossQ/SAC optimization isolated from rollout and reward annotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .core import PolicyNetwork, QNetwork, SimpleReplayBuffer


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} became non-finite: {value.detach()}")


def _require_finite_gradients(name: str, module: torch.nn.Module) -> None:
    for parameter_name, parameter in module.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"{name} gradient became non-finite for {parameter_name}"
            )


def diagonal_gaussian_kl(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_log_std: torch.Tensor,
) -> torch.Tensor:
    """
    Mean KL(current || reference) for diagonal pre-squash Gaussians.
    """
    if not (
        mean.shape == log_std.shape == reference_mean.shape == reference_log_std.shape
    ):
        raise ValueError(
            "Policy KL tensor shapes must match, got "
            f"mean={tuple(mean.shape)}, log_std={tuple(log_std.shape)}, "
            f"reference_mean={tuple(reference_mean.shape)}, "
            f"reference_log_std={tuple(reference_log_std.shape)}"
        )
    variance_ratio = torch.exp(2.0 * (log_std - reference_log_std))
    squared_mean_distance = (mean - reference_mean).pow(2) * torch.exp(
        -2.0 * reference_log_std
    )
    kl_per_dimension = (
        reference_log_std
        - log_std
        + 0.5 * (variance_ratio + squared_mean_distance - 1.0)
    )
    return kl_per_dimension.sum(dim=1).mean()


@dataclass
class UpdateMetrics:
    total_updates: int = 0
    actor_updates_started: bool = False
    critic_loss: float = float("nan")
    sac_actor_loss: float = float("nan")
    reference_kl: float = float("nan")
    total_actor_loss: float = float("nan")
    reference_action_mse: float = float("nan")
    average_batch_reward: float = float("nan")
    alpha: float = float("nan")


class CrossQUpdater:
    """
    Own optimizers and perform replay updates for a CrossQ actor/critic pair.
    """
    def __init__(
        self,
        actor: PolicyNetwork,
        critic: QNetwork,
        reference_actor: Optional[PolicyNetwork],
        device: torch.device,
        batch_size: int,
        gamma: float,
        bootstrap_at_done: str,
        policy_delay: int,
        critic_only_warmup_updates: int,
        reference_kl_weight: float,
        actor_max_grad_norm: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        adam_beta1: float,
        fixed_alpha: Optional[float],
        target_entropy: Optional[float],
        initial_log_alpha: float = 0.0,
    ) -> None:
        if bootstrap_at_done not in {"always", "never"}:
            raise ValueError(f"Invalid bootstrap_at_done: {bootstrap_at_done}")
        self.actor = actor
        self.critic = critic
        self.reference_actor = reference_actor
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.bootstrap_at_done = bootstrap_at_done
        self.policy_delay = policy_delay
        self.critic_only_warmup_updates = critic_only_warmup_updates
        self.reference_kl_weight = reference_kl_weight
        self.actor_max_grad_norm = actor_max_grad_norm
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=actor_learning_rate, betas=(adam_beta1, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=critic_learning_rate, betas=(adam_beta1, 0.999)
        )
        self.fixed_alpha = fixed_alpha
        if fixed_alpha is None:
            if target_entropy is None:
                raise ValueError("Automatic temperature tuning requires target_entropy")
            self.target_entropy = target_entropy
            self.log_alpha: Optional[torch.Tensor] = torch.tensor(
                [initial_log_alpha],
                dtype=torch.float32,
                requires_grad=True,
                device=device,
            )
            self.alpha_optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam(
                [self.log_alpha], lr=critic_learning_rate, betas=(adam_beta1, 0.999)
            )
            self.alpha_tensor: Optional[torch.Tensor] = None
        else:
            if fixed_alpha <= 0.0:
                raise ValueError(f"fixed_alpha must be positive, got {fixed_alpha}")
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha_tensor = torch.tensor(
                fixed_alpha, dtype=torch.float32, device=device
            )
        self.metrics = UpdateMetrics()

    @property
    def current_alpha(self) -> torch.Tensor:
        if self.log_alpha is not None:
            return self.log_alpha.exp()
        if self.alpha_tensor is None:
            raise RuntimeError("CrossQ temperature was not initialized")
        return self.alpha_tensor

    def update(self, replay_buffer: SimpleReplayBuffer, number_of_updates: int) -> None:
        if number_of_updates < 0:
            raise ValueError(
                f"number_of_updates must be non-negative, got {number_of_updates}"
            )
        for _ in range(number_of_updates):
            states, actions, rewards, next_states, dones = replay_buffer.sample(
                self.batch_size
            )
            state_batch = torch.as_tensor(
                states, dtype=torch.float32, device=self.device
            )
            action_batch = torch.as_tensor(
                actions, dtype=torch.float32, device=self.device
            )
            reward_batch = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device
            ).view(-1, 1)
            next_state_batch = torch.as_tensor(
                next_states, dtype=torch.float32, device=self.device
            )
            done_batch = torch.as_tensor(
                dones, dtype=torch.float32, device=self.device
            ).view(-1, 1)
            alpha = self.current_alpha

            with torch.no_grad():
                next_action, next_log_probability = self.actor.sample(next_state_batch)
            concatenated_states = torch.cat(
                [state_batch, next_state_batch], dim=0
            )
            concatenated_actions = torch.cat(
                [action_batch, next_action], dim=0
            )
            all_q1, all_q2 = self.critic(
                concatenated_states, concatenated_actions
            )
            current_q1, next_q1 = torch.split(all_q1, self.batch_size)
            current_q2, next_q2 = torch.split(all_q2, self.batch_size)
            next_q_minimum = torch.min(next_q1, next_q2)
            bootstrapped_q = (
                next_q_minimum - alpha.detach() * next_log_probability
            ).detach()
            if self.bootstrap_at_done == "always":
                target_q = reward_batch + self.gamma * bootstrapped_q
            else:
                target_q = (
                    reward_batch
                    + (1.0 - done_batch) * self.gamma * bootstrapped_q
                )

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2, target_q
            )
            _require_finite("critic loss", critic_loss)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            _require_finite_gradients("critic", self.critic)
            self.critic_optimizer.step()
            self.metrics.total_updates += 1
            self.metrics.critic_loss = float(critic_loss.detach().item())
            self.metrics.average_batch_reward = float(np.mean(rewards))
            self.metrics.alpha = float(alpha.detach().item())

            if (
                self.metrics.total_updates > self.critic_only_warmup_updates
                and self.metrics.total_updates % self.policy_delay == 0
            ):
                self._update_actor_and_temperature(state_batch)

    def _update_actor_and_temperature(self, state_batch: torch.Tensor) -> None:
        for parameter in self.critic.parameters():
            parameter.requires_grad = False
        try:
            policy_action, policy_log_probability = self.actor.sample(state_batch)
            self.critic.eval()
            policy_q1, policy_q2 = self.critic(state_batch, policy_action)
            self.critic.train()
            sac_actor_loss = (
                self.current_alpha.detach() * policy_log_probability
                - torch.min(policy_q1, policy_q2)
            ).mean()

            if self.reference_actor is not None:
                policy_mean, policy_log_std = self.actor(state_batch)
                with torch.no_grad():
                    reference_mean, reference_log_std = self.reference_actor(state_batch)
                reference_kl = torch.clamp(
                    diagonal_gaussian_kl(
                        policy_mean,
                        policy_log_std,
                        reference_mean,
                        reference_log_std,
                    ),
                    min=0.0,
                )
                policy_mean_action = torch.tanh(policy_mean) * self.actor.max_action
                reference_mean_action = (
                    torch.tanh(reference_mean) * self.reference_actor.max_action
                )
                reference_action_mse = F.mse_loss(
                    policy_mean_action, reference_mean_action
                )
            else:
                reference_kl = torch.zeros((), device=self.device)
                reference_action_mse = torch.zeros((), device=self.device)

            actor_loss = sac_actor_loss + self.reference_kl_weight * reference_kl
            _require_finite("actor loss", actor_loss)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            _require_finite_gradients("actor", self.actor)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_max_grad_norm
            )
            _require_finite("actor gradient norm", gradient_norm)
            self.actor_optimizer.step()
            self.metrics.actor_updates_started = True
            self.metrics.sac_actor_loss = float(sac_actor_loss.detach().item())
            self.metrics.reference_kl = float(reference_kl.detach().item())
            self.metrics.total_actor_loss = float(actor_loss.detach().item())
            self.metrics.reference_action_mse = float(
                reference_action_mse.detach().item()
            )

            if self.log_alpha is not None:
                if self.alpha_optimizer is None or self.target_entropy is None:
                    raise RuntimeError("Automatic temperature optimizer is missing")
                alpha_loss = -(
                    self.log_alpha.exp()
                    * (policy_log_probability + self.target_entropy).detach()
                ).mean()
                _require_finite("temperature loss", alpha_loss)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.log_alpha.grad is None:
                    raise RuntimeError("Automatic temperature has no gradient")
                _require_finite("temperature gradient", self.log_alpha.grad)
                self.alpha_optimizer.step()
        finally:
            self.critic.train()
            for parameter in self.critic.parameters():
                parameter.requires_grad = True

    def restore_metrics(self, checkpoint: dict[str, object]) -> None:
        self.metrics.total_updates = int(checkpoint["total_training_updates"])
        self.metrics.actor_updates_started = bool(checkpoint["actor_updates_started"])
        self.metrics.sac_actor_loss = float(checkpoint["last_sac_actor_loss"])
        self.metrics.reference_kl = float(checkpoint["last_reference_kl"])
        self.metrics.total_actor_loss = float(checkpoint["last_total_actor_loss"])
        self.metrics.reference_action_mse = float(
            checkpoint["last_reference_action_mse"]
        )

    def restore_temperature(self, checkpoint: dict[str, object]) -> None:
        if self.log_alpha is None:
            return
        saved_log_alpha = checkpoint["log_alpha"]
        if not isinstance(saved_log_alpha, torch.Tensor):
            raise TypeError("Expected tensor log_alpha in automatic-temperature checkpoint")
        with torch.no_grad():
            value = saved_log_alpha.detach().reshape(-1)[0].to(self.device)
            self.log_alpha.copy_(value.reshape_as(self.log_alpha))
        if self.alpha_optimizer is None:
            raise RuntimeError("Automatic temperature optimizer is missing")
        optimizer_state = checkpoint["alpha_optimizer_state_dict"]
        if not isinstance(optimizer_state, dict):
            raise TypeError("Expected alpha optimizer state dictionary")
        self.alpha_optimizer.load_state_dict(optimizer_state)
