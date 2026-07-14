"""
Shared contrastive wrapper used by progress and reward fine-tuning.
"""
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import (
    AttentionalPooler,
    find_vision_module_reference,
    raw_patch_counts,
    unwrap_vision_output,
)


class ContrastiveQwenModel(nn.Module):
    """Add cross-view InfoNCE learning to a Qwen image-text model.

    The centering, normalization, temporal mask, and loss calculation match the
    successful progress/reward contrastive experiments.
    """

    def __init__(
        self,
        base_model: nn.Module,
        contrastive_weight: float = 1.0,
        temporal_margin: int = 4,
    ) -> None:
        super().__init__()
        if contrastive_weight < 0.0:
            raise ValueError(f"contrastive_weight must be non-negative, got {contrastive_weight}")
        if temporal_margin < 0:
            raise ValueError(f"temporal_margin must be non-negative, got {temporal_margin}")

        self.model = base_model
        self.contrastive_weight = contrastive_weight
        self.temporal_margin = temporal_margin
        embed_dim = int(self.model.config.vision_config.hidden_size)
        self.pooler = AttentionalPooler(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        reference = find_vision_module_reference(self.model)
        self.vision_parent = reference.parent
        self.vision_module_name = reference.name

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_positive: Optional[torch.Tensor] = None,
        image_grid_thw_positive: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        trajectory_indices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if pixel_values is not None and pixel_values.dtype.is_floating_point:
            pixel_values.requires_grad_(True)
        if pixel_values_positive is not None and pixel_values_positive.dtype.is_floating_point:
            pixel_values_positive.requires_grad_(True)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        ce_loss = outputs.loss
        if ce_loss is not None and not torch.isfinite(ce_loss):
            raise FloatingPointError(
                "Qwen returned a non-finite supervised CE loss. Verify that every "
                "sample contains assistant target tokens and finite model activations."
            )

        if pixel_values is None or pixel_values_positive is None:
            contrastive_loss = self._dummy_contrastive_loss()
        else:
            if image_grid_thw is None or image_grid_thw_positive is None:
                raise ValueError("Both anchor and positive image_grid_thw tensors are required")
            contrastive_loss = self.compute_contrastive_loss(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_positive=pixel_values_positive,
                image_grid_thw_positive=image_grid_thw_positive,
                frame_indices=frame_indices,
                trajectory_indices=trajectory_indices,
            )

        if ce_loss is not None:
            contrastive_loss = contrastive_loss.to(device=ce_loss.device)
            total_loss = ce_loss + self.contrastive_weight * contrastive_loss
        else:
            total_loss = self.contrastive_weight * contrastive_loss
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "contrastive_loss": contrastive_loss,
            "logits": outputs.logits,
        }

    def _dummy_contrastive_loss(self) -> torch.Tensor:
        dummy_loss = sum(parameter.sum() for parameter in self.pooler.parameters()) * 0.0
        return dummy_loss + self.logit_scale * 0.0

    def compute_contrastive_loss(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        pixel_values_positive: torch.Tensor,
        image_grid_thw_positive: torch.Tensor,
        frame_indices: Optional[torch.Tensor],
        trajectory_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        vision_module = getattr(self.vision_parent, self.vision_module_name)
        combined_pixel_values = torch.cat([pixel_values, pixel_values_positive], dim=0)
        combined_grid_thw = torch.cat([image_grid_thw, image_grid_thw_positive], dim=0)
        combined_embeddings = unwrap_vision_output(
            vision_module(combined_pixel_values, grid_thw=combined_grid_thw)
        )

        anchor_embeddings, positive_embeddings = torch.split(
            combined_embeddings,
            [pixel_values.shape[0], pixel_values_positive.shape[0]],
        )
        pooler_device = self.pooler.query.device
        pooler_dtype = self.pooler.query.dtype
        anchor_embeddings = anchor_embeddings.to(device=pooler_device, dtype=pooler_dtype)
        positive_embeddings = positive_embeddings.to(device=pooler_device, dtype=pooler_dtype)

        anchor_splits = torch.split(anchor_embeddings, raw_patch_counts(image_grid_thw))
        positive_splits = torch.split(
            positive_embeddings,
            raw_patch_counts(image_grid_thw_positive),
        )
        if len(anchor_splits) != len(positive_splits):
            raise ValueError(
                f"Anchor image count {len(anchor_splits)} does not match positive "
                f"image count {len(positive_splits)}"
            )
        if not anchor_splits:
            return self._dummy_contrastive_loss()

        anchor_pooled = torch.cat(
            [self.pooler(embedding.unsqueeze(0)) for embedding in anchor_splits],
            dim=0,
        )
        positive_pooled = torch.cat(
            [self.pooler(embedding.unsqueeze(0)) for embedding in positive_splits],
            dim=0,
        )

        anchor_pooled = F.normalize(
            anchor_pooled - anchor_pooled.mean(dim=0, keepdim=True),
            dim=-1,
        )
        positive_pooled = F.normalize(
            positive_pooled - positive_pooled.mean(dim=0, keepdim=True),
            dim=-1,
        )

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_anchor = logit_scale * anchor_pooled @ positive_pooled.t()
        logits_per_positive = logit_scale * positive_pooled @ anchor_pooled.t()

        if frame_indices is not None or trajectory_indices is not None:
            if frame_indices is None or trajectory_indices is None:
                raise ValueError("frame_indices and trajectory_indices must be provided together")
            frame_indices = frame_indices.to(pooler_device)
            trajectory_indices = trajectory_indices.to(pooler_device)
            if frame_indices.shape != trajectory_indices.shape:
                raise ValueError(
                    f"frame_indices shape {tuple(frame_indices.shape)} does not match "
                    f"trajectory_indices shape {tuple(trajectory_indices.shape)}"
                )
            if frame_indices.numel() != anchor_pooled.shape[0]:
                raise ValueError(
                    f"Received {frame_indices.numel()} frame indices for "
                    f"{anchor_pooled.shape[0]} images"
                )
            same_trajectory = trajectory_indices.unsqueeze(1) == trajectory_indices.unsqueeze(0)
            close_in_time = (
                torch.abs(frame_indices.unsqueeze(1) - frame_indices.unsqueeze(0))
                <= self.temporal_margin
            )
            false_negative_mask = same_trajectory & close_in_time
            false_negative_mask.fill_diagonal_(False)
            logits_per_anchor = logits_per_anchor.masked_fill(false_negative_mask, -1e9)
            logits_per_positive = logits_per_positive.masked_fill(false_negative_mask, -1e9)

        contrastive_labels = torch.arange(
            anchor_pooled.shape[0],
            device=pooler_device,
            dtype=torch.long,
        )
        return (
            F.cross_entropy(logits_per_anchor, contrastive_labels)
            + F.cross_entropy(logits_per_positive, contrastive_labels)
        ) / 2

    def generate(self, **kwargs: Any) -> Any:
        for key in (
            "pixel_values_positive",
            "image_grid_thw_positive",
            "frame_indices",
            "trajectory_indices",
        ):
            if key in kwargs:
                kwargs.pop(key)
        return self.model.generate(**kwargs)

    def print_trainable_parameters(self) -> None:
        self.model.print_trainable_parameters()

    def enable_input_require_grads(self, **kwargs: Any) -> None:
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs: Any) -> None:
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable(**kwargs)

    def get_input_embeddings(self) -> Any:
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        return None

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Keep Hugging Face Trainer checkpoints limited to trainable parameters.
        """
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        if "state_dict" in kwargs:
            kwargs.pop("state_dict")
        if "safe_serialization" in kwargs:
            kwargs.pop("safe_serialization")
        self.model.save_pretrained(save_directory, **kwargs)
        extras_path = Path(save_directory) / "contrastive_extras.pt"
        torch.save(
            {
                "pooler": self.pooler.state_dict(),
                "logit_scale": self.logit_scale.data,
            },
            extras_path,
        )
