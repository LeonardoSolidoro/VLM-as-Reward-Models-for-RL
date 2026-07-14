"""
Qwen vision-module discovery, output handling, and attention pooling.
"""
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Sequence, Tuple

import torch
import torch.nn as nn


VISION_MODULE_NAMES = ("visual", "vision_model", "vision_tower", "vision_encoder")


class AttentionalPooler(nn.Module):
    """Pool variable-length patch tokens into one embedding per image.

    The parameter names and initialization intentionally match the pooler used by
    the completed experiments so existing ``state_dict`` files remain loadable.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Return pooled embeddings.

        Args:
            patch_embeddings: Tensor shaped ``(batch, patches, embed_dim)``.

        Returns:
            Tensor shaped ``(batch, embed_dim)``.
        """

        if patch_embeddings.ndim != 3:
            raise ValueError(
                "AttentionalPooler expects (batch, patches, embed_dim), got "
                f"{tuple(patch_embeddings.shape)}"
            )
        query = self.query.expand(patch_embeddings.shape[0], -1, -1)
        pooled, _ = self.mha(
            query,
            patch_embeddings,
            patch_embeddings,
            need_weights=False,
        )
        return pooled.squeeze(1)


@dataclass(frozen=True)
class VisionModuleReference:
    """
    A vision module together with the parent needed for dynamic lookup.
    """
    parent: nn.Module
    name: str
    module: nn.Module


def find_vision_module_reference(root: nn.Module) -> VisionModuleReference:
    """
    Find the first Qwen-compatible vision encoder using breadth-first search.
    """
    queue: Deque[nn.Module] = deque([root])
    visited = set()
    while queue:
        current = queue.popleft()
        if id(current) in visited:
            continue
        visited.add(id(current))
        children = list(current.named_children())
        for name, child in children:
            if name in VISION_MODULE_NAMES:
                return VisionModuleReference(parent=current, name=name, module=child)
        queue.extend(child for _, child in children)

    top_level = [name for name, _ in root.named_children()]
    raise AttributeError(
        "No vision encoder found in the model tree. "
        f"Expected one of {VISION_MODULE_NAMES}; top-level modules are {top_level}."
    )


def find_vision_module(root: nn.Module) -> nn.Module:
    """
    Return the Qwen-compatible vision encoder contained in ``root``.
    """
    return find_vision_module_reference(root).module


def freeze_vision_encoder(root: nn.Module) -> int:
    """
    Freeze the discovered vision encoder and return its parameter count.
    """
    vision_module = find_vision_module(root)
    frozen_parameter_count = 0
    for parameter in vision_module.parameters():
        parameter.requires_grad = False
        frozen_parameter_count += parameter.numel()
    return frozen_parameter_count


def unwrap_vision_output(output: Any) -> torch.Tensor:
    """
    Extract the primary hidden-state tensor from a Hugging Face output.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        last_hidden_state = output.last_hidden_state
        if last_hidden_state is not None:
            return last_hidden_state
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported vision encoder output type: {type(output)}")


def _visual_token_counts(
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(
            "image_grid_thw must have shape (images, 3), got "
            f"{tuple(image_grid_thw.shape)}"
        )
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")

    raw_counts = torch.prod(image_grid_thw.to(dtype=torch.long), dim=1)
    merge_area = spatial_merge_size * spatial_merge_size
    if torch.any(raw_counts % merge_area != 0):
        raise ValueError(
            f"Image-grid token counts {raw_counts.tolist()} are not divisible by "
            f"spatial merge area {merge_area}"
        )
    return raw_counts, raw_counts // merge_area


def split_visual_embeddings(
    visual_embeddings: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, ...]:
    """
    Split flattened Qwen visual tokens into one tensor per input image.
    """
    image_count = int(image_grid_thw.shape[0])
    if visual_embeddings.ndim == 3:
        if visual_embeddings.shape[0] != image_count:
            raise ValueError(
                f"Vision batch size {visual_embeddings.shape[0]} does not match "
                f"image count {image_count}"
            )
        return tuple(visual_embeddings[index] for index in range(image_count))

    if visual_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D or 3D visual embeddings, got {tuple(visual_embeddings.shape)}"
        )

    raw_counts, merged_counts = _visual_token_counts(image_grid_thw, spatial_merge_size)
    token_count = int(visual_embeddings.shape[0])
    raw_total = int(raw_counts.sum().item())
    merged_total = int(merged_counts.sum().item())
    if raw_total == token_count:
        split_counts = raw_counts
    elif merged_total == token_count:
        split_counts = merged_counts
    else:
        raise ValueError(
            f"Cannot split {token_count} visual tokens across grids "
            f"{image_grid_thw.tolist()}; raw total={raw_total}, merged total={merged_total}"
        )
    return torch.split(visual_embeddings, split_counts.tolist(), dim=0)


def pool_visual_embeddings(
    visual_embeddings: torch.Tensor,
    image_grid_thw: torch.Tensor,
    pooler: AttentionalPooler,
    spatial_merge_size: int,
) -> torch.Tensor:
    """
    Pool flattened or batched Qwen visual tokens per image.
    """
    image_embeddings = split_visual_embeddings(
        visual_embeddings,
        image_grid_thw,
        spatial_merge_size,
    )
    if not image_embeddings:
        raise ValueError("Cannot pool an empty image batch")
    patch_counts = [embedding.shape[0] for embedding in image_embeddings]
    if len(set(patch_counts)) == 1:
        return pooler(torch.stack(image_embeddings, dim=0))
    return torch.cat(
        [pooler(embedding.unsqueeze(0)) for embedding in image_embeddings],
        dim=0,
    )


def raw_patch_counts(image_grid_thw: torch.Tensor) -> Sequence[int]:
    """
    Return unmerged patch counts used by the completed contrastive runs.
    """
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(
            "image_grid_thw must have shape (images, 3), got "
            f"{tuple(image_grid_thw.shape)}"
        )
    return torch.prod(image_grid_thw.to(dtype=torch.long), dim=1).tolist()
