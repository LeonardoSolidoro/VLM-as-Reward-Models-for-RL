"""
Shared model components for progress estimation and learned rewards.
"""
from .reward import RewardHead
from .vision import (
    AttentionalPooler,
    VisionModuleReference,
    find_vision_module,
    find_vision_module_reference,
    pool_visual_embeddings,
    split_visual_embeddings,
    unwrap_vision_output,
)

__all__ = [
    "AttentionalPooler",
    "RewardHead",
    "VisionModuleReference",
    "find_vision_module",
    "find_vision_module_reference",
    "pool_visual_embeddings",
    "split_visual_embeddings",
    "unwrap_vision_output",
]
