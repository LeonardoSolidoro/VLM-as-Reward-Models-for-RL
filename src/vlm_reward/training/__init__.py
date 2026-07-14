"""
Shared fine-tuning data and training utilities.
"""
from .collation import QwenMultimodalCollator
from .multimodal import (
    build_messages,
    build_supervised_model_inputs,
    build_user_content,
    load_rgb_image,
    squeeze_text_batch_dim,
)

__all__ = [
    "QwenMultimodalCollator",
    "build_messages",
    "build_supervised_model_inputs",
    "build_user_content",
    "load_rgb_image",
    "squeeze_text_batch_dim",
]
