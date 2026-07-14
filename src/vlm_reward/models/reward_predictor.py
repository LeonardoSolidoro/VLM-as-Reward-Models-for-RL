"""
Frozen adapted Qwen vision encoder plus task reward-head inference.
å"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from .checkpoints import resolve_checkpoint_artifact
from .reward import RewardHead
from .vision import (
    AttentionalPooler,
    find_vision_module,
    pool_visual_embeddings,
    unwrap_vision_output,
)


def load_frozen_visual_components(
    model_id: str,
    adapter_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, nn.Module, AttentionalPooler, int]:
    """
    Load and freeze the adapted Qwen vision stack used by reward heads.
    """

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    processor = AutoProcessor.from_pretrained(model_id)
    model_kwargs: Dict[str, Any] = {
        "device_map": str(device),
        "attn_implementation": "sdpa",
    }
    if device.type == "cuda":
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    base_model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    adapted_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapted_model = adapted_model.merge_and_unload()
    adapted_model.eval()
    for parameter in adapted_model.parameters():
        parameter.requires_grad = False

    vision_module = find_vision_module(adapted_model)
    vision_module.eval()
    embed_dim = int(adapted_model.config.vision_config.hidden_size)
    spatial_merge_size = int(adapted_model.config.vision_config.spatial_merge_size)

    pooler = AttentionalPooler(embed_dim=embed_dim)
    extras_path = adapter_path / "contrastive_extras.pt"
    if not extras_path.exists():
        raise FileNotFoundError(extras_path)
    extras = torch.load(extras_path, map_location="cpu", weights_only=True)
    if "pooler" not in extras:
        raise KeyError(f"Missing 'pooler' state in {extras_path}")
    pooler.load_state_dict(extras["pooler"])
    pooler.to(device=device, dtype=dtype)
    pooler.eval()
    for parameter in pooler.parameters():
        parameter.requires_grad = False

    return processor, vision_module, pooler, spatial_merge_size


def preprocess_images(
    processor: Any,
    images: Sequence[Image.Image],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Qwen's image preprocessing to an image batch.
    """

    processed = processor.image_processor(images=list(images), return_tensors="pt")
    return processed["pixel_values"], processed["image_grid_thw"]


class QwenRewardHeadPredictor:
    """
    Batched reward predictor used by online RL and hard-negative mining.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        for key in (
            "task",
            "model_id",
            "adapter_dir",
            "input_dim",
            "hidden_dim",
            "dropout",
            "head_state_dict",
        ):
            if key not in checkpoint:
                raise KeyError(f"Reward-head checkpoint is missing required key: {key}")

        self.task: str = checkpoint["task"]
        self.device = device
        self.dtype = dtype
        adapter_dir = resolve_checkpoint_artifact(
            checkpoint_file,
            checkpoint["adapter_dir"],
        )
        self.processor, self.vision_module, self.pooler, self.spatial_merge_size = (
            load_frozen_visual_components(
                model_id=checkpoint["model_id"],
                adapter_dir=str(adapter_dir),
                device=device,
                dtype=dtype,
            )
        )
        self.head = RewardHead(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
        ).to(device)
        self.head.load_state_dict(checkpoint["head_state_dict"])
        self.head.eval()

    def predict(self, images: Sequence[np.ndarray], batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        predictions: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            pil_images = [
                Image.fromarray(image)
                for image in images[start : start + batch_size]
            ]
            pixel_values, image_grid_thw = preprocess_images(self.processor, pil_images)
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            image_grid_thw = image_grid_thw.to(self.device)

            with torch.inference_mode():
                visual_output = self.vision_module(pixel_values, grid_thw=image_grid_thw)
                visual_embeddings = unwrap_vision_output(visual_output)
                pooled = pool_visual_embeddings(
                    visual_embeddings=visual_embeddings,
                    image_grid_thw=image_grid_thw,
                    pooler=self.pooler,
                    spatial_merge_size=self.spatial_merge_size,
                )
                rewards = self.head(pooled.float()).squeeze(1)
            predictions.append(rewards.cpu().numpy())

        if not predictions:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(predictions).astype(np.float32, copy=False)
