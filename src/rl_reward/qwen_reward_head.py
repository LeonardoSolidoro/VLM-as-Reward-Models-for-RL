import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


class AttentionalPooler(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Patches, EmbedDim)
        query = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.mha(query, x, x, need_weights=False)
        return pooled.squeeze(1)


class RewardHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings shape: (Batch, EmbedDim); logits shape: (Batch, 1)
        return self.net(embeddings)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings shape: (Batch, EmbedDim); rewards shape: (Batch, 1)
        return torch.sigmoid(self.forward_logits(embeddings))


def find_vision_module(root: nn.Module) -> nn.Module:
    queue: List[nn.Module] = [root]
    while queue:
        current = queue.pop(0)
        for name, child in current.named_children():
            if name in ["visual", "vision_model", "vision_tower", "vision_encoder"]:
                return child
        queue.extend(list(current.children()))
    raise AttributeError("No vision encoder found in the model tree")


def unwrap_vision_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported vision encoder output type: {type(output)}")


def split_visual_embeddings(
    visual_embeddings: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, ...]:
    if visual_embeddings.ndim == 3:
        if visual_embeddings.shape[0] != image_grid_thw.shape[0]:
            raise ValueError(
                f"Vision batch size {visual_embeddings.shape[0]} does not match "
                f"image count {image_grid_thw.shape[0]}"
            )
        return tuple(visual_embeddings[index] for index in range(visual_embeddings.shape[0]))

    if visual_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D or 3D visual embeddings, got shape {tuple(visual_embeddings.shape)}")

    raw_counts = torch.prod(image_grid_thw.to(dtype=torch.long), dim=1)
    merged_counts = raw_counts // (spatial_merge_size * spatial_merge_size)
    token_count = visual_embeddings.shape[0]

    if int(raw_counts.sum().item()) == token_count:
        split_counts = raw_counts
    elif int(merged_counts.sum().item()) == token_count:
        split_counts = merged_counts
    else:
        raise ValueError(
            f"Cannot split {token_count} visual tokens across grids {image_grid_thw.tolist()}; "
            f"raw total={int(raw_counts.sum().item())}, merged total={int(merged_counts.sum().item())}"
        )

    return torch.split(visual_embeddings, split_counts.tolist(), dim=0)


def pool_visual_embeddings(
    visual_embeddings: torch.Tensor,
    image_grid_thw: torch.Tensor,
    pooler: AttentionalPooler,
    spatial_merge_size: int,
) -> torch.Tensor:
    image_embeddings = split_visual_embeddings(visual_embeddings, image_grid_thw, spatial_merge_size)
    patch_counts = [embedding.shape[0] for embedding in image_embeddings]
    if len(set(patch_counts)) == 1:
        return pooler(torch.stack(image_embeddings, dim=0))
    pooled = [pooler(embedding.unsqueeze(0)) for embedding in image_embeddings]
    return torch.cat(pooled, dim=0)


def load_frozen_visual_components(
    model_id: str,
    adapter_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, nn.Module, AttentionalPooler, int]:
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
    adapted_model = PeftModel.from_pretrained(base_model, adapter_dir)
    adapted_model = adapted_model.merge_and_unload()
    adapted_model.eval()
    for parameter in adapted_model.parameters():
        parameter.requires_grad = False

    vision_module = find_vision_module(adapted_model)
    vision_module.eval()

    embed_dim = int(adapted_model.config.vision_config.hidden_size)
    spatial_merge_size = int(adapted_model.config.vision_config.spatial_merge_size)
    pooler = AttentionalPooler(embed_dim=embed_dim)
    extras_path = os.path.join(adapter_dir, "contrastive_extras.pt")
    extras = torch.load(extras_path, map_location="cpu", weights_only=True)
    pooler.load_state_dict(extras["pooler"])
    pooler.to(device=device, dtype=dtype)
    pooler.eval()
    for parameter in pooler.parameters():
        parameter.requires_grad = False

    return processor, vision_module, pooler, spatial_merge_size


def preprocess_images(processor: Any, images: Sequence[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
    processed = processor.image_processor(images=list(images), return_tensors="pt")
    return processed["pixel_values"], processed["image_grid_thw"]


class QwenRewardHeadPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.task: str = checkpoint["task"]
        self.device = device
        self.dtype = dtype
        self.processor, self.vision_module, self.pooler, self.spatial_merge_size = load_frozen_visual_components(
            model_id=checkpoint["model_id"],
            adapter_dir=checkpoint["adapter_dir"],
            device=device,
            dtype=dtype,
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
            pil_images = [Image.fromarray(image) for image in images[start:start + batch_size]]
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
                reward = self.head(pooled.float()).squeeze(1)
            predictions.append(reward.cpu().numpy())

        if not predictions:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(predictions).astype(np.float32, copy=False)
