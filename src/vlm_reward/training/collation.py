"""
Padding and visual batching shared by all Qwen fine-tuning entry points.
"""
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class QwenMultimodalCollator:
    """
    Collate supervised or cross-view Qwen examples without schema guessing.
    """
    pad_token_id: int
    ignore_index: int = -100

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty feature batch")

        for key in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
            self._require_key_in_every_feature(features, key)

        batch: Dict[str, Any] = {
            "input_ids": self._pad([feature["input_ids"] for feature in features], self.pad_token_id),
            "attention_mask": self._pad([feature["attention_mask"] for feature in features], 0),
            "labels": self._pad([feature["labels"] for feature in features], self.ignore_index),
            "pixel_values": torch.cat([feature["pixel_values"] for feature in features], dim=0),
            "image_grid_thw": torch.cat([feature["image_grid_thw"] for feature in features], dim=0),
        }

        self._copy_optional_padded(features, batch, "mm_token_type_ids", padding_value=0)

        has_positive = self._optional_key_presence(features, "pixel_values_positive")
        has_positive_grid = self._optional_key_presence(features, "image_grid_thw_positive")
        if has_positive != has_positive_grid:
            raise KeyError(
                "pixel_values_positive and image_grid_thw_positive must be present together"
            )
        if has_positive:
            batch["pixel_values_positive"] = torch.cat(
                [feature["pixel_values_positive"] for feature in features],
                dim=0,
            )
            batch["image_grid_thw_positive"] = torch.cat(
                [feature["image_grid_thw_positive"] for feature in features],
                dim=0,
            )

        has_frame_indices = self._optional_key_presence(features, "frame_indices")
        if has_frame_indices:
            batch["frame_indices"] = torch.cat(
                [feature["frame_indices"] for feature in features],
                dim=0,
            )
            trajectory_indices = []
            for sample_index, feature in enumerate(features):
                frame_count = int(feature["frame_indices"].shape[0])
                trajectory_indices.append(
                    torch.full((frame_count,), sample_index, dtype=torch.long)
                )
            batch["trajectory_indices"] = torch.cat(trajectory_indices, dim=0)

        return batch

    @staticmethod
    def _require_key_in_every_feature(
        features: Sequence[Mapping[str, Any]],
        key: str,
    ) -> None:
        missing = [index for index, feature in enumerate(features) if key not in feature]
        if missing:
            raise KeyError(f"Features at indices {missing} are missing required key {key!r}")

    @classmethod
    def _optional_key_presence(
        cls,
        features: Sequence[Mapping[str, Any]],
        key: str,
    ) -> bool:
        present = [key in feature for feature in features]
        if any(present) and not all(present):
            raise KeyError(f"Optional key {key!r} is present in only part of the batch")
        return all(present)

    def _copy_optional_padded(
        self,
        features: Sequence[Mapping[str, Any]],
        batch: Dict[str, Any],
        key: str,
        padding_value: int,
    ) -> None:
        if self._optional_key_presence(features, key):
            batch[key] = self._pad(
                [feature[key] for feature in features],
                padding_value,
            )

    @staticmethod
    def _pad(tensors: Sequence[torch.Tensor], value: int) -> torch.Tensor:
        return pad_sequence(tensors, batch_first=True, padding_value=value)


def sample_to_generation_inputs(
    sample: Mapping[str, Any],
    ignore_index: int,
) -> Dict[str, Any]:
    """
    Trim a supervised sample immediately before its assistant response.
    """
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"):
        if key not in sample:
            raise KeyError(f"Generation sample is missing required key {key!r}")
    label_positions = torch.where(sample["labels"] != ignore_index)[0]
    if label_positions.numel() == 0:
        raise ValueError("Sample has no supervised label tokens")
    answer_start = int(label_positions[0].item())
    inputs: Dict[str, Any] = {
        "input_ids": sample["input_ids"][:answer_start],
        "attention_mask": sample["attention_mask"][:answer_start],
        "pixel_values": sample["pixel_values"],
        "image_grid_thw": sample["image_grid_thw"],
        "labels": sample["labels"],
    }
    if "mm_token_type_ids" in sample:
        inputs["mm_token_type_ids"] = sample["mm_token_type_ids"][:answer_start]
    return inputs


@dataclass
class QwenGenerationCollator:
    """
    Left-pad Qwen prompts for deterministic batched generation.
    """
    pad_token_id: int
    ignore_index: int = -100

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty generation batch")
        processed = [
            sample_to_generation_inputs(feature, self.ignore_index)
            for feature in features
        ]
        batch: Dict[str, Any] = {
            "input_ids": self._left_pad(
                [feature["input_ids"] for feature in processed],
                self.pad_token_id,
            ),
            "attention_mask": self._left_pad(
                [feature["attention_mask"] for feature in processed],
                0,
            ),
            "pixel_values": torch.cat(
                [feature["pixel_values"] for feature in processed],
                dim=0,
            ),
            "image_grid_thw": torch.cat(
                [feature["image_grid_thw"] for feature in processed],
                dim=0,
            ),
            "labels": [feature["labels"] for feature in processed],
        }
        has_token_types = QwenMultimodalCollator._optional_key_presence(
            processed,
            "mm_token_type_ids",
        )
        if has_token_types:
            batch["mm_token_type_ids"] = self._left_pad(
                [feature["mm_token_type_ids"] for feature in processed],
                0,
            )
        return batch

    @staticmethod
    def _left_pad(tensors: Sequence[torch.Tensor], value: int) -> torch.Tensor:
        max_length = max(int(tensor.shape[0]) for tensor in tensors)
        padded = []
        for tensor in tensors:
            padding_length = max_length - int(tensor.shape[0])
            if padding_length > 0:
                padding = torch.full(
                    (padding_length,),
                    value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                tensor = torch.cat([padding, tensor])
            padded.append(tensor)
        return torch.stack(padded)
