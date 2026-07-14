"""
Cross-view contrastive dataset for task-progress fine-tuning.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm_reward.training.multimodal import (
    add_positive_visual_inputs,
    build_supervised_model_inputs,
    load_rgb_image,
    squeeze_text_batch_dim,
)
from vlm_reward.training.progress_format import (
    NUM_QUERY_FRAMES,
    ProgressRecord,
    build_assistant_answer,
    build_user_prompt,
    load_progress_records,
    load_prompt_template,
)

IGNORE_INDEX = -100


def build_model_inputs(
    processor: Any,
    user_prompt: str,
    assistant_answer: str,
    images: list[Image.Image],
    images_positive: list[Image.Image],
) -> dict[str, Any]:
    full_inputs = build_supervised_model_inputs(
        processor=processor,
        user_prompt=user_prompt,
        assistant_answer=assistant_answer,
        images=images,
        ignore_index=IGNORE_INDEX,
    )
    add_positive_visual_inputs(
        processor=processor,
        model_inputs=full_inputs,
        user_prompt=user_prompt,
        positive_images=images_positive,
    )
    return full_inputs


def initial_positive_image_path(sample: ProgressRecord) -> Path | str:
    """Return the matching second-camera reset frame.

    New records carry ``initial_image_positive`` explicitly. Historical records
    are supported by deterministically deriving the same frame from their paired
    trajectory paths; the old anchor-frame-as-positive behavior is not retained.
    """

    if "initial_image_positive" in sample:
        return sample["initial_image_positive"]
    for key in ("primary_view", "trajectory_path_static", "trajectory_path_moving"):
        if key not in sample:
            raise KeyError(
                f"{sample['id']} lacks {key!r}; cannot derive historical positive reset frame"
            )
    primary_view = sample["primary_view"]
    if primary_view == "moving":
        positive_trajectory = Path(sample["trajectory_path_static"])
    elif primary_view == "static":
        positive_trajectory = Path(sample["trajectory_path_moving"])
    else:
        raise ValueError(f"Unsupported primary_view in {sample['id']}: {primary_view}")
    return positive_trajectory / Path(sample["initial_image"]).name


def validate_sample(sample: ProgressRecord) -> None:
    sample_id = sample["id"]
    lengths = (
        len(sample["images"]),
        len(sample["images_positive"]),
        len(sample["frame_order"]),
        len(sample["progress"]),
    )
    if len(set(lengths)) != 1 or lengths[0] != NUM_QUERY_FRAMES:
        raise ValueError(f"Mismatched contrastive arrays in {sample_id}: {lengths}")
    paths = (
        [sample["initial_image"], str(initial_positive_image_path(sample))]
        + sample["images"]
        + sample["images_positive"]
    )
    for path_text in paths:
        if not Path(path_text).exists():
            raise FileNotFoundError(path_text)

class QwenContrastiveDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        config_path: str | Path,
        processor: Any = None,
        model_id: str | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.prompt_template = load_prompt_template(config_path)
        self.samples = load_progress_records(self.jsonl_path)
        self.processor = processor

        self._prepared_samples = []
        for sample in self.samples:
            validate_sample(sample)
            self._prepared_samples.append(
                {
                    "image_paths": [sample["initial_image"]] + sample["images"],
                    "image_paths_positive": [str(initial_positive_image_path(sample))]
                    + sample["images_positive"],
                    "user_prompt": build_user_prompt(sample, self.prompt_template),
                    "assistant_answer": build_assistant_answer(
                        sample["progress"], self.prompt_template
                    ),
                }
            )

        if self.processor is None and model_id is not None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)

    def __len__(self) -> int:
        return len(self.samples)

    def format_sample(
        self,
        idx: int,
    ) -> tuple[ProgressRecord, list[Image.Image], list[Image.Image], str, str]:
        sample = self.samples[idx]
        prepared_sample = self._prepared_samples[idx]
        images = [load_rgb_image(path) for path in prepared_sample["image_paths"]]
        images_positive = [load_rgb_image(path) for path in prepared_sample["image_paths_positive"]]

        return sample, images, images_positive, prepared_sample["user_prompt"], prepared_sample["assistant_answer"]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.processor is None:
            raise ValueError("A Qwen AutoProcessor is required to build model inputs")

        _, images, images_positive, user_prompt, assistant_answer = self.format_sample(idx)
        model_inputs = build_model_inputs(self.processor, user_prompt, assistant_answer, images, images_positive)
        
        sample = self.samples[idx]
        model_inputs["frame_indices"] = torch.tensor([0] + sample["frame_order"], dtype=torch.long)
        
        return squeeze_text_batch_dim(model_inputs)
