"""
Supervised dataset for Qwen task-progress fine-tuning.
"""

import argparse
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm_reward.training.multimodal import (
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


def validate_sample(sample: ProgressRecord) -> None:
    if len(sample["images"]) != NUM_QUERY_FRAMES:
        raise ValueError(f"{sample['id']} has {len(sample['images'])} query images")
    if len(sample["frame_order"]) != NUM_QUERY_FRAMES:
        raise ValueError(f"{sample['id']} has {len(sample['frame_order'])} frame_order entries")
    if len(sample["progress"]) != NUM_QUERY_FRAMES:
        raise ValueError(f"{sample['id']} has {len(sample['progress'])} progress labels")

    if not Path(sample["initial_image"]).exists():
        raise FileNotFoundError(sample["initial_image"])
    for frame_idx, image_path, progress in zip(sample["frame_order"], sample["images"], sample["progress"]):
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)


def build_model_inputs(
    processor: Any,
    user_prompt: str,
    assistant_answer: str,
    images: list[Image.Image],
) -> dict[str, Any]:
    return build_supervised_model_inputs(
        processor=processor,
        user_prompt=user_prompt,
        assistant_answer=assistant_answer,
        images=images,
        ignore_index=IGNORE_INDEX,
    )


def decode_supervised_labels(labels: torch.Tensor, tokenizer: Any) -> str:
    if labels.ndim == 2:
        labels = labels[0]

    labels = labels.clone()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    labels[labels == IGNORE_INDEX] = pad_token_id
    return tokenizer.decode(
        labels.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


class QwenProgressDataset(Dataset):
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
    ) -> tuple[ProgressRecord, list[Image.Image], str, str]:
        sample = self.samples[idx]
        prepared_sample = self._prepared_samples[idx]
        images = [load_rgb_image(path) for path in prepared_sample["image_paths"]]

        return sample, images, prepared_sample["user_prompt"], prepared_sample["assistant_answer"]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.processor is None:
            raise ValueError("A Qwen AutoProcessor is required to build model inputs")

        _, images, user_prompt, assistant_answer = self.format_sample(idx)
        model_inputs = build_model_inputs(self.processor, user_prompt, assistant_answer, images)
        return squeeze_text_batch_dim(model_inputs)


def describe_value(value: Any) -> tuple[int, ...] | str:
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    processor = None
    if args.model_id:
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(args.model_id)
        except Exception as exc:
            print(f"Could not load processor for {args.model_id}: {exc}")
            traceback.print_exc()
            raise

    dataset = QwenProgressDataset(args.jsonl, args.config, processor=processor)
    sample, images, user_prompt, assistant_answer = dataset.format_sample(0)

    print(f"sample id: {sample['id']}")
    print(f"loaded images: {len(images)} (expected 20)")
    print(f"task description: {sample['task_description']}")
    print("\nrendered user prompt:")
    print(user_prompt)
    print("\nassistant answer:")
    print(assistant_answer)

    if processor is not None:
        model_inputs = dataset[0]
        print("\nprocessor output keys/shapes:")
        for key, value in model_inputs.items():
            print(f"  {key}: {describe_value(value)}")

        print("\ndecoded supervised labels:")
        print(decode_supervised_labels(model_inputs["labels"], processor.tokenizer))


if __name__ == "__main__":
    main()
