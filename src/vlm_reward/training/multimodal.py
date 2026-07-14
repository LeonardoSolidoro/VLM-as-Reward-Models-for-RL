"""
Shared multimodal prompt assembly and supervised label masking.
"""
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Sequence

import torch
from PIL import Image


IMAGE_PLACEHOLDER = "[IMG]"
TEXT_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")


def load_rgb_image(path: str) -> Image.Image:
    """
    Load an image eagerly so its file handle is closed immediately.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        return image.convert("RGB")


def build_user_content(prompt: str, images: Sequence[Image.Image]) -> List[Dict[str, Any]]:
    """
    Replace ``[IMG]`` markers with Qwen image-content entries.
    """
    parts = prompt.split(IMAGE_PLACEHOLDER)
    placeholder_count = len(parts) - 1
    if placeholder_count != len(images):
        raise ValueError(
            f"Prompt has {placeholder_count} image placeholders but got {len(images)} images"
        )

    content: List[Dict[str, Any]] = []
    for text, image in zip(parts[:-1], images):
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "image", "image": image})
    if parts[-1]:
        content.append({"type": "text", "text": parts[-1]})
    return content


def build_messages(
    user_prompt: str,
    assistant_answer: str,
    images: Sequence[Image.Image],
) -> List[Dict[str, Any]]:
    """
    Build one supervised Qwen user/assistant conversation.
    """
    return [
        {"role": "user", "content": build_user_content(user_prompt, images)},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_answer}],
        },
    ]


def build_supervised_model_inputs(
    processor: Any,
    user_prompt: str,
    assistant_answer: str,
    images: Sequence[Image.Image],
    ignore_index: int,
) -> Dict[str, Any]:
    """
    Tokenize an example and mask every non-assistant token in its labels.
    """
    full_messages = build_messages(user_prompt, assistant_answer, images)
    user_messages = [
        {"role": "user", "content": build_user_content(user_prompt, images)}
    ]
    full_inputs = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_inputs = processor.apply_chat_template(
        user_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    if prompt_length >= labels.shape[-1]:
        raise ValueError("Prompt token length is not shorter than full training example")
    labels[:, :prompt_length] = ignore_index

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[full_inputs["input_ids"] == pad_token_id] = ignore_index
    full_inputs["labels"] = labels
    return full_inputs


def add_positive_visual_inputs(
    processor: Any,
    model_inputs: MutableMapping[str, Any],
    user_prompt: str,
    positive_images: Sequence[Image.Image],
) -> None:
    """
    Attach visual tensors for the matching cross-camera positive sequence.
    """
    positive_messages = [
        {
            "role": "user",
            "content": build_user_content(user_prompt, positive_images),
        }
    ]
    positive_inputs = processor.apply_chat_template(
        positive_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs["pixel_values_positive"] = positive_inputs["pixel_values"]
    model_inputs["image_grid_thw_positive"] = positive_inputs["image_grid_thw"]


def squeeze_text_batch_dim(
    model_inputs: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Remove the processor's singleton batch axis from text fields only.
    """
    for key in TEXT_KEYS:
        if key not in model_inputs:
            continue
        value = model_inputs[key]
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1:
            model_inputs[key] = value.squeeze(0)
    return model_inputs
