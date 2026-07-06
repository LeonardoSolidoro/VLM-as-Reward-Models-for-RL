import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = "[IMG]"
NUM_QUERY_FRAMES = 19
TEXT_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")

def load_finetuning_prompt_template(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["finetuning_prompt_template"]

def load_jsonl(jsonl_path):
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_frames_list(num_frames=NUM_QUERY_FRAMES):
    return "\n".join(f"Frame {i}: {IMAGE_PLACEHOLDER}" for i in range(1, num_frames + 1))

def build_user_prompt(sample, prompt_template):
    frames_list = build_frames_list(len(sample["images"]))
    return prompt_template.format(
        task_description=sample["task_description"],
        frames_list=frames_list,
    )

def build_assistant_answer(sample, prompt_template):
    use_score_tags = "<score>" in prompt_template and "</score>" in prompt_template
    blocks = []

    for i, progress in enumerate(sample["progress"], start=1):
        if use_score_tags:
            score = f"<score>{progress}%</score>"
        else:
            score = f"{progress}%"

        blocks.append(
            f"Frame {i}:\n"
            f"Task Completion Percentage: {score}"
        )

    return "\n".join(blocks)

def load_rgb_image(path):
    return Image.open(path).convert("RGB")

def build_user_content(prompt, images):
    parts = prompt.split(IMAGE_PLACEHOLDER)
    if len(parts) - 1 != len(images):
        raise ValueError(f"Prompt has {len(parts) - 1} image placeholders but got {len(images)} images")

    content = []
    for text, image in zip(parts[:-1], images):
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "image", "image": image})

    if parts[-1]:
        content.append({"type": "text", "text": parts[-1]})

    return content

def build_messages(user_prompt, assistant_answer, images):
    return [
        {"role": "user", "content": build_user_content(user_prompt, images)},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_answer}]},
    ]

def processor_apply_chat_template(processor, messages):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

def build_model_inputs(processor, user_prompt, assistant_answer, images, images_positive):
    # Anchor inputs (used for CE loss + contrastive anchor)
    full_messages = build_messages(user_prompt, assistant_answer, images)
    full_inputs = processor_apply_chat_template(processor, full_messages)

    labels = full_inputs["input_ids"].clone()
    input_ids_list = labels[0].tolist()

    # --- Fast Prompt Length Calculation ---
    # Find the prompt length by searching for the assistant header tokens.
    # Qwen-VL uses `<|im_start|>assistant\n` before the assistant's response.
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    header_tokens = processor.tokenizer.encode("assistant\n", add_special_tokens=False)
    
    if im_start_id is not None:
        response_template = [im_start_id] + header_tokens
    else:
        # Fallback if tokenizer handles it differently
        response_template = processor.tokenizer.encode("<|im_start|>assistant\n", allowed_special="all", add_special_tokens=False)
        
    prompt_len = None
    # Search backwards to find the last assistant header
    for i in range(len(input_ids_list) - len(response_template), -1, -1):
        if input_ids_list[i : i + len(response_template)] == response_template:
            prompt_len = i + len(response_template)
            break
            
    if prompt_len is None:
        # Extreme fallback: if sequence matching fails, fall back to the slow method
        print("Warning: Fast token matching failed. Falling back to slow double-processing.")
        user_messages = [{"role": "user", "content": build_user_content(user_prompt, images)}]
        prompt_inputs = processor.apply_chat_template(
            user_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[-1]

    if prompt_len >= labels.shape[-1]:
        raise ValueError("Prompt token length is not shorter than full training example")

    labels[:, :prompt_len] = IGNORE_INDEX

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[full_inputs["input_ids"] == pad_token_id] = IGNORE_INDEX

    full_inputs["labels"] = labels
    
    # Positive inputs (only need visual embeddings)
    pos_messages = [{"role": "user", "content": build_user_content(user_prompt, images_positive)}]
    pos_inputs = processor_apply_chat_template(processor, pos_messages)
    
    full_inputs["pixel_values_positive"] = pos_inputs["pixel_values"]
    full_inputs["image_grid_thw_positive"] = pos_inputs["image_grid_thw"]
    
    return full_inputs

def squeeze_text_batch_dim(model_inputs):
    for key in TEXT_KEYS:
        if key in model_inputs:
            value = model_inputs[key]
            if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1:
                model_inputs[key] = value.squeeze(0)
    return model_inputs

class QwenContrastiveDataset(Dataset):
    def __init__(self, jsonl_path, config_path, processor=None, model_id=None):
        self.jsonl_path = Path(jsonl_path)
        self.prompt_template = load_finetuning_prompt_template(config_path)
        self.samples = load_jsonl(self.jsonl_path)
        self.processor = processor

        self._prepared_samples = []
        for sample in self.samples:
            self._prepared_samples.append(
                {
                    "image_paths": [sample["initial_image"]] + sample["images"],
                    "image_paths_positive": [sample["initial_image"]] + sample["images_positive"], # Assuming initial image is the same or we just use anchor's initial image
                    "user_prompt": build_user_prompt(sample, self.prompt_template),
                    "assistant_answer": build_assistant_answer(sample, self.prompt_template),
                }
            )

        if self.processor is None and model_id is not None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)

    def __len__(self):
        return len(self.samples)

    def format_sample(self, idx):
        sample = self.samples[idx]
        prepared_sample = self._prepared_samples[idx]
        images = [load_rgb_image(path) for path in prepared_sample["image_paths"]]
        images_positive = [load_rgb_image(path) for path in prepared_sample["image_paths_positive"]]

        return sample, images, images_positive, prepared_sample["user_prompt"], prepared_sample["assistant_answer"]

    def __getitem__(self, idx):
        if self.processor is None:
            raise ValueError("A Qwen AutoProcessor is required to build model inputs")

        _, images, images_positive, user_prompt, assistant_answer = self.format_sample(idx)
        model_inputs = build_model_inputs(self.processor, user_prompt, assistant_answer, images, images_positive)
        
        sample = self.samples[idx]
        model_inputs["frame_indices"] = torch.tensor([0] + sample["frame_order"], dtype=torch.long)
        
        return squeeze_text_batch_dim(model_inputs)

