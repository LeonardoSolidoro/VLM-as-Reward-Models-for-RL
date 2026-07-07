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


def validate_sample(sample):
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


def build_model_inputs(processor, user_prompt, assistant_answer, images):
    full_messages = build_messages(user_prompt, assistant_answer, images)
    user_messages = [{"role": "user", "content": build_user_content(user_prompt, images)}]

    full_inputs = processor_apply_chat_template(processor, full_messages)
    prompt_inputs = processor.apply_chat_template(
        user_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[-1]
    if prompt_len >= labels.shape[-1]:
        raise ValueError("Prompt token length is not shorter than full training example")

    labels[:, :prompt_len] = IGNORE_INDEX

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[full_inputs["input_ids"] == pad_token_id] = IGNORE_INDEX

    full_inputs["labels"] = labels
    return full_inputs


def squeeze_text_batch_dim(model_inputs):
    for key in TEXT_KEYS:
        value = model_inputs.get(key)
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1:
            model_inputs[key] = value.squeeze(0)
    return model_inputs


def decode_supervised_labels(labels, tokenizer):
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
    def __init__(self, jsonl_path, config_path, processor=None, model_id=None):
        self.jsonl_path = Path(jsonl_path)
        self.prompt_template = load_finetuning_prompt_template(config_path)
        self.samples = load_jsonl(self.jsonl_path)
        self.processor = processor

        self._prepared_samples = []
        for sample in self.samples:
            validate_sample(sample)
            self._prepared_samples.append(
                {
                    "image_paths": [sample["initial_image"]] + sample["images"],
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

        return sample, images, prepared_sample["user_prompt"], prepared_sample["assistant_answer"]

    def __getitem__(self, idx):
        if self.processor is None:
            raise ValueError("A Qwen AutoProcessor is required to build model inputs")

        _, images, user_prompt, assistant_answer = self.format_sample(idx)
        model_inputs = build_model_inputs(self.processor, user_prompt, assistant_answer, images)
        return squeeze_text_batch_dim(model_inputs)


def describe_value(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def main():
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
