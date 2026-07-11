import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = "[IMG]"
NUM_QUERY_FRAMES = 20
TEXT_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")

TASK_REWARD_GUIDANCE = {
    "PickCube-v1": (
        "The reward combines gripper-to-cube proximity, a binary two-finger grasp-contact bonus, "
        "cube-to-goal proximity while grasped, and robot stillness when the cube is at the goal. "
        "A successful placement overrides the reward to 100%."
    ),
    "PushCube-v1": (
        "The reward combines gripper proximity to the preferred pushing pose and, after that pose is "
        "reached, cube-to-goal proximity plus keeping the cube on the table. Reaching success overrides "
        "the reward to 100%."
    ),
    "PegInsertionSide-v1": (
        "The reward combines gripper-to-peg proximity, a binary two-finger grasp-contact bonus, peg-to-hole "
        "alignment while grasped, and insertion proximity after the alignment threshold is met. Successful "
        "insertion overrides the reward to 100%."
    ),
}

REWARD_PROMPT_TEMPLATE = """You are an expert roboticist predicting ManiSkill normalized dense environment rewards from individual robot-scene frames.

Task: {task}
Task objective: {task_description}
Task-specific reward structure: {reward_guidance}

The target is ManiSkill's instantaneous shaped control reward, normalized to the range 0% to 100%. It is not a linear task-completion percentage. It can change nonlinearly or decrease, and it can jump at contact, stage-threshold, and success events. Each non-initial frame is paired with the reward returned after the action that produced that recorded state. Some reward inputs, especially contact forces and robot velocity, are hidden from a single RGB frame, so estimate them only from visible evidence rather than assuming a perfectly observable or monotonic score. For an initial reset frame before any action, use the dataset convention 0.00% because no transition reward has occurred.

The frames may be presented in arbitrary order. Judge each frame independently rather than assuming that later-listed frames have higher rewards.

For each frame, format your response exactly as follows:
<score>XX.XX%</score>

Frames:

{frames_list}
"""


def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    with jsonl_path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_frames_list(num_frames: int = NUM_QUERY_FRAMES) -> str:
    return "\n".join(f"Frame {i}: {IMAGE_PLACEHOLDER}" for i in range(1, num_frames + 1))


def format_reward_percent(reward: float) -> str:
    if reward < 0.0 or reward > 1.0:
        raise ValueError(f"Reward outside [0, 1]: {reward}")
    return f"{reward * 100:.2f}%"


def parse_reward_percentages(text: str) -> List[float]:
    score_matches = re.findall(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</score>", text, flags=re.IGNORECASE)
    if score_matches:
        return [float(value) / 100.0 for value in score_matches]

    percent_matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
    return [float(value) / 100.0 for value in percent_matches]


def build_user_prompt(sample: Dict[str, Any]) -> str:
    frames_list = build_frames_list(len(sample["images"]))
    task = sample["task"]
    if task not in TASK_REWARD_GUIDANCE:
        raise ValueError(f"Missing reward guidance for task: {task}")
    return REWARD_PROMPT_TEMPLATE.format(
        task=task,
        task_description=sample["task_description"],
        reward_guidance=TASK_REWARD_GUIDANCE[task],
        frames_list=frames_list,
    )


def build_assistant_answer(sample: Dict[str, Any]) -> str:
    blocks = []
    for frame_idx, reward in enumerate(sample["rewards"], start=1):
        blocks.append(
            f"Frame {frame_idx}:\n"
            f"Normalized Dense Reward: <score>{format_reward_percent(float(reward))}</score>"
        )
    return "\n".join(blocks)


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def validate_sample(sample: Dict[str, Any]) -> None:
    sample_id = sample["id"]
    if len(sample["images"]) != NUM_QUERY_FRAMES:
        raise ValueError(f"{sample_id} has {len(sample['images'])} images")
    if len(sample["images_positive"]) != NUM_QUERY_FRAMES:
        raise ValueError(f"{sample_id} has {len(sample['images_positive'])} positive images")
    if len(sample["frame_order"]) != NUM_QUERY_FRAMES:
        raise ValueError(f"{sample_id} has {len(sample['frame_order'])} frame indices")
    if len(sample["rewards"]) != NUM_QUERY_FRAMES:
        raise ValueError(f"{sample_id} has {len(sample['rewards'])} rewards")

    for image_path in sample["images"]:
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)
    for image_path in sample["images_positive"]:
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)
    for reward in sample["rewards"]:
        reward_value = float(reward)
        if reward_value < 0.0 or reward_value > 1.0:
            raise ValueError(f"{sample_id} has reward outside [0, 1]: {reward_value}")


def build_user_content(prompt: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
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


def build_messages(user_prompt: str, assistant_answer: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": build_user_content(user_prompt, images)},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_answer}]},
    ]


def build_model_inputs(
    processor: Any,
    user_prompt: str,
    assistant_answer: str,
    images: List[Image.Image],
    images_positive: List[Image.Image],
) -> Dict[str, Any]:
    full_messages = build_messages(user_prompt, assistant_answer, images)
    user_messages = [{"role": "user", "content": build_user_content(user_prompt, images)}]

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
    prompt_len = prompt_inputs["input_ids"].shape[-1]
    if prompt_len >= labels.shape[-1]:
        raise ValueError("Prompt token length is not shorter than full training example")

    labels[:, :prompt_len] = IGNORE_INDEX
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[full_inputs["input_ids"] == pad_token_id] = IGNORE_INDEX

    pos_messages = [{"role": "user", "content": build_user_content(user_prompt, images_positive)}]
    pos_inputs = processor.apply_chat_template(
        pos_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    full_inputs["labels"] = labels
    full_inputs["pixel_values_positive"] = pos_inputs["pixel_values"]
    full_inputs["image_grid_thw_positive"] = pos_inputs["image_grid_thw"]
    return full_inputs


def squeeze_text_batch_dim(model_inputs: Dict[str, Any]) -> Dict[str, Any]:
    for key in TEXT_KEYS:
        if key in model_inputs:
            value = model_inputs[key]
            if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1:
                model_inputs[key] = value.squeeze(0)
    return model_inputs


class QwenRewardContrastiveDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: Any = None, model_id: str = None):
        self.jsonl_path = Path(jsonl_path)
        self.samples = load_jsonl(self.jsonl_path)
        self.processor = processor

        self._prepared_samples = []
        for sample in self.samples:
            validate_sample(sample)
            self._prepared_samples.append(
                {
                    "image_paths": sample["images"],
                    "image_paths_positive": sample["images_positive"],
                    "user_prompt": build_user_prompt(sample),
                    "assistant_answer": build_assistant_answer(sample),
                }
            )

        if self.processor is None and model_id is not None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_id)

    def __len__(self) -> int:
        return len(self.samples)

    def format_sample(self, idx: int) -> Tuple[Dict[str, Any], List[Image.Image], List[Image.Image], str, str]:
        sample = self.samples[idx]
        prepared_sample = self._prepared_samples[idx]
        images = [load_rgb_image(path) for path in prepared_sample["image_paths"]]
        images_positive = [load_rgb_image(path) for path in prepared_sample["image_paths_positive"]]
        return sample, images, images_positive, prepared_sample["user_prompt"], prepared_sample["assistant_answer"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.processor is None:
            raise ValueError("A Qwen AutoProcessor is required to build model inputs")

        sample, images, images_positive, user_prompt, assistant_answer = self.format_sample(idx)
        model_inputs = build_model_inputs(self.processor, user_prompt, assistant_answer, images, images_positive)
        model_inputs["frame_indices"] = torch.tensor(sample["frame_order"], dtype=torch.long)
        return squeeze_text_batch_dim(model_inputs)


def decode_supervised_labels(labels: torch.Tensor, tokenizer: Any) -> str:
    if labels.ndim == 2:
        labels = labels[0]

    labels = labels.clone()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    labels[labels == IGNORE_INDEX] = pad_token_id
    return tokenizer.decode(labels.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    processor = None
    if args.model_id is not None:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_id)

    dataset = QwenRewardContrastiveDataset(args.jsonl, processor=processor)
    sample, images, images_positive, user_prompt, assistant_answer = dataset.format_sample(0)

    print(f"sample id: {sample['id']}")
    print(f"loaded anchor images: {len(images)}")
    print(f"loaded positive images: {len(images_positive)}")
    print(f"task: {sample['task']}")
    print(f"level: {sample['level']}")
    print("\nrendered user prompt:")
    print(user_prompt)
    print("\nassistant answer:")
    print(assistant_answer)

    if processor is not None:
        model_inputs = dataset[0]
        print("\nprocessor output keys/shapes:")
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                print(f"  {key}: {tuple(value.shape)}")
            else:
                print(f"  {key}: {type(value).__name__}")
        print("\ndecoded supervised labels:")
        print(decode_supervised_labels(model_inputs["labels"], processor.tokenizer))


if __name__ == "__main__":
    main()
