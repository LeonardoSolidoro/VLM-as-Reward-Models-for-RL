import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(CURRENT_DIR)
from qwen_reward_contrastive_dataset import (
    IGNORE_INDEX,
    QwenRewardContrastiveDataset,
    parse_reward_percentages,
)


def decode_target(labels: torch.Tensor, tokenizer: Any) -> str:
    label_ids = labels[labels != IGNORE_INDEX]
    return tokenizer.decode(label_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def sample_to_prompt_inputs(sample: Dict[str, Any]) -> Dict[str, Any]:
    label_positions = torch.where(sample["labels"] != IGNORE_INDEX)[0]
    if len(label_positions) == 0:
        raise ValueError("Sample has no supervised label tokens")

    answer_start = label_positions[0].item()
    inputs = {
        "input_ids": sample["input_ids"][:answer_start],
        "attention_mask": sample["attention_mask"][:answer_start],
        "pixel_values": sample["pixel_values"],
        "image_grid_thw": sample["image_grid_thw"],
        "labels": sample["labels"],
    }
    if "mm_token_type_ids" in sample:
        inputs["mm_token_type_ids"] = sample["mm_token_type_ids"][:answer_start]
    return inputs


class QwenRewardInferenceCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        processed = [sample_to_prompt_inputs(feature) for feature in features]

        def left_pad(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
            max_len = max(tensor.shape[0] for tensor in tensors)
            padded = []
            for tensor in tensors:
                pad_len = max_len - tensor.shape[0]
                if pad_len > 0:
                    padding = torch.full((pad_len,), pad_value, dtype=tensor.dtype, device=tensor.device)
                    padded.append(torch.cat([padding, tensor]))
                else:
                    padded.append(tensor)
            return torch.stack(padded)

        batch = {
            "input_ids": left_pad([f["input_ids"] for f in processed], self.pad_token_id),
            "attention_mask": left_pad([f["attention_mask"] for f in processed], 0),
            "pixel_values": torch.cat([f["pixel_values"] for f in processed], dim=0),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in processed], dim=0),
            "labels": [f["labels"] for f in processed],
        }
        if "mm_token_type_ids" in processed[0]:
            batch["mm_token_type_ids"] = left_pad([f["mm_token_type_ids"] for f in processed], 0)
        return batch


def move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter-dir", default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights")
    parser.add_argument("--jsonl", default="finetune_data/reward_contrastive/test.jsonl")
    parser.add_argument("--output-json", default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/results/inference_results.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from peft import PeftModel

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_id)

    model_kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    model.eval()

    dataset = QwenRewardContrastiveDataset(args.jsonl, processor=processor)
    device = next(model.parameters()).device

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=QwenRewardInferenceCollator(pad_token_id),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    results = []
    sample_idx = 0

    for batch in tqdm(dataloader, desc="Generating"):
        labels_list = batch.pop("labels")
        prompt_inputs = move_to_device(batch, device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        prompt_len = prompt_inputs["input_ids"].shape[1]
        generated_new_tokens = generated_ids[:, prompt_len:]

        for row_idx, labels in enumerate(labels_list):
            generated_text = processor.tokenizer.decode(
                generated_new_tokens[row_idx].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            target_text = decode_target(labels, processor.tokenizer)
            sample = dataset.samples[sample_idx]

            results.append(
                {
                    "index": sample_idx,
                    "id": sample["id"],
                    "task": sample["task"],
                    "level": sample["level"],
                    "primary_view": sample["primary_view"],
                    "frame_order": sample["frame_order"],
                    "generated_text": generated_text,
                    "target_text": target_text,
                    "generated_rewards": parse_reward_percentages(generated_text),
                    "target_rewards": [float(value) for value in sample["rewards"]],
                }
            )
            sample_idx += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved reward predictions to {output_path}")


if __name__ == "__main__":
    main()
