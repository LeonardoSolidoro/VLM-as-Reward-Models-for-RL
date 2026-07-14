"""
Run reward inference with a contrastively adapted Qwen model.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from vlm_reward.training.reward_dataset import (
    IGNORE_INDEX,
    QwenRewardContrastiveDataset,
    parse_reward_percentages,
)
from vlm_reward.models.qwen import load_merged_lora_for_inference
from vlm_reward.training.collation import (
    QwenGenerationCollator,
)
from vlm_reward.runtime import move_tensors_to_device


def decode_target(labels: torch.Tensor, tokenizer: Any) -> str:
    label_ids = labels[labels != IGNORE_INDEX]
    return tokenizer.decode(label_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


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

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = load_merged_lora_for_inference(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        bf16=args.bf16,
    )

    dataset = QwenRewardContrastiveDataset(args.jsonl, processor=processor)
    device = next(model.parameters()).device

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=QwenGenerationCollator(pad_token_id),
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    results = []
    sample_idx = 0

    for batch in tqdm(dataloader, desc="Generating"):
        labels_list = batch.pop("labels")
        prompt_inputs = move_tensors_to_device(batch, device)

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
