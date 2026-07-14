"""
Run task-progress inference with a fine-tuned Qwen model.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any
from tqdm import tqdm

import torch
from transformers import AutoProcessor

from vlm_reward.training.progress_dataset import IGNORE_INDEX, QwenProgressDataset
from vlm_reward.models.qwen import load_merged_lora_for_inference
from vlm_reward.training.collation import (
    QwenGenerationCollator,
)
from vlm_reward.runtime import move_tensors_to_device


def parse_percentages(text: str) -> list[int | float]:
    values = []
    for match in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        value = float(match)
        values.append(int(value) if value.is_integer() else value)
    return values


def decode_target(labels: torch.Tensor, tokenizer: Any) -> str:
    label_ids = labels[labels != IGNORE_INDEX]
    return tokenizer.decode(
        label_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--adapter-dir",
        default="finetuning_output/Qwen3-VL-8B-Progress-Supervised/lora_weights",
    )
    parser.add_argument("--jsonl", default="finetune_data/static/test.jsonl")
    parser.add_argument("--config", default="configs/configs.yaml")
    parser.add_argument(
        "--output-json",
        default="finetuning_output/Qwen3-VL-8B-Progress-Supervised/predictions_static_test.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = load_merged_lora_for_inference(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        bf16=args.bf16,
    )

    dataset = QwenProgressDataset(args.jsonl, args.config, processor=processor)
    device = next(model.parameters()).device
    
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
        
    from torch.utils.data import DataLoader
    collator = QwenGenerationCollator(pad_token_id)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collator, 
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available()
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
        new_tokens_batch = generated_ids[:, prompt_len:]
        
        for i in range(len(labels_list)):
            new_tokens = new_tokens_batch[i]
            generated_text = processor.tokenizer.decode(
                new_tokens.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            target_text = decode_target(labels_list[i], processor.tokenizer)

            print(f"\n=== Sample {sample_idx} ===")
            print("Generated:")
            print(generated_text)
            print("\nTarget:")
            print(target_text)

            results.append(
                {
                    "index": sample_idx,
                    "generated_text": generated_text,
                    "target_text": target_text,
                    "generated_percentages": parse_percentages(generated_text),
                    "target_percentages": parse_percentages(target_text),
                }
            )
            sample_idx += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
