import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from qwen_progress_dataset import IGNORE_INDEX, QwenProgressDataset
except ModuleNotFoundError:
    from src.qwen_progress_dataset import IGNORE_INDEX, QwenProgressDataset


def parse_percentages(text):
    values = []
    for match in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        value = float(match)
        values.append(int(value) if value.is_integer() else value)
    return values


def decode_target(labels, tokenizer):
    label_ids = labels[labels != IGNORE_INDEX]
    return tokenizer.decode(
        label_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def sample_to_prompt_inputs(sample):
    label_positions = torch.where(sample["labels"] != IGNORE_INDEX)[0]
    if len(label_positions) == 0:
        raise ValueError("Sample has no supervised label tokens")

    answer_start = label_positions[0].item()
    inputs = {
        "input_ids": sample["input_ids"][:answer_start],
        "attention_mask": sample["attention_mask"][:answer_start],
        "pixel_values": sample["pixel_values"],
        "image_grid_thw": sample["image_grid_thw"],
        "labels": sample["labels"]
    }

    if "mm_token_type_ids" in sample:
        inputs["mm_token_type_ids"] = sample["mm_token_type_ids"][:answer_start]

    return inputs


class QwenProgressCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        processed_features = [sample_to_prompt_inputs(f) for f in features]
        
        def left_pad(tensors, pad_value):
            max_len = max(t.shape[0] for t in tensors)
            padded = []
            for t in tensors:
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    padding = torch.full((pad_len,), pad_value, dtype=t.dtype, device=t.device)
                    padded.append(torch.cat([padding, t]))
                else:
                    padded.append(t)
            return torch.stack(padded)

        batch = {
            "input_ids": left_pad([f["input_ids"] for f in processed_features], self.pad_token_id),
            "attention_mask": left_pad([f["attention_mask"] for f in processed_features], 0),
            "pixel_values": torch.cat([f["pixel_values"] for f in processed_features], dim=0),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in processed_features], dim=0),
        }
        if "mm_token_type_ids" in processed_features[0]:
            batch["mm_token_type_ids"] = left_pad([f["mm_token_type_ids"] for f in processed_features], 0)
            
        batch["labels"] = [f["labels"] for f in processed_features]
        return batch


def move_to_device(inputs, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/qwen3vl-progress-lora-tiny")
    parser.add_argument("--jsonl", default="finetune_data/tiny.jsonl")
    parser.add_argument("--config", default="configs/configs.yaml")
    parser.add_argument("--output-json", default="outputs/tiny_overfit_predictions.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    from peft import PeftModel

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"Loading base model: {args.model_id}")
    model_kwargs = {
        "device_map": "auto"
    }
    if torch.cuda.is_available():
        model_kwargs["dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    dataset = QwenProgressDataset(args.jsonl, args.config, processor=processor)
    device = next(model.parameters()).device
    
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
        
    from torch.utils.data import DataLoader
    collator = QwenProgressCollator(pad_token_id)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collator, 
        num_workers=args.dataloader_num_workers,
        pin_memory=True
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
