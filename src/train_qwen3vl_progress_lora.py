import argparse
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from qwen_progress_dataset import IGNORE_INDEX, QwenProgressDataset
except ModuleNotFoundError:
    from src.qwen_progress_dataset import IGNORE_INDEX, QwenProgressDataset


@dataclass
class QwenProgressCollator:
    pad_token_id: int

    def __call__(self, features):
        batch = {
            "input_ids": self._pad([f["input_ids"] for f in features], self.pad_token_id),
            "attention_mask": self._pad([f["attention_mask"] for f in features], 0),
            "labels": self._pad([f["labels"] for f in features], IGNORE_INDEX),
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
        }

        if "mm_token_type_ids" in features[0]:
            batch["mm_token_type_ids"] = self._pad([f["mm_token_type_ids"] for f in features], 0)

        return batch

    @staticmethod
    def _pad(tensors, value):
        return pad_sequence(tensors, batch_first=True, padding_value=value)


def freeze_vision_encoder(model):
    frozen = 0
    checked = []
    base_model = getattr(model, "base_model", None)
    base_inner = getattr(base_model, "model", None) if base_model is not None else None

    for module_owner in [model, base_model, base_inner]:
        if module_owner is None or id(module_owner) in checked:
            continue
        checked.append(id(module_owner))

        for attr in ["visual", "vision_tower", "vision_model"]:
            module = getattr(module_owner, attr, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
                frozen += param.numel()

    return frozen


def resolve_qlora_arg(args):
    if args.qlora is not None:
        return args.qlora
    return torch.cuda.is_available()


def load_model(args, use_qlora):
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = dtype

    if use_qlora:
        if not torch.cuda.is_available():
            raise ValueError("QLoRA requires CUDA. Use --no-qlora for normal loading.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    return AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)


def add_lora(model, args, use_qlora):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora_config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-jsonl", default="finetune_data/tiny.jsonl")
    parser.add_argument("--val-jsonl", default="finetune_data/tiny.jsonl")
    parser.add_argument("--config", default="configs/configs.yaml")
    parser.add_argument("--output-dir", default="outputs/qwen3vl-progress-lora-tiny")
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--qlora", dest="qlora", action="store_true", default=None)
    parser.add_argument("--no-qlora", dest="qlora", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    use_qlora = resolve_qlora_arg(args)

    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    print(f"Loading model: {args.model_id}")
    print(f"QLoRA enabled: {use_qlora}")
    model = load_model(args, use_qlora)
    model.config.use_cache = False
    freeze_vision_encoder(model)
    model = add_lora(model, args, use_qlora)
    freeze_vision_encoder(model)
    model.print_trainable_parameters()

    train_dataset = QwenProgressDataset(args.train_jsonl, args.config, processor=processor)
    val_dataset = QwenProgressDataset(args.val_jsonl, args.config, processor=processor) if args.val_jsonl else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16,
        fp16=torch.cuda.is_available() and not args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenProgressCollator(pad_token_id=pad_token_id),
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
