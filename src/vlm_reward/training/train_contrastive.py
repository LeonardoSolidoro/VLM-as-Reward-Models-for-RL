"""
Contrastive Qwen fine-tuning for camera-robust task progress estimation.
"""

import argparse
from typing import Any

import torch
from transformers import AutoProcessor, Trainer, TrainingArguments

from vlm_reward.runtime import set_global_seed
from vlm_reward.training.contrastive_dataset import IGNORE_INDEX, QwenContrastiveDataset
from vlm_reward.models.contrastive import ContrastiveQwenModel
from vlm_reward.models.qwen import (
    add_lora_adapter,
    load_qwen_image_text_model,
    resolve_qlora,
)
from vlm_reward.models.vision import AttentionalPooler
from vlm_reward.training.collation import QwenMultimodalCollator

class ContrastiveTrainer(Trainer):
    """
    Log the language-model and InfoNCE components independently.
    """
    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        outputs = model(**inputs)
        loss = outputs["loss"]
        if model.training and self.state.global_step % self.args.logging_steps == 0:
            ce_loss = outputs["ce_loss"]
            contrastive_loss = outputs["contrastive_loss"]
            self.log(
                {
                    "ce_loss": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else 0.0,
                    "contrastive_loss": (
                        contrastive_loss.item()
                        if isinstance(contrastive_loss, torch.Tensor)
                        else 0.0
                    ),
                }
            )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-jsonl", default="finetune_data_contrastive/train.jsonl")
    parser.add_argument("--val-jsonl", default="finetune_data_contrastive/val.jsonl")
    parser.add_argument("--config", default="configs/configs.yaml")
    parser.add_argument(
        "--output-dir",
        default="finetuning_output/Qwen3-VL-8B-Progress-Contrastive/lora_weights",
    )
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--save-strategy",
        default="epoch",
        choices=["no", "epoch", "steps"],
    )
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--eval-strategy",
        default="epoch",
        choices=["no", "epoch", "steps"],
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument(
        "--report-to-wandb",
        action="store_true",
        help="Enable Weights & Biases logging (disabled by default).",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--qlora", dest="qlora", action="store_true", default=None)
    parser.add_argument("--no-qlora", dest="qlora", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed, deterministic=True)
    use_qlora = resolve_qlora(args.qlora)

    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    print(f"Loading model: {args.model_id}")
    base_model = load_qwen_image_text_model(args.model_id, use_qlora, args.bf16)
    base_model.config.use_cache = False
    base_model = add_lora_adapter(
        base_model,
        use_qlora=use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )

    model = ContrastiveQwenModel(
        base_model,
        contrastive_weight=args.contrastive_weight,
    )
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    model.pooler.to(device=device, dtype=dtype)
    model.logit_scale.data = model.logit_scale.data.to(device=device, dtype=dtype)
    model.print_trainable_parameters()

    train_dataset = QwenContrastiveDataset(
        args.train_jsonl,
        args.config,
        processor=processor,
    )
    val_dataset = (
        QwenContrastiveDataset(args.val_jsonl, args.config, processor=processor)
        if args.val_jsonl
        else None
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        eval_strategy=args.eval_strategy if val_dataset is not None else "no",
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=2,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        bf16=args.bf16,
        fp16=torch.cuda.is_available() and not args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=["wandb"] if args.report_to_wandb else [],
    )
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenMultimodalCollator(
            pad_token_id=pad_token_id,
            ignore_index=IGNORE_INDEX,
        ),
        processing_class=processor,
    )
    trainer.train()

    # Keep both the historical filename and the complete extras bundle.
    base_model.save_pretrained(args.output_dir)
    torch.save(model.pooler.state_dict(), f"{args.output_dir}/attentional_pooler.pt")
    torch.save(
        {"pooler": model.pooler.state_dict(), "logit_scale": model.logit_scale.data},
        f"{args.output_dir}/contrastive_extras.pt",
    )
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter, pooler, and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
