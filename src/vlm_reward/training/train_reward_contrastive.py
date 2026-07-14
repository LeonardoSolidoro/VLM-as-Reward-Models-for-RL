"""
Contrastively adapt Qwen to normalized ManiSkill reward estimation.
"""

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from transformers import AutoProcessor, Trainer, TrainingArguments

from vlm_reward.training.reward_dataset import (
    IGNORE_INDEX,
    QwenRewardContrastiveDataset,
)
from vlm_reward.runtime import set_global_seed
from vlm_reward.models.contrastive import ContrastiveQwenModel
from vlm_reward.models.qwen import (
    add_lora_adapter,
    load_qwen_image_text_model,
    resolve_qlora,
)
from vlm_reward.models.vision import AttentionalPooler
from vlm_reward.training.collation import QwenMultimodalCollator


class RewardContrastiveTrainer(Trainer):
    """
    Log the language and cross-view losses separately.
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
                    "contrastive_loss": contrastive_loss.item(),
                }
            )
        return (loss, outputs) if return_outputs else loss


def save_emergency_adapter(
    model: ContrastiveQwenModel,
    processor: Any,
    output_dir: str,
    global_step: int,
) -> None:
    """
    Preserve trainable state and re-raise the original training error.
    """

    emergency_dir = Path(output_dir) / f"emergency_save_step_{global_step}"
    emergency_dir.mkdir(parents=True, exist_ok=True)
    print(f"Attempting emergency adapter save to {emergency_dir}")
    try:
        model.save_pretrained(str(emergency_dir))
        processor.save_pretrained(emergency_dir)
        print(f"Emergency adapter saved to {emergency_dir}")
    except Exception as save_error:
        print(f"Emergency save failed: {save_error}")
        traceback.print_exc()


def resolve_qlora_arg(args: argparse.Namespace) -> bool:
    return resolve_qlora(args.qlora)


def load_model(args: argparse.Namespace, use_qlora: bool) -> Any:
    return load_qwen_image_text_model(args.model_id, use_qlora, args.bf16)


def add_lora(model: Any, args: argparse.Namespace, use_qlora: bool) -> Any:
    return add_lora_adapter(
        model,
        use_qlora=use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--train-jsonl",
        default="finetune_data/reward_contrastive/train.jsonl",
    )
    parser.add_argument(
        "--val-jsonl",
        default="finetune_data/reward_contrastive/val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights",
    )
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["no", "epoch", "steps"],
    )
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--eval-strategy",
        default="steps",
        choices=["no", "epoch", "steps"],
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-persistent-workers", action="store_true")
    parser.add_argument("--dataloader-pin-memory", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--qlora", dest="qlora", action="store_true", default=None)
    parser.add_argument("--no-qlora", dest="qlora", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)
    use_qlora = resolve_qlora_arg(args)

    processor = AutoProcessor.from_pretrained(args.model_id)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    base_model = load_model(args, use_qlora)
    base_model.config.use_cache = False
    base_model = add_lora(base_model, args, use_qlora)
    model = ContrastiveQwenModel(
        base_model,
        contrastive_weight=args.contrastive_weight,
    )
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    model.pooler.to(device=device, dtype=dtype)
    model.logit_scale.data = model.logit_scale.data.to(device=device, dtype=dtype)
    model.print_trainable_parameters()

    train_dataset = QwenRewardContrastiveDataset(args.train_jsonl, processor=processor)
    val_dataset = (
        QwenRewardContrastiveDataset(args.val_jsonl, processor=processor)
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
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=(
            args.dataloader_persistent_workers and args.dataloader_num_workers > 0
        ),
        bf16=args.bf16,
        fp16=torch.cuda.is_available() and not args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = RewardContrastiveTrainer(
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

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except KeyboardInterrupt as error:
        print(f"Training interrupted by user: {error}")
        traceback.print_exc()
        save_emergency_adapter(model, processor, args.output_dir, trainer.state.global_step)
        raise
    except Exception as error:
        print(f"Training crashed: {error}")
        traceback.print_exc()
        save_emergency_adapter(model, processor, args.output_dir, trainer.state.global_step)
        raise

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved reward contrastive LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
