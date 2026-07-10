import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))
from utilities import set_all_seeds
from qwen_reward_contrastive_dataset import IGNORE_INDEX, QwenRewardContrastiveDataset


@dataclass
class QwenRewardContrastiveCollator:
    pad_token_id: int

    def __call__(self, features):
        batch = {
            "input_ids": self._pad([f["input_ids"] for f in features], self.pad_token_id),
            "attention_mask": self._pad([f["attention_mask"] for f in features], 0),
            "labels": self._pad([f["labels"] for f in features], IGNORE_INDEX),
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
            "pixel_values_positive": torch.cat([f["pixel_values_positive"] for f in features], dim=0),
            "image_grid_thw_positive": torch.cat([f["image_grid_thw_positive"] for f in features], dim=0),
            "frame_indices": torch.cat([f["frame_indices"] for f in features], dim=0),
        }

        trajectory_indices = []
        for sample_idx, feature in enumerate(features):
            num_frames = feature["frame_indices"].shape[0]
            trajectory_indices.append(torch.full((num_frames,), sample_idx, dtype=torch.long))
        batch["trajectory_indices"] = torch.cat(trajectory_indices, dim=0)

        if "mm_token_type_ids" in features[0]:
            batch["mm_token_type_ids"] = self._pad([f["mm_token_type_ids"] for f in features], 0)

        return batch

    @staticmethod
    def _pad(tensors, value):
        return pad_sequence(tensors, batch_first=True, padding_value=value)


class AttentionalPooler(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Patches, EmbedDim)
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.mha(query, x, x)
        return pooled.squeeze(1)


def find_vision_module(root: nn.Module):
    queue = [root]
    while queue:
        current = queue.pop(0)
        for name, child in current.named_children():
            if name in ["visual", "vision_model", "vision_tower", "vision_encoder"]:
                return current, name
        queue.extend(list(current.children()))
    raise AttributeError("No vision encoder found in the model tree")


class Qwen3VLRewardContrastiveWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, contrastive_weight: float):
        super().__init__()
        self.model = base_model
        self.contrastive_weight = contrastive_weight
        embed_dim = self.model.config.vision_config.hidden_size
        self.pooler = AttentionalPooler(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.vision_parent, self.vision_module_name = find_vision_module(self.model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_positive=None,
        image_grid_thw_positive=None,
        frame_indices=None,
        trajectory_indices=None,
        **kwargs,
    ):
        if pixel_values is not None and pixel_values.dtype.is_floating_point:
            pixel_values.requires_grad_(True)
        if pixel_values_positive is not None and pixel_values_positive.dtype.is_floating_point:
            pixel_values_positive.requires_grad_(True)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        ce_loss = outputs.loss

        if pixel_values is None or pixel_values_positive is None:
            dummy_loss = sum(param.sum() for param in self.pooler.parameters()) * 0.0
            dummy_loss = dummy_loss + self.logit_scale * 0.0
            return {"loss": ce_loss + dummy_loss, "ce_loss": ce_loss, "contrastive_loss": dummy_loss, "logits": outputs.logits}

        contrastive_loss = self.compute_contrastive_loss(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_positive=pixel_values_positive,
            image_grid_thw_positive=image_grid_thw_positive,
            frame_indices=frame_indices,
            trajectory_indices=trajectory_indices,
        )
        contrastive_loss = contrastive_loss.to(device=ce_loss.device)
        total_loss = ce_loss + self.contrastive_weight * contrastive_loss
        return {"loss": total_loss, "ce_loss": ce_loss, "contrastive_loss": contrastive_loss, "logits": outputs.logits}

    def compute_contrastive_loss(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        pixel_values_positive: torch.Tensor,
        image_grid_thw_positive: torch.Tensor,
        frame_indices: torch.Tensor,
        trajectory_indices: torch.Tensor,
    ) -> torch.Tensor:
        vision_module = getattr(self.vision_parent, self.vision_module_name)
        combined_pixel_values = torch.cat([pixel_values, pixel_values_positive], dim=0)
        combined_grid_thw = torch.cat([image_grid_thw, image_grid_thw_positive], dim=0)
        combined_embeds = vision_module(combined_pixel_values, grid_thw=combined_grid_thw)

        if not isinstance(combined_embeds, torch.Tensor):
            if hasattr(combined_embeds, "last_hidden_state") and combined_embeds.last_hidden_state is not None:
                combined_embeds = combined_embeds.last_hidden_state
            else:
                combined_embeds = combined_embeds[0]

        anchor_embeds, pos_embeds = torch.split(combined_embeds, [pixel_values.shape[0], pixel_values_positive.shape[0]])
        anchor_patches = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        pos_patches = (image_grid_thw_positive[:, 0] * image_grid_thw_positive[:, 1] * image_grid_thw_positive[:, 2]).tolist()

        pooler_device = self.pooler.query.device
        pooler_dtype = self.pooler.query.dtype
        anchor_embeds = anchor_embeds.to(device=pooler_device, dtype=pooler_dtype)
        pos_embeds = pos_embeds.to(device=pooler_device, dtype=pooler_dtype)

        anchor_pooled = torch.cat([self.pooler(split.unsqueeze(0)) for split in torch.split(anchor_embeds, anchor_patches)], dim=0)
        pos_pooled = torch.cat([self.pooler(split.unsqueeze(0)) for split in torch.split(pos_embeds, pos_patches)], dim=0)

        anchor_pooled = F.normalize(anchor_pooled - anchor_pooled.mean(dim=0, keepdim=True), dim=-1)
        pos_pooled = F.normalize(pos_pooled - pos_pooled.mean(dim=0, keepdim=True), dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_anchor = logit_scale * anchor_pooled @ pos_pooled.t()
        logits_per_pos = logit_scale * pos_pooled @ anchor_pooled.t()

        if frame_indices is not None and trajectory_indices is not None:
            frame_indices = frame_indices.to(pooler_device)
            trajectory_indices = trajectory_indices.to(pooler_device)
            same_traj = trajectory_indices.unsqueeze(1) == trajectory_indices.unsqueeze(0)
            close_time = torch.abs(frame_indices.unsqueeze(1) - frame_indices.unsqueeze(0)) <= 4
            false_negative_mask = same_traj & close_time
            false_negative_mask.fill_diagonal_(False)
            logits_per_anchor = logits_per_anchor.masked_fill(false_negative_mask, -1e9)
            logits_per_pos = logits_per_pos.masked_fill(false_negative_mask, -1e9)

        labels_cl = torch.arange(anchor_pooled.shape[0], device=pooler_device, dtype=torch.long)
        return (F.cross_entropy(logits_per_anchor, labels_cl) + F.cross_entropy(logits_per_pos, labels_cl)) / 2

    def generate(self, **kwargs):
        kwargs.pop("pixel_values_positive", None)
        kwargs.pop("image_grid_thw_positive", None)
        kwargs.pop("frame_indices", None)
        kwargs.pop("trajectory_indices", None)
        return self.model.generate(**kwargs)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def enable_input_require_grads(self, **kwargs):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable(**kwargs)

    def get_input_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        return None

    def state_dict(self, *args, **kwargs):
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param
        return state

    def save_pretrained(self, save_directory, **kwargs):
        kwargs.pop("state_dict", None)
        kwargs.pop("safe_serialization", None)
        self.model.save_pretrained(save_directory, **kwargs)
        torch.save(
            {"pooler": self.pooler.state_dict(), "logit_scale": self.logit_scale.data},
            os.path.join(save_directory, "contrastive_extras.pt"),
        )


class RewardContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        if model.training and self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "ce_loss": outputs["ce_loss"].item(),
                    "contrastive_loss": outputs["contrastive_loss"].item(),
                }
            )
        return (loss, outputs) if return_outputs else loss


def save_emergency_adapter(
    model: Qwen3VLRewardContrastiveWrapper,
    processor: Any,
    output_dir: str,
    global_step: int,
) -> None:
    emergency_dir = os.path.join(output_dir, f"emergency_save_step_{global_step}")
    os.makedirs(emergency_dir, exist_ok=True)
    print(f"Attempting emergency adapter save to {emergency_dir}")
    try:
        model.save_pretrained(emergency_dir)
        processor.save_pretrained(emergency_dir)
        print(f"Emergency adapter saved to {emergency_dir}")
    except Exception as save_error:
        print(f"Emergency save failed: {save_error}")
        traceback.print_exc()


def resolve_qlora_arg(args: argparse.Namespace) -> bool:
    if args.qlora is not None:
        return args.qlora
    return torch.cuda.is_available()


def load_model(args: argparse.Namespace, use_qlora: bool):
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs: Dict[str, Any] = {"attn_implementation": "sdpa"}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = dtype

    if use_qlora:
        if not torch.cuda.is_available():
            raise ValueError("QLoRA requires CUDA. Use --no-qlora for normal loading.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto"

    return AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)


def add_lora(model, args: argparse.Namespace, use_qlora: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    return get_peft_model(model, lora_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-jsonl", default="finetune_data/reward_contrastive/train.jsonl")
    parser.add_argument("--val-jsonl", default="finetune_data/reward_contrastive/val.jsonl")
    parser.add_argument("--output-dir", default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights")
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-strategy", default="steps", choices=["no", "epoch", "steps"])
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-strategy", default="steps", choices=["no", "epoch", "steps"])
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
    set_all_seeds(args.seed, deterministic=args.deterministic)
    use_qlora = resolve_qlora_arg(args)

    processor = AutoProcessor.from_pretrained(args.model_id)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    base_model = load_model(args, use_qlora)
    base_model.config.use_cache = False
    base_model = add_lora(base_model, args, use_qlora)

    model = Qwen3VLRewardContrastiveWrapper(base_model, contrastive_weight=args.contrastive_weight)
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    model.pooler.to(device=device, dtype=dtype)
    model.logit_scale.data = model.logit_scale.data.to(device=device, dtype=dtype)
    model.print_trainable_parameters()

    train_dataset = QwenRewardContrastiveDataset(args.train_jsonl, processor=processor)
    val_dataset = QwenRewardContrastiveDataset(args.val_jsonl, processor=processor) if args.val_jsonl else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        dataloader_persistent_workers=args.dataloader_persistent_workers and args.dataloader_num_workers > 0,
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
        data_collator=QwenRewardContrastiveCollator(pad_token_id=pad_token_id),
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
