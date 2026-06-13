import argparse
import math
from dataclasses import dataclass

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

from qwen_contrastive_dataset import IGNORE_INDEX, QwenContrastiveDataset


@dataclass
class QwenContrastiveCollator:
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
        
        # Build trajectory indices to track which image belongs to which sequence
        traj_idx_list = []
        for i, f in enumerate(features):
            num_frames = f["frame_indices"].shape[0]
            traj_idx_list.append(torch.full((num_frames,), i, dtype=torch.long))
        batch["trajectory_indices"] = torch.cat(traj_idx_list, dim=0)

        if "mm_token_type_ids" in features[0]:
            batch["mm_token_type_ids"] = self._pad([f["mm_token_type_ids"] for f in features], 0)

        return batch

    @staticmethod
    def _pad(tensors, value):
        return pad_sequence(tensors, batch_first=True, padding_value=value)


class AttentionalPooler(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: [Batch, SeqLen, EmbedDim]
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        out, _ = self.mha(q, x, x)
        return out.squeeze(1) # [Batch, EmbedDim]


class Qwen3VLContrastiveWrapper(nn.Module):
    def __init__(self, base_model, config, contrastive_weight=1.0):
        super().__init__()
        self.model = base_model
        self.contrastive_weight = contrastive_weight
        
        # Get vision encoder embed dim. Assuming Qwen2/3-VL architecture.
        embed_dim = self.model.config.vision_config.hidden_size
            
        self.pooler = AttentionalPooler(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.vision_hook_output = None
        
        def hook(module, input, output):
            # Output might be a tuple or a tensor
            self.vision_hook_output = output[0] if isinstance(output, tuple) else output
            
        # Hook into the vision encoder (model.visual)
        self.hook_handle = self.model.visual.register_forward_hook(hook)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_values=None, image_grid_thw=None, pixel_values_positive=None, image_grid_thw_positive=None, frame_indices=None, trajectory_indices=None, **kwargs):
        # 1. Forward pass for the anchor (computes Cross Entropy loss)
        self.vision_hook_output = None
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )
        ce_loss = outputs.loss
        
        # If no images are present, return immediately with a dummy loss for DDP safety
        if pixel_values is None or pixel_values_positive is None:
            dummy_loss = sum(p.sum() for p in self.pooler.parameters()) * 0.0
            dummy_loss += self.logit_scale * 0.0
            total_loss = (ce_loss if ce_loss is not None else 0.0) + dummy_loss
            return {"loss": total_loss, "ce_loss": ce_loss, "contrastive_loss": dummy_loss, "logits": outputs.logits}
            
        anchor_embeds = self.vision_hook_output # [TotalPatches, EmbedDim]
        
        # 2. Forward pass for the positive images through the vision encoder only
        self.vision_hook_output = None
        # Provide grid_thw as Qwen visual expects it
        pos_embeds = self.model.visual(pixel_values_positive, grid_thw=image_grid_thw_positive)
        if isinstance(pos_embeds, tuple):
            pos_embeds = pos_embeds[0]
            
        # Determine patches per image to split the embeddings
        # image_grid_thw shape: [NumImages, 3] (T, H, W)
        def get_patches_per_image(grid_thw):
            return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        
        anchor_patches = get_patches_per_image(image_grid_thw)
        pos_patches = get_patches_per_image(image_grid_thw_positive)
        
        # Ensure embeddings are on the same device and dtype as the pooler before processing
        pooler_device = self.pooler.query.device
        pooler_dtype = self.pooler.query.dtype
        anchor_embeds = anchor_embeds.to(device=pooler_device, dtype=pooler_dtype)
        pos_embeds = pos_embeds.to(device=pooler_device, dtype=pooler_dtype)
        
        anchor_splits = torch.split(anchor_embeds, anchor_patches)
        pos_splits = torch.split(pos_embeds, pos_patches)
        
        num_images = len(anchor_splits)
        contrastive_loss = 0.0
        
        if num_images > 0:
            # Pool each image separately
            # Use max seq len in batch for padding or just run sequentially
            anchor_pooled = []
            for split in anchor_splits:
                # split: [Patches, EmbedDim]
                pooled = self.pooler(split.unsqueeze(0)) # [1, EmbedDim]
                anchor_pooled.append(pooled)
                
            pos_pooled = []
            for split in pos_splits:
                pooled = self.pooler(split.unsqueeze(0))
                pos_pooled.append(pooled)
                
            anchor_pooled = torch.cat(anchor_pooled, dim=0) # [NumImages, EmbedDim]
            pos_pooled = torch.cat(pos_pooled, dim=0) # [NumImages, EmbedDim]
            
            # Normalize features
            anchor_pooled = F.normalize(anchor_pooled, dim=-1)
            pos_pooled = F.normalize(pos_pooled, dim=-1)
            
            # Compute InfoNCE loss across all images in the batch
            # Clamp logit scale to prevent exploding gradients (max 100 as in CLIP)
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            logits_per_anchor = logit_scale * anchor_pooled @ pos_pooled.t()
            logits_per_pos = logit_scale * pos_pooled @ anchor_pooled.t()
            
            labels_cl = torch.arange(num_images, device=pooler_device, dtype=torch.long)
            
            # --- Temporal Margin Masking ---
            if frame_indices is not None and trajectory_indices is not None:
                frame_indices = frame_indices.to(pooler_device)
                trajectory_indices = trajectory_indices.to(pooler_device)
                
                same_traj = (trajectory_indices.unsqueeze(1) == trajectory_indices.unsqueeze(0))
                time_diff = torch.abs(frame_indices.unsqueeze(1) - frame_indices.unsqueeze(0))
                
                margin = 4
                close_time = (time_diff <= margin)
                
                mask = same_traj & close_time
                mask.fill_diagonal_(False) # DO NOT mask the positive pair!
                
                # Temporarily print the mask shape and sum for verification if it's the first step
                if not hasattr(self, "_printed_mask"):
                    print(f"\n[Masking Info] Total pairs: {mask.numel()}, Masked false negatives: {mask.sum().item()}")
                    self._printed_mask = True
                
                logits_per_anchor = logits_per_anchor.masked_fill(mask, -1e9)
                logits_per_pos = logits_per_pos.masked_fill(mask, -1e9)
            # -------------------------------
            
            loss_a = F.cross_entropy(logits_per_anchor, labels_cl)
            loss_p = F.cross_entropy(logits_per_pos, labels_cl)
            contrastive_loss = (loss_a + loss_p) / 2
        else:
            # Dummy loss for DDP safety when num_images == 0 but pixel_values was not None
            dummy_loss = sum(p.sum() for p in self.pooler.parameters()) * 0.0
            dummy_loss += self.logit_scale * 0.0
            contrastive_loss = dummy_loss
            
        total_loss = (ce_loss if ce_loss is not None else 0.0) + self.contrastive_weight * contrastive_loss
        
        # Prevent massive GPU VRAM spikes by releasing the hooked computation graph before backward()
        self.vision_hook_output = None
        
        return {"loss": total_loss, "ce_loss": ce_loss, "contrastive_loss": contrastive_loss, "logits": outputs.logits}

    def generate(self, **kwargs):
        # Just delegate to base model for inference
        kwargs.pop("pixel_values_positive", None)
        kwargs.pop("image_grid_thw_positive", None)
        kwargs.pop("frame_indices", None)
        kwargs.pop("trajectory_indices", None)
        return self.model.generate(**kwargs)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
        
    def enable_input_require_grads(self, **kwargs):
        # Expose to HF Trainer to ensure gradient checkpointing activates correctly
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads(**kwargs)
            
    def get_input_embeddings(self):
        # Expose to HF Trainer for standard LLM compatibility
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        return None

    def save_pretrained(self, save_directory, **kwargs):
        # Delegate to the PeftModel to strictly save only the lightweight LoRA adapters
        self.model.save_pretrained(save_directory, **kwargs)
        # Save our custom contrastive parameters alongside the adapter
        import os
        torch.save({
            "pooler": self.pooler.state_dict(),
            "logit_scale": self.logit_scale.data
        }, os.path.join(save_directory, "contrastive_extras.pt"))


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Override to support custom dictionary returns and logging
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        # Log individual losses only during training
        if model.training and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "ce_loss": outputs["ce_loss"].item() if isinstance(outputs, dict) else 0.0,
                "contrastive_loss": outputs["contrastive_loss"].item() if isinstance(outputs, dict) and isinstance(outputs["contrastive_loss"], torch.Tensor) else 0.0,
            })
            
        return (loss, outputs) if return_outputs else loss


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

    # We now apply LoRA to the vision encoder as well, per requirements.
    # Target modules for LLM (q_proj, k_proj, v_proj, o_proj) and Vision Encoder (usually contains qkv, proj, etc.)
    # We will broadly target typical attention projections in both.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-jsonl", default="finetune_data_contrastive/tiny.jsonl")
    parser.add_argument("--val-jsonl", default="finetune_data_contrastive/tiny.jsonl")
    parser.add_argument("--config", default="configs/configs.yaml")
    parser.add_argument("--output-dir", default="outputs/qwen3vl-contrastive-lora-tiny")
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
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
    base_model = load_model(args, use_qlora)
    base_model.config.use_cache = False
    
    # Add LoRA to base model (including vision encoder)
    base_model = add_lora(base_model, args, use_qlora)
    
    # Wrap with contrastive learning logic
    model = Qwen3VLContrastiveWrapper(base_model, base_model.config, contrastive_weight=args.contrastive_weight)
    
    # We must explicitly set pooler and logit_scale to require gradients and move to appropriate dtype/device
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    model.pooler.to(device=device, dtype=dtype)
    model.logit_scale.data = model.logit_scale.data.to(device=device, dtype=dtype)
    
    model.print_trainable_parameters()

    train_dataset = QwenContrastiveDataset(args.train_jsonl, args.config, processor=processor)
    val_dataset = QwenContrastiveDataset(args.val_jsonl, args.config, processor=processor) if args.val_jsonl else None

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
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        bf16=args.bf16,
        fp16=torch.cuda.is_available() and not args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenContrastiveCollator(pad_token_id=pad_token_id),
        processing_class=processor,
    )

    trainer.train()
    
    # Save base model (LoRA adapter)
    base_model.save_pretrained(args.output_dir)
    # Save the attentional pooler state dict
    torch.save(model.pooler.state_dict(), f"{args.output_dir}/attentional_pooler.pt")
    
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter, pooler, and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
