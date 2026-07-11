import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.rl_reward.qwen_reward_head import (
    AttentionalPooler,
    RewardHead,
    find_vision_module,
    pool_visual_embeddings,
    preprocess_images,
    unwrap_vision_output,
)


REWARD_BIN_BOUNDARIES = [0.1, 0.3, 0.6, 0.9]


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


class JointRewardDataset(Dataset):
    def __init__(
        self,
        offline_jsonl: str,
        online_jsonl: str,
        task: str,
        include_both_offline_views: bool,
    ):
        self.examples: List[Dict[str, Any]] = []
        self._load_offline(offline_jsonl, task, include_both_offline_views)
        self._load_online(online_jsonl, task)
        if not self.examples:
            raise ValueError("Joint reward dataset is empty")

    def _load_offline(self, jsonl_path: str, task: str, include_both_views: bool) -> None:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if sample["task"] != task:
                    continue
                images = sample["images"]
                positive_images = sample["images_positive"]
                rewards = sample["rewards"]
                if not (len(images) == len(positive_images) == len(rewards)):
                    raise ValueError(f"Mismatched offline arrays in {sample['id']}")
                for image_path, positive_path, reward in zip(images, positive_images, rewards):
                    self.examples.append(
                        {
                            "image": resolve_path(image_path),
                            "reward": float(reward),
                            "source": "offline",
                            "hard_negative": False,
                        }
                    )
                    if include_both_views:
                        self.examples.append(
                            {
                                "image": resolve_path(positive_path),
                                "reward": float(reward),
                                "source": "offline",
                                "hard_negative": False,
                            }
                        )

    def _load_online(self, jsonl_path: str, task: str) -> None:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if sample["task"] != task:
                    continue
                self.examples.append(
                    {
                        "image": resolve_path(sample["image"]),
                        "reward": float(sample["reward"]),
                        "source": "online",
                        "hard_negative": bool(sample["hard_negative"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, float, str, bool]:
        example = self.examples[index]
        image_path = example["image"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        return rgb_image, example["reward"], example["source"], example["hard_negative"]


class JointRewardCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(
        self,
        features: Sequence[Tuple[Image.Image, float, str, bool]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        images = [feature[0] for feature in features]
        rewards = torch.tensor([feature[1] for feature in features], dtype=torch.float32)
        sources = [feature[2] for feature in features]
        hard_negatives = torch.tensor([feature[3] for feature in features], dtype=torch.bool)
        pixel_values, image_grid_thw = preprocess_images(self.processor, images)
        return pixel_values, image_grid_thw, rewards, sources, hard_negatives


def calculate_sample_weights(
    dataset: JointRewardDataset,
    balance_exponent: float,
    online_weight: float,
    hard_negative_weight: float,
) -> torch.Tensor:
    rewards = torch.tensor([example["reward"] for example in dataset.examples], dtype=torch.float32)
    boundaries = torch.tensor(REWARD_BIN_BOUNDARIES, dtype=torch.float32)
    bin_indices = torch.bucketize(rewards, boundaries, right=True)
    counts = torch.bincount(bin_indices, minlength=len(REWARD_BIN_BOUNDARIES) + 1).float()
    if torch.any(counts == 0):
        raise ValueError(f"At least one training reward bin is empty: {counts.tolist()}")
    weights = counts.pow(-balance_exponent)[bin_indices]
    for index, example in enumerate(dataset.examples):
        if example["source"] == "online":
            weights[index] *= online_weight
        if example["hard_negative"]:
            weights[index] *= hard_negative_weight
    return weights


def forward_reward_logits(
    vision_module: nn.Module,
    pooler: AttentionalPooler,
    head: RewardHead,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    visual_output = vision_module(pixel_values, grid_thw=image_grid_thw)
    visual_embeddings = unwrap_vision_output(visual_output).float()
    pooled = pool_visual_embeddings(
        visual_embeddings=visual_embeddings,
        image_grid_thw=image_grid_thw,
        pooler=pooler,
        spatial_merge_size=spatial_merge_size,
    )
    return head.forward_logits(pooled).squeeze(1)


def evaluate(
    vision_module: nn.Module,
    pooler: AttentionalPooler,
    head: RewardHead,
    loader: DataLoader,
    spatial_merge_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    vision_module.eval()
    pooler.eval()
    head.eval()
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    sources: List[str] = []
    with torch.inference_mode():
        for pixel_values, image_grid_thw, rewards, batch_sources, _ in loader:
            pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
            image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)
            logits = forward_reward_logits(
                vision_module,
                pooler,
                head,
                pixel_values,
                image_grid_thw,
                spatial_merge_size,
            )
            predicted = torch.sigmoid(logits)
            predictions.append(predicted.cpu())
            targets.append(rewards)
            sources.extend(batch_sources)

    predicted_rewards = torch.cat(predictions)
    target_rewards = torch.cat(targets)
    absolute_errors = torch.abs(predicted_rewards - target_rewards)
    metrics: Dict[str, float] = {
        "mae": float(absolute_errors.mean().item()),
        "prediction_mean": float(predicted_rewards.mean().item()),
        "prediction_std": float(predicted_rewards.std(unbiased=False).item()),
        "prediction_min": float(predicted_rewards.min().item()),
        "prediction_max": float(predicted_rewards.max().item()),
    }

    boundaries = torch.tensor(REWARD_BIN_BOUNDARIES, dtype=target_rewards.dtype)
    bin_indices = torch.bucketize(target_rewards, boundaries, right=True)
    labels = ["0.0-0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", "0.9-1.0"]
    bin_maes = []
    for bin_index, label in enumerate(labels):
        mask = bin_indices == bin_index
        if torch.any(mask):
            value = float(absolute_errors[mask].mean().item())
            metrics[f"mae_{label}"] = value
            bin_maes.append(value)
    metrics["macro_bin_mae"] = float(np.mean(bin_maes))

    for source in ["offline", "online"]:
        mask = torch.tensor([value == source for value in sources], dtype=torch.bool)
        if torch.any(mask):
            metrics[f"mae_{source}"] = float(absolute_errors[mask].mean().item())

    low_reward_mask = target_rewards <= 0.10
    if not torch.any(low_reward_mask):
        raise ValueError("Validation data contains no low-reward examples")
    metrics["low_reward_false_positive_rate"] = float(
        (predicted_rewards[low_reward_mask] >= 0.25).float().mean().item()
    )
    return metrics


def selection_score(
    metrics: Dict[str, float],
    success_weight: float,
    false_positive_weight: float,
) -> float:
    return (
        metrics["macro_bin_mae"]
        + metrics["mae_online"]
        + success_weight * metrics["mae_0.9-1.0"]
        + false_positive_weight * metrics["low_reward_false_positive_rate"]
    )


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"{name} metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")


def load_trainable_model(
    model_id: str,
    adapter_dir: str,
    reward_head_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, PeftModel, nn.Module, AttentionalPooler, RewardHead, int]:
    processor = AutoProcessor.from_pretrained(model_id)
    model_kwargs: Dict[str, Any] = {"device_map": str(device), "attn_implementation": "sdpa"}
    if device.type == "cuda":
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype
    base_model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    adapted_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    for parameter in adapted_model.parameters():
        parameter.requires_grad = False

    vision_module = find_vision_module(adapted_model)
    vision_lora_parameters = []
    for name, parameter in vision_module.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True
            vision_lora_parameters.append(parameter)
    if not vision_lora_parameters:
        raise ValueError("No trainable vision LoRA parameters were found")

    embed_dim = int(adapted_model.config.vision_config.hidden_size)
    spatial_merge_size = int(adapted_model.config.vision_config.spatial_merge_size)
    pooler = AttentionalPooler(embed_dim=embed_dim)
    extras = torch.load(
        os.path.join(adapter_dir, "contrastive_extras.pt"),
        map_location="cpu",
        weights_only=True,
    )
    pooler.load_state_dict(extras["pooler"])
    pooler.to(device=device, dtype=torch.float32)

    old_head_checkpoint = torch.load(reward_head_checkpoint, map_location="cpu", weights_only=False)
    if old_head_checkpoint["task"] != "PickCube-v1":
        raise ValueError(f"Expected PickCube reward head, got {old_head_checkpoint['task']}")
    head = RewardHead(
        input_dim=old_head_checkpoint["input_dim"],
        hidden_dim=old_head_checkpoint["hidden_dim"],
        dropout=old_head_checkpoint["dropout"],
    ).to(device)
    head.load_state_dict(old_head_checkpoint["head_state_dict"])
    return processor, adapted_model, vision_module, pooler, head, spatial_merge_size


def trainable_adapter_state(adapted_model: PeftModel) -> Dict[str, torch.Tensor]:
    state = {}
    for name, parameter in adapted_model.named_parameters():
        if parameter.requires_grad:
            state[name] = parameter.detach().cpu()
    if not state:
        raise ValueError("Adapter has no trainable parameters")
    return state


def restore_adapter_state(adapted_model: PeftModel, state: Dict[str, torch.Tensor]) -> None:
    named_parameters = dict(adapted_model.named_parameters())
    for name, value in state.items():
        parameter = named_parameters[name]
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickCube-v1", choices=["PickCube-v1"])
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--adapter-dir",
        default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights",
    )
    parser.add_argument(
        "--reward-head-checkpoint",
        default="finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube/best_partial_balanced_reward_head.pth",
    )
    parser.add_argument("--offline-train-jsonl", default="finetune_data/reward_head_pickcube/train.jsonl")
    parser.add_argument("--offline-val-jsonl", default="finetune_data/reward_head_pickcube/val.jsonl")
    parser.add_argument("--offline-test-jsonl", default="finetune_data/reward_head_pickcube/test.jsonl")
    parser.add_argument("--online-train-jsonl", default="finetune_data/reward_head_online_pickcube/train.jsonl")
    parser.add_argument("--online-val-jsonl", default="finetune_data/reward_head_online_pickcube/val.jsonl")
    parser.add_argument("--online-test-jsonl", default="finetune_data/reward_head_online_pickcube/test.jsonl")
    parser.add_argument("--output-dir", default="finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--vision-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balance-exponent", type=float, default=0.5)
    parser.add_argument("--online-weight", type=float, default=1.5)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--success-selection-weight", type=float, default=1.0)
    parser.add_argument("--false-positive-selection-weight", type=float, default=0.25)
    parser.add_argument("--minimum-prediction-std", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Epoch and batch settings must be positive")
    if args.balance_exponent < 0.0 or args.balance_exponent > 1.0:
        raise ValueError("--balance-exponent must be in [0, 1]")
    if args.online_weight <= 0.0 or args.hard_negative_weight <= 0.0:
        raise ValueError("Sampling weights must be positive")
    if args.success_selection_weight < 0.0 or args.false_positive_selection_weight < 0.0:
        raise ValueError("Selection weights must be non-negative")
    if args.minimum_prediction_std <= 0.0:
        raise ValueError("--minimum-prediction-std must be positive")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")

    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16 if args.bf16 else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = JointRewardDataset(
        args.offline_train_jsonl,
        args.online_train_jsonl,
        args.task,
        include_both_offline_views=True,
    )
    val_dataset = JointRewardDataset(
        args.offline_val_jsonl,
        args.online_val_jsonl,
        args.task,
        include_both_offline_views=True,
    )
    test_dataset = JointRewardDataset(
        args.offline_test_jsonl,
        args.online_test_jsonl,
        args.task,
        include_both_offline_views=True,
    )
    print(f"Dataset images | train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")

    processor, adapted_model, vision_module, pooler, head, spatial_merge_size = load_trainable_model(
        args.model_id,
        args.adapter_dir,
        args.reward_head_checkpoint,
        device,
        dtype,
    )
    collator = JointRewardCollator(processor)
    sample_weights = calculate_sample_weights(
        train_dataset,
        args.balance_exponent,
        args.online_weight,
        args.hard_negative_weight,
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )

    vision_lora_parameters = [parameter for parameter in adapted_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": vision_lora_parameters, "lr": args.vision_learning_rate},
            {"params": pooler.parameters(), "lr": args.head_learning_rate},
            {"params": head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    trainable_parameters = (
        vision_lora_parameters
        + list(pooler.parameters())
        + list(head.parameters())
    )

    best_training_state_path = output_dir / "best_joint_training_state.pth"
    initial_val_metrics = evaluate(
        vision_module,
        pooler,
        head,
        val_loader,
        spatial_merge_size,
        device,
        dtype,
    )
    if initial_val_metrics["prediction_std"] < args.minimum_prediction_std:
        raise RuntimeError(
            "The initial reward model is already collapsed: "
            f"prediction_std={initial_val_metrics['prediction_std']:.8f}"
        )
    best_score = selection_score(
        initial_val_metrics,
        args.success_selection_weight,
        args.false_positive_selection_weight,
    )
    print_metrics("Initial validation", initial_val_metrics)
    print(f"Initial selection score: {best_score:.6f}")
    torch.save(
        {
            "adapter_trainable_state_dict": trainable_adapter_state(adapted_model),
            "pooler_state_dict": pooler.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 0,
            "train_loss": None,
            "val_metrics": initial_val_metrics,
            "selection_score": best_score,
            "seed": args.seed,
        },
        best_training_state_path,
    )

    for epoch in range(1, args.epochs + 1):
        vision_module.train()
        pooler.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        example_count = 0

        for batch_index, (pixel_values, image_grid_thw, rewards, _, _) in enumerate(
            tqdm(train_loader, desc=f"Joint epoch {epoch}/{args.epochs}"),
            start=1,
        ):
            pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
            image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)
            rewards = rewards.to(device=device, non_blocking=True)
            logits = forward_reward_logits(
                vision_module,
                pooler,
                head,
                pixel_values,
                image_grid_thw,
                spatial_merge_size,
            )
            unscaled_loss = F.binary_cross_entropy_with_logits(logits, rewards)
            loss = unscaled_loss / args.gradient_accumulation_steps
            loss.backward()

            if batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(unscaled_loss.item()) * rewards.shape[0]
            example_count += rewards.shape[0]

        average_loss = running_loss / example_count
        val_metrics = evaluate(
            vision_module,
            pooler,
            head,
            val_loader,
            spatial_merge_size,
            device,
            dtype,
        )
        if val_metrics["prediction_std"] < args.minimum_prediction_std:
            raise RuntimeError(
                f"Reward predictions collapsed after epoch {epoch}: "
                f"prediction_mean={val_metrics['prediction_mean']:.8f}, "
                f"prediction_std={val_metrics['prediction_std']:.8f}"
            )
        current_score = selection_score(
            val_metrics,
            args.success_selection_weight,
            args.false_positive_selection_weight,
        )
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={average_loss:.6f} | "
            f"val_mae={val_metrics['mae']:.6f} | val_online_mae={val_metrics['mae_online']:.6f} | "
            f"prediction_mean={val_metrics['prediction_mean']:.6f} | "
            f"prediction_std={val_metrics['prediction_std']:.6f} | "
            f"false_positive_rate={val_metrics['low_reward_false_positive_rate']:.6f} | "
            f"selection_score={current_score:.6f}"
        )

        if current_score < best_score:
            best_score = current_score
            checkpoint = {
                "adapter_trainable_state_dict": trainable_adapter_state(adapted_model),
                "pooler_state_dict": pooler.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": average_loss,
                "val_metrics": val_metrics,
                "selection_score": current_score,
                "success_selection_weight": args.success_selection_weight,
                "false_positive_selection_weight": args.false_positive_selection_weight,
                "seed": args.seed,
            }
            torch.save(checkpoint, best_training_state_path)

    best_state = torch.load(best_training_state_path, map_location="cpu", weights_only=False)
    restore_adapter_state(adapted_model, best_state["adapter_trainable_state_dict"])
    pooler.load_state_dict(best_state["pooler_state_dict"])
    head.load_state_dict(best_state["head_state_dict"])
    test_metrics = evaluate(
        vision_module,
        pooler,
        head,
        test_loader,
        spatial_merge_size,
        device,
        dtype,
    )
    print_metrics("Best validation", best_state["val_metrics"])
    print_metrics("Test", test_metrics)

    adapter_output_dir = output_dir / "vision_adapter"
    adapted_model.save_pretrained(adapter_output_dir)
    processor.save_pretrained(adapter_output_dir)
    torch.save({"pooler": pooler.state_dict()}, adapter_output_dir / "contrastive_extras.pt")

    reward_head_output_path = output_dir / "best_reward_head_joint.pth"
    reward_head_checkpoint = {
        "task": args.task,
        "model_id": args.model_id,
        "adapter_dir": str(adapter_output_dir.resolve()),
        "input_dim": int(adapted_model.config.vision_config.hidden_size),
        "hidden_dim": head.net[1].out_features,
        "dropout": head.net[3].p,
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": best_state["optimizer_state_dict"],
        "epoch": best_state["epoch"],
        "train_loss": best_state["train_loss"],
        "val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
        "selection_score": best_state["selection_score"],
        "success_selection_weight": args.success_selection_weight,
        "false_positive_selection_weight": args.false_positive_selection_weight,
        "seed": args.seed,
    }
    torch.save(reward_head_checkpoint, reward_head_output_path)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(
            {
                "task": args.task,
                "checkpoint": str(reward_head_output_path),
                "adapter_dir": str(adapter_output_dir),
                "best_epoch": best_state["epoch"],
                "selection_score": best_state["selection_score"],
                "success_selection_weight": args.success_selection_weight,
                "false_positive_selection_weight": args.false_positive_selection_weight,
                "validation": best_state["val_metrics"],
                "test": test_metrics,
            },
            handle,
            indent=2,
        )
    print(f"Saved joint reward head: {reward_head_output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
