import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.rl_reward.qwen_reward_head import (
    RewardHead,
    load_frozen_visual_components,
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


class RewardFrameDataset(Dataset):
    def __init__(self, jsonl_path: str, task: str, use_both_views: bool):
        self.examples: List[Tuple[Path, float, str]] = []
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
                    raise ValueError(f"Mismatched frame arrays in sample {sample['id']}")

                primary_view = sample["primary_view"]
                positive_view = "static" if primary_view == "moving" else "moving"
                for image_path, positive_path, reward in zip(images, positive_images, rewards):
                    self.examples.append((self._resolve_path(image_path), float(reward), primary_view))
                    if use_both_views:
                        self.examples.append((self._resolve_path(positive_path), float(reward), positive_view))

        if not self.examples:
            raise ValueError(f"No examples for task {task} in {jsonl_path}")

    @staticmethod
    def _resolve_path(path_text: str) -> Path:
        path = Path(path_text)
        return path if path.is_absolute() else REPO_ROOT / path

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, float, str]:
        image_path, reward, view = self.examples[index]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        return rgb_image, reward, view


class RewardImageCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(
        self,
        features: Sequence[Tuple[Image.Image, float, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        images = [feature[0] for feature in features]
        rewards = torch.tensor([feature[1] for feature in features], dtype=torch.float32)
        views = [feature[2] for feature in features]
        pixel_values, image_grid_thw = preprocess_images(self.processor, images)
        return pixel_values, image_grid_thw, rewards, views


def extract_embeddings(
    dataset: RewardFrameDataset,
    processor: Any,
    vision_module: torch.nn.Module,
    pooler: torch.nn.Module,
    spatial_merge_size: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=RewardImageCollator(processor),
    )
    all_embeddings: List[torch.Tensor] = []
    all_rewards: List[torch.Tensor] = []
    all_views: List[str] = []

    for pixel_values, image_grid_thw, rewards, views in tqdm(loader, desc="Extracting visual embeddings"):
        pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
        image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)
        with torch.inference_mode():
            visual_output = vision_module(pixel_values, grid_thw=image_grid_thw)
            visual_embeddings = unwrap_vision_output(visual_output)
            pooled = pool_visual_embeddings(
                visual_embeddings=visual_embeddings,
                image_grid_thw=image_grid_thw,
                pooler=pooler,
                spatial_merge_size=spatial_merge_size,
            )
        all_embeddings.append(pooled.float().cpu())
        all_rewards.append(rewards)
        all_views.extend(views)

    return {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "rewards": torch.cat(all_rewards, dim=0),
        "views": all_views,
    }


def load_or_extract_cache(
    cache_path: Path,
    rebuild_cache: bool,
    dataset: RewardFrameDataset,
    processor: Any,
    vision_module: torch.nn.Module,
    pooler: torch.nn.Module,
    spatial_merge_size: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    if cache_path.exists() and not rebuild_cache:
        print(f"Loading cached embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["embeddings"].shape[0] != len(dataset):
            raise ValueError(
                f"Cached example count {cache['embeddings'].shape[0]} does not match "
                f"dataset length {len(dataset)}. Re-run with --rebuild-cache."
            )
        return cache

    cache = extract_embeddings(
        dataset=dataset,
        processor=processor,
        vision_module=vision_module,
        pooler=pooler,
        spatial_merge_size=spatial_merge_size,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved embeddings to {cache_path}")
    return cache


def calculate_bin_weights(rewards: torch.Tensor, max_weight: float) -> torch.Tensor:
    boundaries = torch.tensor(REWARD_BIN_BOUNDARIES, dtype=rewards.dtype)
    bin_indices = torch.bucketize(rewards, boundaries, right=True)
    counts = torch.bincount(bin_indices, minlength=len(REWARD_BIN_BOUNDARIES) + 1).float()
    if torch.any(counts == 0):
        raise ValueError(f"At least one reward bin is empty: {counts.tolist()}")
    weights = rewards.numel() / (counts.numel() * counts)
    return torch.clamp(weights, max=max_weight)


def evaluate_head(
    head: RewardHead,
    cache: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    embeddings = cache["embeddings"]
    rewards = cache["rewards"]
    loader = DataLoader(TensorDataset(embeddings, rewards), batch_size=batch_size, shuffle=False)
    predictions: List[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for batch_embeddings, _ in loader:
            predictions.append(head(batch_embeddings.to(device)).squeeze(1).cpu())
    predicted_rewards = torch.cat(predictions)
    absolute_errors = torch.abs(predicted_rewards - rewards)

    metrics: Dict[str, float] = {"mae": float(absolute_errors.mean().item())}
    boundaries = torch.tensor(REWARD_BIN_BOUNDARIES, dtype=rewards.dtype)
    bin_indices = torch.bucketize(rewards, boundaries, right=True)
    labels = ["0.0-0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", "0.9-1.0"]
    for bin_index, label in enumerate(labels):
        mask = bin_indices == bin_index
        if not torch.any(mask):
            raise ValueError(f"No examples in evaluation reward bin {label}")
        metrics[f"mae_{label}"] = float(absolute_errors[mask].mean().item())

    views = cache["views"]
    for view in ["moving", "static"]:
        mask = torch.tensor([value == view for value in views], dtype=torch.bool)
        if torch.any(mask):
            metrics[f"mae_{view}"] = float(absolute_errors[mask].mean().item())
    return metrics


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"{name} metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickCube-v1", choices=["PickCube-v1"])
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--adapter-dir",
        default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights",
    )
    parser.add_argument("--train-jsonl", default="finetune_data/reward_head_pickcube/train.jsonl")
    parser.add_argument("--val-jsonl", default="finetune_data/reward_head_pickcube/val.jsonl")
    parser.add_argument("--test-jsonl", default="finetune_data/reward_head_pickcube/test.jsonl")
    parser.add_argument("--output-dir", default="finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--max-sample-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use-both-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.embedding_batch_size <= 0 or args.head_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16 if args.bf16 else torch.float16
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "embedding_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RewardFrameDataset(args.train_jsonl, args.task, args.use_both_views)
    val_dataset = RewardFrameDataset(args.val_jsonl, args.task, args.use_both_views)
    test_dataset = RewardFrameDataset(args.test_jsonl, args.task, args.use_both_views)
    print(
        f"Dataset images | train={len(train_dataset)} | val={len(val_dataset)} | "
        f"test={len(test_dataset)}"
    )

    processor, vision_module, pooler, spatial_merge_size = load_frozen_visual_components(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        device=device,
        dtype=dtype,
    )

    split_datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    caches: Dict[str, Dict[str, Any]] = {}
    for split_name, dataset in split_datasets.items():
        caches[split_name] = load_or_extract_cache(
            cache_path=cache_dir / f"{split_name}.pth",
            rebuild_cache=args.rebuild_cache,
            dataset=dataset,
            processor=processor,
            vision_module=vision_module,
            pooler=pooler,
            spatial_merge_size=spatial_merge_size,
            device=device,
            dtype=dtype,
            batch_size=args.embedding_batch_size,
            num_workers=args.num_workers,
        )

    del vision_module
    del pooler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_embeddings = caches["train"]["embeddings"]
    train_rewards = caches["train"]["rewards"]
    input_dim = int(train_embeddings.shape[1])
    head = RewardHead(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    bin_weights = calculate_bin_weights(train_rewards, args.max_sample_weight).to(device)
    train_dataset_cached = TensorDataset(train_embeddings, train_rewards)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset_cached,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=generator,
    )

    best_val_mae = float("inf")
    best_checkpoint_path = output_dir / "best_reward_head.pth"
    boundaries = torch.tensor(REWARD_BIN_BOUNDARIES, device=device)

    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_loss = 0.0
        example_count = 0
        for embeddings, rewards in train_loader:
            embeddings = embeddings.to(device)
            rewards = rewards.to(device)
            predictions = head(embeddings).squeeze(1)
            loss_per_example = F.smooth_l1_loss(
                predictions,
                rewards,
                reduction="none",
                beta=args.huber_beta,
            )
            sample_weights = bin_weights[torch.bucketize(rewards, boundaries, right=True)]
            loss = (loss_per_example * sample_weights).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * embeddings.shape[0]
            example_count += embeddings.shape[0]

        average_loss = epoch_loss / example_count
        val_metrics = evaluate_head(head, caches["val"], device, args.head_batch_size)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={average_loss:.6f} | "
            f"val_mae={val_metrics['mae']:.6f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            checkpoint = {
                "task": args.task,
                "model_id": args.model_id,
                "adapter_dir": os.path.abspath(args.adapter_dir),
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": average_loss,
                "val_metrics": val_metrics,
                "seed": args.seed,
            }
            torch.save(checkpoint, best_checkpoint_path)

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    head.load_state_dict(best_checkpoint["head_state_dict"])
    val_metrics = evaluate_head(head, caches["val"], device, args.head_batch_size)
    test_metrics = evaluate_head(head, caches["test"], device, args.head_batch_size)
    print(f"Best checkpoint: {best_checkpoint_path} (epoch {best_checkpoint['epoch']})")
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(
            {
                "task": args.task,
                "checkpoint": str(best_checkpoint_path),
                "best_epoch": best_checkpoint["epoch"],
                "validation": val_metrics,
                "test": test_metrics,
            },
            handle,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
