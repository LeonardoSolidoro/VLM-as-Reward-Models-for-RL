"""
Train the frozen-encoder reward-head prototype.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from vlm_reward.models.reward import RewardHead, reward_head_inference_checkpoint
from vlm_reward.models.reward_predictor import (
    load_frozen_visual_components,
    preprocess_images,
)
from vlm_reward.models.vision import (
    pool_visual_embeddings,
    unwrap_vision_output,
)
from vlm_reward.models.checkpoints import relative_artifact_path
from vlm_reward.training.cache import (
    adapter_sha256,
    embedding_cache_fingerprint,
    referenced_files_sha256,
    validate_embedding_cache,
)
from vlm_reward.runtime import (
    default_device_name,
    set_global_seed,
    training_dtype,
)
from vlm_reward.training.output import prepare_run_directory


REWARD_BIN_BOUNDARIES = [0.1, 0.3, 0.6, 0.9]


class RewardFrameDataset(Dataset):
    def __init__(self, jsonl_path: str, task: str, use_both_views: bool) -> None:
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.task = task
        self.use_both_views = use_both_views
        self.examples: List[Tuple[Path, float, str]] = []
        path = self.jsonl_path
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
        return path if path.is_absolute() else PROJECT_ROOT / path

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
    def __init__(self, processor: Any) -> None:
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
    fingerprint: str,
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
        "fingerprint": fingerprint,
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
    expected_fingerprint: str,
) -> Dict[str, Any]:
    if cache_path.exists() and not rebuild_cache:
        print(f"Loading cached embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_embedding_cache(
            cache,
            expected_fingerprint=expected_fingerprint,
            expected_examples=len(dataset),
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
        fingerprint=expected_fingerprint,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved embeddings to {cache_path}")
    return cache


def calculate_balanced_sample_weights(rewards: torch.Tensor, exponent: float) -> torch.Tensor:
    if exponent < 0.0 or exponent > 1.0:
        raise ValueError(f"Balance exponent must be in [0, 1], got {exponent}")
    boundaries = torch.tensor(REWARD_BIN_BOUNDARIES, dtype=rewards.dtype)
    bin_indices = torch.bucketize(rewards, boundaries, right=True)
    counts = torch.bincount(bin_indices, minlength=len(REWARD_BIN_BOUNDARIES) + 1).float()
    if torch.any(counts == 0):
        raise ValueError(f"At least one reward bin is empty: {counts.tolist()}")
    bin_weights = counts.pow(-exponent)
    return bin_weights[bin_indices]


def selection_score(metrics: Dict[str, float], macro_weight: float) -> float:
    return metrics["mae"] + macro_weight * metrics["macro_bin_mae"]


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

    metrics["macro_bin_mae"] = float(
        np.mean([metrics[f"mae_{label}"] for label in labels])
    )

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
    parser.add_argument("--balance-exponents", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--macro-selection-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use-both-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default=default_device_name())
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.embedding_batch_size <= 0 or args.head_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.macro_selection_weight < 0.0:
        raise ValueError(
            f"--macro-selection-weight must be non-negative, got {args.macro_selection_weight}"
        )
    if len(set(args.balance_exponents)) != len(args.balance_exponents):
        raise ValueError(f"Duplicate balance exponents are not allowed: {args.balance_exponents}")
    for exponent in args.balance_exponents:
        if exponent < 0.0 or exponent > 1.0:
            raise ValueError(f"Balance exponent must be in [0, 1], got {exponent}")


def prepare_embedding_caches(
    args: argparse.Namespace,
    split_datasets: Dict[str, RewardFrameDataset],
    adapter_path: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Dict[str, Any]]:
    """
    Load verified caches or extract embeddings with the frozen encoder.
    """
    cache_dir = output_dir / "embedding_cache"
    caches: Dict[str, Dict[str, Any]] = {}
    cache_paths = {
        split_name: cache_dir / f"{split_name}.pth"
        for split_name in split_datasets
    }
    adapter_fingerprint = adapter_sha256(adapter_path)
    cache_fingerprints = {
        split_name: embedding_cache_fingerprint(
            dataset_jsonl=dataset.jsonl_path,
            model_id=args.model_id,
            adapter_dir=adapter_path,
            adapter_fingerprint=adapter_fingerprint,
            referenced_files_fingerprint=referenced_files_sha256(
                example[0] for example in dataset.examples
            ),
            dataset_options={
                "task": dataset.task,
                "use_both_views": dataset.use_both_views,
            },
        )
        for split_name, dataset in split_datasets.items()
    }
    all_caches_exist = all(path.exists() for path in cache_paths.values())
    if all_caches_exist and not args.rebuild_cache:
        for split_name, dataset in split_datasets.items():
            cache = torch.load(cache_paths[split_name], map_location="cpu", weights_only=False)
            validate_embedding_cache(
                cache,
                expected_fingerprint=cache_fingerprints[split_name],
                expected_examples=len(dataset),
            )
            caches[split_name] = cache
            print(f"Loaded cached {split_name} embeddings from {cache_paths[split_name]}")
    else:
        processor, vision_module, pooler, spatial_merge_size = load_frozen_visual_components(
            model_id=args.model_id,
            adapter_dir=str(adapter_path),
            device=device,
            dtype=dtype,
        )
        for split_name, dataset in split_datasets.items():
            caches[split_name] = load_or_extract_cache(
                cache_path=cache_paths[split_name],
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
                expected_fingerprint=cache_fingerprints[split_name],
            )
        del vision_module
        del pooler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return caches


def make_training_checkpoint(
    args: argparse.Namespace,
    *,
    adapter_path: Path,
    checkpoint_path: Path,
    head: RewardHead,
    optimizer: torch.optim.Optimizer,
    input_dim: int,
    exponent: float,
    epoch: int,
    train_loss: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    selection_metric: str,
) -> Dict[str, Any]:
    """
    Build a resumable reward-head training state.
    """
    return {
        "checkpoint_format_version": 2,
        "checkpoint_type": "reward_head_training_state",
        "task": args.task,
        "model_id": args.model_id,
        "adapter_dir": relative_artifact_path(adapter_path, checkpoint_path),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "selection_metric": selection_metric,
        "selection_score": selection_score(val_metrics, args.macro_selection_weight),
        "macro_selection_weight": args.macro_selection_weight,
        "balance_exponent": exponent,
        "seed": args.seed,
    }


def train_balance_exponent(
    args: argparse.Namespace,
    *,
    exponent: float,
    input_dim: int,
    caches: Dict[str, Dict[str, Any]],
    adapter_path: Path,
    output_dir: Path,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train and evaluate one balancing exponent from the sweep.
    """
    set_global_seed(args.seed, args.deterministic)
    exponent_tag = str(exponent).replace(".", "p")
    print(f"\nStarting balance exponent {exponent:.3f} with seed {args.seed}")
    head = RewardHead(input_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_embeddings = caches["train"]["embeddings"]
    train_rewards = caches["train"]["rewards"]
    cached_dataset = TensorDataset(train_embeddings, train_rewards)
    sampler = WeightedRandomSampler(
        weights=calculate_balanced_sample_weights(train_rewards, exponent),
        num_samples=len(cached_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        cached_dataset,
        batch_size=args.head_batch_size,
        sampler=sampler,
    )
    best_path = output_dir / f"best_reward_head_exp_{exponent_tag}.pth"
    final_path = output_dir / f"final_reward_head_exp_{exponent_tag}.pth"
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_loss = 0.0
        example_count = 0
        for embeddings, rewards in train_loader:
            embeddings = embeddings.to(device)
            rewards = rewards.to(device)
            loss = torch.mean((head(embeddings).squeeze(1) - rewards) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * embeddings.shape[0]
            example_count += embeddings.shape[0]

        average_loss = epoch_loss / example_count
        train_metrics = evaluate_head(head, caches["train"], device, args.head_batch_size)
        val_metrics = evaluate_head(head, caches["val"], device, args.head_batch_size)
        current_score = selection_score(val_metrics, args.macro_selection_weight)
        print(
            f"Exponent {exponent:.3f} | Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={average_loss:.6f} | "
            f"train_macro_mae={train_metrics['macro_bin_mae']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f} | "
            f"val_macro_mae={val_metrics['macro_bin_mae']:.6f} | "
            f"val_success_mae={val_metrics['mae_0.9-1.0']:.6f} | "
            f"selection_score={current_score:.6f}"
        )
        if current_score < best_score:
            best_score = current_score
            torch.save(
                make_training_checkpoint(
                    args,
                    adapter_path=adapter_path,
                    checkpoint_path=best_path,
                    head=head,
                    optimizer=optimizer,
                    input_dim=input_dim,
                    exponent=exponent,
                    epoch=epoch,
                    train_loss=average_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    selection_metric="mae_plus_weighted_macro_bin_mae",
                ),
                best_path,
            )

    final_train_metrics = evaluate_head(head, caches["train"], device, args.head_batch_size)
    final_val_metrics = evaluate_head(head, caches["val"], device, args.head_batch_size)
    torch.save(
        make_training_checkpoint(
            args,
            adapter_path=adapter_path,
            checkpoint_path=final_path,
            head=head,
            optimizer=optimizer,
            input_dim=input_dim,
            exponent=exponent,
            epoch=args.epochs,
            train_loss=average_loss,
            train_metrics=final_train_metrics,
            val_metrics=final_val_metrics,
            selection_metric="final_epoch",
        ),
        final_path,
    )
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    head.load_state_dict(best_checkpoint["head_state_dict"])
    val_metrics = evaluate_head(head, caches["val"], device, args.head_batch_size)
    test_metrics = evaluate_head(head, caches["test"], device, args.head_batch_size)
    print(f"Best exponent {exponent:.3f} checkpoint: {best_path}")
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)
    best_checkpoint["test_metrics"] = test_metrics
    result = {
        "balance_exponent": exponent,
        "checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "best_epoch": best_checkpoint["epoch"],
        "selection_score": best_checkpoint["selection_score"],
        "training": best_checkpoint["train_metrics"],
        "validation": val_metrics,
        "test": test_metrics,
    }
    return result, best_checkpoint


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_global_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    dtype = training_dtype(device, args.bf16)
    output_dir = Path(args.output_dir)
    adapter_path = Path(args.adapter_dir).expanduser().resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(adapter_path)
    split_datasets = {
        "train": RewardFrameDataset(args.train_jsonl, args.task, args.use_both_views),
        "val": RewardFrameDataset(args.val_jsonl, args.task, args.use_both_views),
        "test": RewardFrameDataset(args.test_jsonl, args.task, args.use_both_views),
    }
    prepare_run_directory(
        output_dir,
        overwrite=args.overwrite,
        allowed_existing_names=("embedding_cache",),
    )
    print(
        "Dataset images | "
        + " | ".join(
            f"{split_name}={len(dataset)}"
            for split_name, dataset in split_datasets.items()
        )
    )
    caches = prepare_embedding_caches(
        args,
        split_datasets,
        adapter_path,
        output_dir,
        device,
        dtype,
    )

    train_embeddings = caches["train"]["embeddings"]
    input_dim = int(train_embeddings.shape[1])
    sweep_results: List[Dict[str, Any]] = []
    global_best_score = float("inf")
    global_best_checkpoint: Dict[str, Any] = {}
    stable_checkpoint_path = output_dir / "best_partial_balanced_reward_head.pth"

    for exponent in args.balance_exponents:
        run_result, best_run_checkpoint = train_balance_exponent(
            args,
            exponent=exponent,
            input_dim=input_dim,
            caches=caches,
            adapter_path=adapter_path,
            output_dir=output_dir,
            device=device,
        )
        sweep_results.append(run_result)

        if best_run_checkpoint["selection_score"] < global_best_score:
            global_best_score = best_run_checkpoint["selection_score"]
            global_best_checkpoint = best_run_checkpoint

    if not global_best_checkpoint:
        raise RuntimeError("The partial-balancing sweep did not produce a checkpoint")
    global_best_checkpoint["adapter_dir"] = relative_artifact_path(
        adapter_path,
        stable_checkpoint_path,
    )
    torch.save(
        reward_head_inference_checkpoint(global_best_checkpoint),
        stable_checkpoint_path,
    )
    best_exponent = global_best_checkpoint["balance_exponent"]
    print(
        f"\nBest sweep checkpoint: {stable_checkpoint_path} | "
        f"exponent={best_exponent:.3f} | score={global_best_score:.6f}"
    )
    print_metrics("Best validation", global_best_checkpoint["val_metrics"])
    print_metrics("Best test", global_best_checkpoint["test_metrics"])

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(
            {
                "task": args.task,
                "checkpoint": str(stable_checkpoint_path),
                "best_epoch": global_best_checkpoint["epoch"],
                "balance_exponent": best_exponent,
                "selection_metric": global_best_checkpoint["selection_metric"],
                "selection_score": global_best_score,
                "macro_selection_weight": args.macro_selection_weight,
                "training": global_best_checkpoint["train_metrics"],
                "validation": global_best_checkpoint["val_metrics"],
                "test": global_best_checkpoint["test_metrics"],
                "sweep": sweep_results,
            },
            handle,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
