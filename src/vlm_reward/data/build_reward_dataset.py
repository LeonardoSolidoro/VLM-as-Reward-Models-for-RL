"""
Build paired-camera normalized-reward JSONL datasets.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .build_contrastive_dataset import SECONDARY_CAMERA_MODES, matching_rollout_pairs
from .common import (
    DEFAULT_NUM_FRAMES,
    DEFAULT_TASKS,
    DEFAULT_VIEW,
    REWARD_LEVELS,
    JsonRecord,
    frame_path,
    load_project_config,
    project_root,
    relative_path,
    resolve_path,
    rollout_number,
    select_tiny_per_task,
    split_fractions,
    stable_seed,
    validate_disjoint_paths,
    validate_unique_ids,
    write_jsonl_files,
)


DEFAULT_TRAIN_FRACTION = 0.80
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TINY_PER_TASK = 2


def load_rollout_rewards(rollout_path: Path, num_frames: int) -> list[float]:
    rewards_path = rollout_path / "rewards.json"
    if not rewards_path.is_file():
        raise FileNotFoundError(f"Missing reward labels: {rewards_path}")
    with rewards_path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    if len(values) != num_frames:
        raise ValueError(f"{rewards_path} has {len(values)} rewards, expected {num_frames}")

    rewards: list[float] = []
    for frame_index, value in enumerate(values):
        reward = float(value)
        if -1e-6 < reward < 0.0:
            reward = 0.0
        if 1.0 < reward < 1.0 + 1e-6:
            reward = 1.0
        if not np.isfinite(reward) or reward < 0.0 or reward > 1.0:
            raise ValueError(
                f"Reward outside [0, 1] in {rewards_path} at frame {frame_index}: {reward}"
            )
        rewards.append(reward)
    return rewards


def validate_paired_rewards(
    static_rollout: Path,
    secondary_rollout: Path,
    num_frames: int,
) -> list[float]:
    static_rewards = load_rollout_rewards(static_rollout, num_frames)
    secondary_rewards = load_rollout_rewards(secondary_rollout, num_frames)
    if not np.allclose(static_rewards, secondary_rewards, atol=1e-6, rtol=0.0):
        maximum_difference = float(
            np.max(np.abs(np.asarray(static_rewards) - np.asarray(secondary_rewards)))
        )
        raise ValueError(
            f"Paired rollouts have different reward labels: {static_rollout}, "
            f"{secondary_rollout}; max difference={maximum_difference}"
        )
    return static_rewards


def build_reward_record(
    *,
    task: str,
    task_description: str,
    level: str,
    static_rollout: Path,
    secondary_rollout: Path,
    secondary_camera_mode: str,
    primary_camera_mode: str,
    repo_root: Path,
    view: str,
    num_frames: int,
    seed: int,
) -> JsonRecord:
    rewards = validate_paired_rewards(static_rollout, secondary_rollout, num_frames)
    if primary_camera_mode == "static":
        anchor_rollout = static_rollout
        positive_rollout = secondary_rollout
        legacy_primary_view = "static"
    elif primary_camera_mode == secondary_camera_mode:
        anchor_rollout = secondary_rollout
        positive_rollout = static_rollout
        legacy_primary_view = "moving"
    else:
        raise ValueError(
            f"Primary camera must be static or {secondary_camera_mode}, got {primary_camera_mode}"
        )

    frame_order = list(range(num_frames))
    random.Random(seed).shuffle(frame_order)
    rollout_index = rollout_number(static_rollout)
    return {
        "id": f"{task}_{level}_rollout_{rollout_index:06d}_{legacy_primary_view}",
        "task": task,
        "task_description": task_description,
        "level": level,
        "primary_view": legacy_primary_view,
        "primary_camera_mode": primary_camera_mode,
        "secondary_camera_mode": secondary_camera_mode,
        "trajectory_path_static": relative_path(static_rollout, repo_root),
        # Compatibility key retained for the existing reward training loaders.
        "trajectory_path_moving": relative_path(secondary_rollout, repo_root),
        "trajectory_path_anchor": relative_path(anchor_rollout, repo_root),
        "trajectory_path_positive": relative_path(positive_rollout, repo_root),
        "frame_order": frame_order,
        "images": [
            relative_path(frame_path(anchor_rollout, index, view), repo_root)
            for index in frame_order
        ],
        "images_positive": [
            relative_path(frame_path(positive_rollout, index, view), repo_root)
            for index in frame_order
        ],
        "rewards": [round(rewards[index], 4) for index in frame_order],
    }


def validate_reward_records(
    rows: Sequence[JsonRecord],
    repo_root: Path,
    num_frames: int,
) -> None:
    validate_unique_ids(rows)
    expected_order = list(range(num_frames))
    for row in rows:
        row_id = str(row["id"])
        frame_order = row["frame_order"]
        images = row["images"]
        positives = row["images_positive"]
        rewards = row["rewards"]
        if not (
            len(frame_order) == len(images) == len(positives) == len(rewards) == num_frames
        ):
            raise ValueError(f"Mismatched record lengths for {row_id}")
        if sorted(frame_order) != expected_order:
            raise ValueError(f"Invalid frame order for {row_id}: {frame_order}")
        for anchor, positive, reward in zip(images, positives, rewards):
            if not (repo_root / str(anchor)).is_file():
                raise FileNotFoundError(f"Missing anchor image for {row_id}: {anchor}")
            if not (repo_root / str(positive)).is_file():
                raise FileNotFoundError(f"Missing positive image for {row_id}: {positive}")
            value = float(reward)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Invalid reward for {row_id}: {value}")


def build_reward_splits(
    *,
    repo_root: Path,
    data_root: Path,
    secondary_camera_mode: str,
    task_descriptions: Mapping[str, str],
    tasks: Sequence[str],
    levels: Sequence[str],
    view: str,
    num_frames: int,
    train_fraction: float,
    val_fraction: float,
    train_secondary_fraction: float,
    tiny_per_task: int,
    seed: int,
) -> dict[str, list[JsonRecord]]:
    if secondary_camera_mode not in SECONDARY_CAMERA_MODES:
        raise ValueError(f"Unsupported secondary camera mode: {secondary_camera_mode}")
    if train_secondary_fraction < 0.0 or train_secondary_fraction > 1.0:
        raise ValueError(
            f"train_secondary_fraction must be in [0, 1], got {train_secondary_fraction}"
        )

    static_root = data_root / "static"
    secondary_root = data_root / secondary_camera_mode
    split_rows: dict[str, list[JsonRecord]] = {"train": [], "val": [], "test": []}
    for task in tasks:
        for level in levels:
            pairs = matching_rollout_pairs(
                static_root=static_root,
                secondary_root=secondary_root,
                task=task,
                level=level,
                view=view,
                num_frames=num_frames,
            )
            level_seed = stable_seed(seed, task, ":", level)
            task_level_splits = split_fractions(
                pairs,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
                seed=level_seed,
            )
            for split_name, split_pairs in task_level_splits.items():
                secondary_count = (
                    int(len(split_pairs) * train_secondary_fraction)
                    if split_name == "train"
                    else len(split_pairs)
                )
                for pair_index, (static_rollout, secondary_rollout) in enumerate(split_pairs):
                    primary_camera_mode = (
                        secondary_camera_mode if pair_index < secondary_count else "static"
                    )
                    record = build_reward_record(
                        task=task,
                        task_description=task_descriptions[task],
                        level=level,
                        static_rollout=static_rollout,
                        secondary_rollout=secondary_rollout,
                        secondary_camera_mode=secondary_camera_mode,
                        primary_camera_mode=primary_camera_mode,
                        repo_root=repo_root,
                        view=view,
                        num_frames=num_frames,
                        seed=level_seed + rollout_number(static_rollout) * 17 + pair_index,
                    )
                    split_rows[split_name].append(record)

    split_rows["tiny"] = select_tiny_per_task(
        split_rows["train"],
        tasks,
        tiny_per_task,
    )
    for rows in split_rows.values():
        validate_reward_records(rows, repo_root, num_frames)
    validate_disjoint_paths(split_rows, "trajectory_path_static")
    return split_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paired normalized-reward JSONL data.")
    parser.add_argument("--data-root", default="data/reward_contrastive")
    parser.add_argument("--output-root", default="finetune_data/reward_contrastive")
    parser.add_argument(
        "--secondary-camera-mode",
        choices=SECONDARY_CAMERA_MODES,
        default="moving_mounted",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--levels", nargs="+", choices=REWARD_LEVELS, default=list(REWARD_LEVELS))
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--train-secondary-fraction", type=float, default=0.70)
    parser.add_argument("--tiny-per-task", type=int, default=DEFAULT_TINY_PER_TASK)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = project_root()
    config = load_project_config(repo_root)
    seed = int(config["seed"]) if args.seed is None else args.seed
    data_root = resolve_path(args.data_root, repo_root)
    output_root = resolve_path(args.output_root, repo_root)
    task_descriptions = {
        task: str(config["tasks"][task]["description"])
        for task in args.tasks
    }

    split_rows = build_reward_splits(
        repo_root=repo_root,
        data_root=data_root,
        secondary_camera_mode=args.secondary_camera_mode,
        task_descriptions=task_descriptions,
        tasks=args.tasks,
        levels=args.levels,
        view=args.view,
        num_frames=args.num_frames,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        train_secondary_fraction=args.train_secondary_fraction,
        tiny_per_task=args.tiny_per_task,
        seed=seed,
    )
    write_jsonl_files(output_root, split_rows, overwrite=args.overwrite)
    print("Created paired normalized-reward dataset:")
    for split_name, rows in split_rows.items():
        print(f"  {split_name}: {len(rows)} -> {output_root / f'{split_name}.jsonl'}")


if __name__ == "__main__":
    main()
