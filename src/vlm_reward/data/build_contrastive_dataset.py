"""
Build paired-camera task-progress datasets for contrastive fine-tuning.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Mapping, Sequence

from .build_progress_dataset import progress_label
from .common import (
    DEFAULT_NUM_FRAMES,
    DEFAULT_TASKS,
    DEFAULT_VIEW,
    JsonRecord,
    frame_path,
    list_rollouts,
    load_project_config,
    project_root,
    relative_path,
    resolve_path,
    rollout_number,
    select_tiny_per_task,
    split_fixed_counts,
    stable_seed,
    validate_disjoint_paths,
    validate_frame_sequence,
    validate_unique_ids,
    write_jsonl_files,
)


DEFAULT_TRAIN_PER_TASK = 400
DEFAULT_VAL_PER_TASK = 50
DEFAULT_TEST_PER_TASK = 50
DEFAULT_TINY_PER_TASK = 2
SECONDARY_CAMERA_MODES = ("moving", "moving_mounted")

RolloutPair = tuple[Path, Path]


def matching_rollout_pairs(
    *,
    static_root: Path,
    secondary_root: Path,
    task: str,
    level: str,
    view: str,
    num_frames: int,
) -> list[RolloutPair]:
    static_rollouts = list_rollouts(static_root, task, level)
    secondary_rollouts = list_rollouts(secondary_root, task, level)
    static_by_number = {rollout_number(path): path for path in static_rollouts}
    secondary_by_number = {rollout_number(path): path for path in secondary_rollouts}
    if static_by_number.keys() != secondary_by_number.keys():
        missing_secondary = sorted(static_by_number.keys() - secondary_by_number.keys())
        missing_static = sorted(secondary_by_number.keys() - static_by_number.keys())
        raise FileNotFoundError(
            f"Camera rollouts do not match for {task}/{level}; "
            f"missing secondary={missing_secondary[:10]}, missing static={missing_static[:10]}"
        )

    pairs: list[RolloutPair] = []
    for number in sorted(static_by_number):
        static_rollout = static_by_number[number]
        secondary_rollout = secondary_by_number[number]
        validate_frame_sequence(static_rollout, view, num_frames)
        validate_frame_sequence(secondary_rollout, view, num_frames)
        pairs.append((static_rollout, secondary_rollout))
    return pairs


def build_contrastive_record(
    *,
    task: str,
    task_description: str,
    static_rollout: Path,
    secondary_rollout: Path,
    secondary_camera_mode: str,
    primary_camera_mode: str,
    repo_root: Path,
    view: str,
    num_frames: int,
    seed: int,
) -> JsonRecord:
    if secondary_camera_mode not in SECONDARY_CAMERA_MODES:
        raise ValueError(f"Unsupported secondary camera mode: {secondary_camera_mode}")
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

    frame_order = list(range(1, num_frames))
    random.Random(seed).shuffle(frame_order)
    rollout_index = rollout_number(static_rollout)
    return {
        "id": f"{task}_traj_{rollout_index:06d}_{legacy_primary_view}_primary",
        "task": task,
        "task_description": task_description,
        "trajectory_path_static": relative_path(static_rollout, repo_root),
        # Compatibility key: historical code called both moving variants "moving".
        "trajectory_path_moving": relative_path(secondary_rollout, repo_root),
        "trajectory_path_anchor": relative_path(anchor_rollout, repo_root),
        "trajectory_path_positive": relative_path(positive_rollout, repo_root),
        "initial_image": relative_path(frame_path(anchor_rollout, 0, view), repo_root),
        "initial_image_positive": relative_path(
            frame_path(positive_rollout, 0, view),
            repo_root,
        ),
        "frame_order": frame_order,
        "images": [
            relative_path(frame_path(anchor_rollout, index, view), repo_root)
            for index in frame_order
        ],
        "images_positive": [
            relative_path(frame_path(positive_rollout, index, view), repo_root)
            for index in frame_order
        ],
        "progress": [progress_label(index, num_frames) for index in frame_order],
        "primary_view": legacy_primary_view,
        "primary_camera_mode": primary_camera_mode,
        "secondary_camera_mode": secondary_camera_mode,
    }


def validate_contrastive_records(
    rows: Sequence[JsonRecord],
    repo_root: Path,
    num_frames: int,
) -> None:
    validate_unique_ids(rows)
    expected_order = list(range(1, num_frames))
    for row in rows:
        row_id = str(row["id"])
        frame_order = row["frame_order"]
        images = row["images"]
        positive_images = row["images_positive"]
        progress = row["progress"]
        if not (
            len(frame_order)
            == len(images)
            == len(positive_images)
            == len(progress)
            == num_frames - 1
        ):
            raise ValueError(f"Mismatched record lengths for {row_id}")
        if sorted(frame_order) != expected_order:
            raise ValueError(f"Invalid frame order for {row_id}: {frame_order}")
        for key in ("initial_image", "initial_image_positive"):
            if not (repo_root / str(row[key])).is_file():
                raise FileNotFoundError(f"Missing {key} for {row_id}: {row[key]}")
        for frame_index, anchor, positive, label in zip(
            frame_order,
            images,
            positive_images,
            progress,
        ):
            if not (repo_root / str(anchor)).is_file():
                raise FileNotFoundError(f"Missing anchor image for {row_id}: {anchor}")
            if not (repo_root / str(positive)).is_file():
                raise FileNotFoundError(f"Missing positive image for {row_id}: {positive}")
            expected = progress_label(int(frame_index), num_frames)
            if int(label) != expected:
                raise ValueError(f"Invalid progress for {row_id}: got {label}, expected {expected}")


def build_contrastive_splits(
    *,
    repo_root: Path,
    data_root: Path,
    secondary_camera_mode: str,
    level: str,
    task_descriptions: Mapping[str, str],
    tasks: Sequence[str],
    view: str,
    num_frames: int,
    train_per_task: int,
    val_per_task: int,
    test_per_task: int,
    tiny_per_task: int,
    train_secondary_fraction: float,
    seed: int,
) -> dict[str, list[JsonRecord]]:
    if train_secondary_fraction < 0.0 or train_secondary_fraction > 1.0:
        raise ValueError(
            f"train_secondary_fraction must be in [0, 1], got {train_secondary_fraction}"
        )

    static_root = data_root / "static"
    secondary_root = data_root / secondary_camera_mode
    split_rows: dict[str, list[JsonRecord]] = {"train": [], "val": [], "test": []}
    for task in tasks:
        pairs = matching_rollout_pairs(
            static_root=static_root,
            secondary_root=secondary_root,
            task=task,
            level=level,
            view=view,
            num_frames=num_frames,
        )
        task_splits = split_fixed_counts(
            pairs,
            train_count=train_per_task,
            val_count=val_per_task,
            test_count=test_per_task,
            seed=stable_seed(seed, task),
        )
        for split_name, split_pairs in task_splits.items():
            secondary_count = (
                int(len(split_pairs) * train_secondary_fraction)
                if split_name == "train"
                else len(split_pairs)
            )
            for pair_index, (static_rollout, secondary_rollout) in enumerate(split_pairs):
                primary_camera_mode = (
                    secondary_camera_mode if pair_index < secondary_count else "static"
                )
                record = build_contrastive_record(
                    task=task,
                    task_description=task_descriptions[task],
                    static_rollout=static_rollout,
                    secondary_rollout=secondary_rollout,
                    secondary_camera_mode=secondary_camera_mode,
                    primary_camera_mode=primary_camera_mode,
                    repo_root=repo_root,
                    view=view,
                    num_frames=num_frames,
                    seed=stable_seed(seed + rollout_number(static_rollout), task),
                )
                split_rows[split_name].append(record)

    split_rows["tiny"] = select_tiny_per_task(
        split_rows["train"],
        tasks,
        tiny_per_task,
    )
    for rows in split_rows.values():
        validate_contrastive_records(rows, repo_root, num_frames)
    validate_disjoint_paths(split_rows, "trajectory_path_static")
    return split_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paired-camera progress JSONL datasets.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--secondary-camera-mode",
        choices=SECONDARY_CAMERA_MODES,
        default="moving",
    )
    parser.add_argument("--level", default="expert")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--train-per-task", type=int, default=DEFAULT_TRAIN_PER_TASK)
    parser.add_argument("--val-per-task", type=int, default=DEFAULT_VAL_PER_TASK)
    parser.add_argument("--test-per-task", type=int, default=DEFAULT_TEST_PER_TASK)
    parser.add_argument("--tiny-per-task", type=int, default=DEFAULT_TINY_PER_TASK)
    parser.add_argument("--train-secondary-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = project_root()
    config = load_project_config(repo_root)
    seed = int(config["seed"]) if args.seed is None else args.seed
    data_root_value = config["data_root"] if args.data_root is None else args.data_root
    data_root = resolve_path(data_root_value, repo_root)
    if args.output_root is None:
        suffix = "" if args.secondary_camera_mode == "moving" else "_moving_mounted"
        output_root = repo_root / f"finetune_data_contrastive{suffix}"
    else:
        output_root = resolve_path(args.output_root, repo_root)

    task_descriptions = {
        task: str(config["tasks"][task]["description"])
        for task in args.tasks
    }
    split_rows = build_contrastive_splits(
        repo_root=repo_root,
        data_root=data_root,
        secondary_camera_mode=args.secondary_camera_mode,
        level=args.level,
        task_descriptions=task_descriptions,
        tasks=args.tasks,
        view=args.view,
        num_frames=args.num_frames,
        train_per_task=args.train_per_task,
        val_per_task=args.val_per_task,
        test_per_task=args.test_per_task,
        tiny_per_task=args.tiny_per_task,
        train_secondary_fraction=args.train_secondary_fraction,
        seed=seed,
    )
    write_jsonl_files(output_root, split_rows, overwrite=args.overwrite)
    print(f"Created paired progress dataset ({args.secondary_camera_mode} versus static):")
    for split_name, rows in split_rows.items():
        print(f"  {split_name}: {len(rows)} -> {output_root / f'{split_name}.jsonl'}")


if __name__ == "__main__":
    main()
