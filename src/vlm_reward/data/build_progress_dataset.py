"""
Build supervised task-progress JSONL datasets from expert images.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Mapping, Sequence

from .common import (
    CAMERA_MODES,
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


def progress_label(frame_index: int, num_frames: int) -> int:
    if num_frames < 2:
        raise ValueError(f"num_frames must be at least two, got {num_frames}")
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(f"Frame index {frame_index} is outside [0, {num_frames})")
    return round(frame_index / (num_frames - 1) * 100)


def build_progress_record(
    *,
    task: str,
    task_description: str,
    rollout_path: Path,
    repo_root: Path,
    view: str,
    num_frames: int,
    seed: int,
) -> JsonRecord:
    validate_frame_sequence(rollout_path, view, num_frames)
    frame_order = list(range(1, num_frames))
    random.Random(seed).shuffle(frame_order)
    rollout_index = rollout_number(rollout_path)
    return {
        "id": f"{task}_traj_{rollout_index:06d}",
        "task": task,
        "task_description": task_description,
        "trajectory_path": relative_path(rollout_path, repo_root),
        "initial_image": relative_path(frame_path(rollout_path, 0, view), repo_root),
        "frame_order": frame_order,
        "images": [
            relative_path(frame_path(rollout_path, index, view), repo_root)
            for index in frame_order
        ],
        "progress": [progress_label(index, num_frames) for index in frame_order],
    }


def validate_progress_records(
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
        labels = row["progress"]
        if not (
            len(frame_order) == len(images) == len(labels) == num_frames - 1
        ):
            raise ValueError(f"Mismatched record lengths for {row_id}")
        if sorted(frame_order) != expected_order:
            raise ValueError(f"Invalid frame order for {row_id}: {frame_order}")
        if not (repo_root / str(row["initial_image"])).is_file():
            raise FileNotFoundError(f"Missing initial image for {row_id}: {row['initial_image']}")
        for frame_index, image, label in zip(frame_order, images, labels):
            if not (repo_root / str(image)).is_file():
                raise FileNotFoundError(f"Missing query image for {row_id}: {image}")
            expected = progress_label(int(frame_index), num_frames)
            if int(label) != expected:
                raise ValueError(f"Invalid progress for {row_id}: got {label}, expected {expected}")


def build_progress_splits(
    *,
    repo_root: Path,
    data_root: Path,
    camera_mode: str,
    level: str,
    task_descriptions: Mapping[str, str],
    tasks: Sequence[str],
    view: str,
    num_frames: int,
    train_per_task: int,
    val_per_task: int,
    test_per_task: int,
    tiny_per_task: int,
    seed: int,
) -> dict[str, list[JsonRecord]]:
    camera_root = data_root / camera_mode
    split_rows: dict[str, list[JsonRecord]] = {"train": [], "val": [], "test": []}
    for task in tasks:
        rollouts = list_rollouts(camera_root, task, level)
        for rollout in rollouts:
            validate_frame_sequence(rollout, view, num_frames)
        task_splits = split_fixed_counts(
            rollouts,
            train_count=train_per_task,
            val_count=val_per_task,
            test_count=test_per_task,
            seed=stable_seed(seed, task),
        )
        for split_name, split_rollouts in task_splits.items():
            for rollout in split_rollouts:
                record = build_progress_record(
                    task=task,
                    task_description=task_descriptions[task],
                    rollout_path=rollout,
                    repo_root=repo_root,
                    view=view,
                    num_frames=num_frames,
                    seed=stable_seed(seed + rollout_number(rollout), task),
                )
                split_rows[split_name].append(record)

    split_rows["tiny"] = select_tiny_per_task(
        split_rows["train"],
        tasks,
        tiny_per_task,
    )
    for rows in split_rows.values():
        validate_progress_records(rows, repo_root, num_frames)
    validate_disjoint_paths(split_rows, "trajectory_path")
    return split_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build temporal task-progress JSONL datasets.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--camera-mode", choices=CAMERA_MODES, default="static")
    parser.add_argument("--level", default="expert")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--train-per-task", type=int, default=DEFAULT_TRAIN_PER_TASK)
    parser.add_argument("--val-per-task", type=int, default=DEFAULT_VAL_PER_TASK)
    parser.add_argument("--test-per-task", type=int, default=DEFAULT_TEST_PER_TASK)
    parser.add_argument("--tiny-per-task", type=int, default=DEFAULT_TINY_PER_TASK)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = project_root()
    config = load_project_config(repo_root)
    seed = int(config["seed"]) if args.seed is None else args.seed
    camera_mode = args.camera_mode
    data_root_value = config["data_root"] if args.data_root is None else args.data_root
    data_root = resolve_path(data_root_value, repo_root)
    output_root = (
        repo_root / "finetune_data" / camera_mode
        if args.output_root is None
        else resolve_path(args.output_root, repo_root)
    )

    task_descriptions = {
        task: str(config["tasks"][task]["description"])
        for task in args.tasks
    }
    split_rows = build_progress_splits(
        repo_root=repo_root,
        data_root=data_root,
        camera_mode=camera_mode,
        level=args.level,
        task_descriptions=task_descriptions,
        tasks=args.tasks,
        view=args.view,
        num_frames=args.num_frames,
        train_per_task=args.train_per_task,
        val_per_task=args.val_per_task,
        test_per_task=args.test_per_task,
        tiny_per_task=args.tiny_per_task,
        seed=seed,
    )
    write_jsonl_files(output_root, split_rows, overwrite=args.overwrite)
    print(f"Created task-progress dataset for camera mode {camera_mode}:")
    for split_name, rows in split_rows.items():
        print(f"  {split_name}: {len(rows)} -> {output_root / f'{split_name}.jsonl'}")


if __name__ == "__main__":
    main()
