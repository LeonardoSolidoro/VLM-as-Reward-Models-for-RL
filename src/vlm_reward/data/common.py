"""
Shared, simulator-independent helpers for data pipelines.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import yaml

from vlm_reward.runtime import set_global_seed as seed_everything


DEFAULT_TASKS = ("PickCube-v1", "PushCube-v1", "PegInsertionSide-v1")
REWARD_LEVELS = ("expert", "partial", "random", "regressing")
CAMERA_MODES = ("static", "moving", "moving_mounted")
DEFAULT_VIEW = "topview"
DEFAULT_NUM_FRAMES = 20

JsonRecord = dict[str, Any]
T = TypeVar("T")

_ROLLOUT_PATTERN = re.compile(r"rollout_(\d+)")


def project_root() -> Path:
    """
    Return the repository root, independent of the current directory.
    """
    return Path(__file__).resolve().parents[3]


def load_project_config(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / "configs" / "configs.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {config_path}, got {type(config).__name__}")
    return config


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else repo_root / resolved


def relative_path(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def rollout_number(path: Path) -> int:
    match = _ROLLOUT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Expected a directory named rollout_<number>, got {path}")
    return int(match.group(1))


def frame_path(rollout_path: Path, frame_index: int, view: str) -> Path:
    if frame_index < 0:
        raise ValueError(f"Frame index must be non-negative, got {frame_index}")
    if not view:
        raise ValueError("View name must not be empty")
    return rollout_path / f"{view}_frame_{frame_index:03d}.jpg"


def list_rollouts(camera_root: Path, task: str, level: str) -> list[Path]:
    level_root = camera_root / task / level
    if not level_root.is_dir():
        raise FileNotFoundError(f"Missing rollout directory: {level_root}")

    rollouts = [
        path
        for path in level_root.iterdir()
        if path.is_dir() and _ROLLOUT_PATTERN.fullmatch(path.name) is not None
    ]
    rollouts.sort(key=rollout_number)
    return rollouts


def validate_frame_sequence(rollout_path: Path, view: str, num_frames: int) -> None:
    if num_frames < 2:
        raise ValueError(f"At least two frames are required, got {num_frames}")

    expected = [frame_path(rollout_path, index, view) for index in range(num_frames)]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{rollout_path} is missing {len(missing)} expected {view} frame(s); "
            f"first missing file: {missing[0]}"
        )

    actual = list(rollout_path.glob(f"{view}_frame_*.jpg"))
    if len(actual) != num_frames:
        raise ValueError(
            f"{rollout_path} contains {len(actual)} {view} frames, expected exactly {num_frames}"
        )


def stable_seed(base_seed: int, *labels: str) -> int:
    return base_seed + sum(ord(character) for label in labels for character in label)


def shuffled(items: Sequence[T], seed: int) -> list[T]:
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


def split_fixed_counts(
    items: Sequence[T],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[T]]:
    counts = (train_count, val_count, test_count)
    if any(count < 0 for count in counts):
        raise ValueError(f"Split counts must be non-negative, got {counts}")
    required = sum(counts)
    if len(items) < required:
        raise ValueError(f"Only {len(items)} items are available, but {required} are required")

    ordered = shuffled(items, seed)
    train_end = train_count
    val_end = train_end + val_count
    test_end = val_end + test_count
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:test_end],
    }


def split_fractions(
    items: Sequence[T],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[T]]:
    if train_fraction < 0.0 or val_fraction < 0.0:
        raise ValueError("Split fractions must be non-negative")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train_fraction + val_fraction must be below 1, got {train_fraction + val_fraction}"
        )

    ordered = shuffled(items, seed)
    train_count = int(len(ordered) * train_fraction)
    val_count = int(len(ordered) * val_fraction)
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count : train_count + val_count],
        "test": ordered[train_count + val_count :],
    }


def select_tiny_per_task(
    train_rows: Sequence[JsonRecord],
    tasks: Sequence[str],
    count_per_task: int,
) -> list[JsonRecord]:
    if count_per_task < 1:
        raise ValueError(f"count_per_task must be positive, got {count_per_task}")

    tiny: list[JsonRecord] = []
    for task in tasks:
        task_rows = [row for row in train_rows if row["task"] == task]
        if len(task_rows) < count_per_task:
            raise ValueError(
                f"Task {task} has only {len(task_rows)} training rows, need {count_per_task}"
            )
        tiny.extend(task_rows[:count_per_task])
    return tiny


def validate_unique_ids(rows: Iterable[JsonRecord]) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"Duplicate record id: {row_id}")
        seen.add(row_id)


def validate_disjoint_paths(
    split_rows: Mapping[str, Sequence[JsonRecord]],
    path_key: str,
) -> None:
    owners: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        rows = split_rows[split_name]
        for row in rows:
            trajectory_path = str(row[path_key])
            if trajectory_path in owners:
                raise ValueError(
                    f"Trajectory appears in both {owners[trajectory_path]} and {split_name}: "
                    f"{trajectory_path}"
                )
            owners[trajectory_path] = split_name


def write_jsonl_files(
    output_root: Path,
    split_rows: Mapping[str, Sequence[JsonRecord]],
    *,
    overwrite: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_root / f"{name}.jsonl" for name in split_rows}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing dataset file(s); "
            "pass --overwrite to replace them"
        )

    for split_name, rows in split_rows.items():
        destination = output_paths[split_name]
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        temporary.replace(destination)


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)
