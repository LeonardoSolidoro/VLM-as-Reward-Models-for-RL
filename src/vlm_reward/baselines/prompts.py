"""
Prompt and in-context-example construction for the API baseline.
"""
from __future__ import annotations

import random
from pathlib import Path


def sample_frame_indices(num_frames: int, sample_count: int, seed: int) -> list[int]:
    if num_frames < 2:
        raise ValueError(f"A rollout needs at least two frames, got {num_frames}")
    if sample_count < 2:
        raise ValueError(f"sample_count must be at least two, got {sample_count}")
    if sample_count > num_frames:
        raise ValueError(
            f"Requested {sample_count} frames from a rollout containing {num_frames}"
        )
    if sample_count == num_frames:
        return list(range(num_frames))
    rng = random.Random(seed)
    middle = rng.sample(range(1, num_frames - 1), sample_count - 2)
    return [0, *sorted(middle), num_frames - 1]


def temporal_progress_percentages(num_frames: int) -> list[int]:
    if num_frames < 2:
        raise ValueError(f"A rollout needs at least two frames, got {num_frames}")
    return [round(index * 100 / (num_frames - 1)) for index in range(num_frames)]


def list_rollout_images(rollout_dir: Path, view: str) -> list[Path]:
    images = sorted(rollout_dir.glob(f"{view}_frame_*.jpg"))
    if not images:
        raise FileNotFoundError(f"No {view} images found in {rollout_dir}")
    return images


def build_in_context_example(
    rollout_dir: Path,
    view: str,
    sample_count: int,
    seed: int,
    shuffle: bool,
) -> tuple[str, list[Path]]:
    images = list_rollout_images(rollout_dir, view)
    selected = sample_frame_indices(len(images), sample_count, seed)
    labels = temporal_progress_percentages(len(images))
    if shuffle and len(selected) > 1:
        rng = random.Random(seed)
        tail = selected[1:]
        rng.shuffle(tail)
        selected = [selected[0], *tail]

    lines = ["In-context Example (Expert Demo):"]
    selected_images: list[Path] = []
    for prompt_index, source_index in enumerate(selected, start=1):
        lines.append(f"Frame {prompt_index}: [IMG]")
        lines.append(
            "Task Completion Percentage: "
            f"<score>{labels[source_index]}%</score>"
        )
        selected_images.append(images[source_index])
    return "\n".join(lines) + "\n", selected_images
