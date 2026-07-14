"""
Shared JSONL and prompt formatting for task-progress datasets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from vlm_reward.training.multimodal import IMAGE_PLACEHOLDER


NUM_QUERY_FRAMES = 19
ProgressRecord = dict[str, Any]


def load_prompt_template(config_path: str | Path) -> str:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(config).__name__}")
    template = config["finetuning_prompt_template"]
    if not isinstance(template, str):
        raise TypeError("finetuning_prompt_template must be a string")
    return template


def load_progress_records(jsonl_path: str | Path) -> list[ProgressRecord]:
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[ProgressRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def build_frames_list(num_frames: int = NUM_QUERY_FRAMES) -> str:
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return "\n".join(
        f"Frame {index}: {IMAGE_PLACEHOLDER}"
        for index in range(1, num_frames + 1)
    )


def build_user_prompt(record: ProgressRecord, prompt_template: str) -> str:
    return prompt_template.format(
        task_description=record["task_description"],
        frames_list=build_frames_list(len(record["images"])),
    )


def build_assistant_answer(
    progress_values: Sequence[int | float],
    prompt_template: str,
) -> str:
    use_score_tags = "<score>" in prompt_template and "</score>" in prompt_template
    blocks: list[str] = []
    for frame_index, progress in enumerate(progress_values, start=1):
        score = f"<score>{progress}%</score>" if use_score_tags else f"{progress}%"
        blocks.append(
            f"Frame {frame_index}:\nTask Completion Percentage: {score}"
        )
    return "\n".join(blocks)
