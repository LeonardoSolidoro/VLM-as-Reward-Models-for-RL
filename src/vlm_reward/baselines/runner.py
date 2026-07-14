"""
Run zero-shot and few-shot VLM task-progress baselines.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import ssl
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import aiohttp
import certifi
import yaml

from vlm_reward.baselines.api import VLMAPIClient, VLMAPISettings
from vlm_reward.baselines.prompts import (
    build_in_context_example,
    list_rollout_images,
    sample_frame_indices,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "baselines"
    / "paper_reproduction.yaml"
)


@dataclass(frozen=True)
class BaselineConfig:
    """
    All parameters needed to reproduce an API progress experiment.
    """

    seed: int
    data_root: Path
    output_root: Path
    experiment_name: str
    view: str
    query_frames: int
    in_context_frames: int
    in_context_examples: int
    shuffle_query_frames: bool
    shuffle_in_context_frames: bool
    levels: tuple[str, ...]
    rollouts_per_task: int
    task_descriptions: dict[str, str]
    prompt_template: str
    concurrency: int
    max_retries: int
    retry_delay_seconds: float
    max_response_tokens: int
    request_timeout_seconds: float

    @classmethod
    def from_yaml(cls, config_path: Path) -> "BaselineConfig":
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise TypeError(f"Expected a YAML mapping in {config_path}")

        tasks = raw["tasks"]
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError("'tasks' must be a non-empty mapping")
        task_descriptions: dict[str, str] = {}
        for task, description in tasks.items():
            if not isinstance(task, str) or not isinstance(description, str):
                raise TypeError("Each task and description must be a string")
            task_descriptions[task] = description

        config = cls(
            seed=int(raw["seed"]),
            data_root=Path(raw["data_root"]).expanduser(),
            output_root=Path(raw["output_root"]).expanduser(),
            experiment_name=str(raw["experiment_name"]),
            view=str(raw["view"]),
            query_frames=int(raw["query_frames"]),
            in_context_frames=int(raw["in_context_frames"]),
            in_context_examples=int(raw["in_context_examples"]),
            shuffle_query_frames=bool(raw["shuffle_query_frames"]),
            shuffle_in_context_frames=bool(raw["shuffle_in_context_frames"]),
            levels=tuple(str(level) for level in raw["levels"]),
            rollouts_per_task=int(raw["rollouts_per_task"]),
            task_descriptions=task_descriptions,
            prompt_template=str(raw["prompt_template"]),
            concurrency=int(raw["concurrency"]),
            max_retries=int(raw["max_retries"]),
            retry_delay_seconds=float(raw["retry_delay_seconds"]),
            max_response_tokens=int(raw["max_response_tokens"]),
            request_timeout_seconds=float(raw["request_timeout_seconds"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.query_frames < 2:
            raise ValueError("query_frames must be at least 2")
        if self.in_context_examples < 0:
            raise ValueError("in_context_examples cannot be negative")
        if self.in_context_examples > 0 and self.in_context_frames < 2:
            raise ValueError("in_context_frames must be at least 2 for few-shot runs")
        if not self.levels:
            raise ValueError("At least one trajectory level is required")
        if self.rollouts_per_task <= self.in_context_examples:
            raise ValueError(
                "rollouts_per_task must exceed in_context_examples so expert "
                "evaluation is non-empty"
            )
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        if self.retry_delay_seconds < 0.0:
            raise ValueError("retry_delay_seconds cannot be negative")
        if self.max_response_tokens < 1:
            raise ValueError("max_response_tokens must be positive")
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be positive")
        required_fields = {"task_description", "in_context_example", "frames_list"}
        missing = [
            field for field in required_fields if "{" + field + "}" not in self.prompt_template
        ]
        if missing:
            raise ValueError(f"prompt_template is missing placeholders: {missing}")


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], byteorder="big")


def _rollout_number(path: Path) -> tuple[int, str]:
    prefix = "rollout_"
    if not path.name.startswith(prefix):
        raise ValueError(f"Unexpected rollout directory name: {path.name}")
    return int(path.name[len(prefix) :]), path.name


def _build_few_shot_context(
    config: BaselineConfig,
    task_dir: Path,
) -> tuple[str, list[Path]]:
    blocks: list[str] = []
    images: list[Path] = []
    for example_index in range(config.in_context_examples):
        rollout = task_dir / "expert" / f"rollout_{example_index}"
        if not rollout.is_dir():
            raise FileNotFoundError(
                f"Few-shot example {example_index} is missing for {task_dir.name}: {rollout}"
            )
        block, block_images = build_in_context_example(
            rollout_dir=rollout,
            view=config.view,
            sample_count=config.in_context_frames,
            seed=_stable_seed(config.seed, task_dir.name, "context", str(example_index)),
            shuffle=config.shuffle_in_context_frames,
        )
        blocks.append(block.replace(
            "In-context Example (Expert Demo):",
            f"In-context Example {example_index + 1} (Expert Demo):",
            1,
        ))
        images.extend(block_images)
    return "\n".join(blocks), images


def _build_query(
    config: BaselineConfig,
    task: str,
    rollout_dir: Path,
    context_text: str,
    context_images: list[Path],
) -> tuple[str, list[Path], list[Path]]:
    all_images = list_rollout_images(rollout_dir, config.view)
    seed = _stable_seed(config.seed, task, rollout_dir.parent.name, rollout_dir.name)
    selected = sample_frame_indices(len(all_images), config.query_frames, seed)
    anchor_index = selected[0]
    query_indices = selected[1:]
    if config.shuffle_query_frames:
        random.Random(seed).shuffle(query_indices)

    query_images = [all_images[index] for index in query_indices]
    frame_lines = [
        f"Frame {prompt_index}: [IMG]"
        for prompt_index in range(1, len(query_images) + 1)
    ]
    prompt = config.prompt_template.format(
        task_description=config.task_descriptions[task],
        in_context_example=context_text,
        frames_list="\n".join(frame_lines),
    )
    image_paths = [*context_images, all_images[anchor_index], *query_images]
    return prompt, image_paths, [all_images[anchor_index], *query_images]


async def _predict_rollout(
    client: VLMAPIClient,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    config: BaselineConfig,
    task: str,
    rollout_dir: Path,
    context_text: str,
    context_images: list[Path],
) -> tuple[str, list[dict[str, Any]]]:
    prompt, image_paths, result_images = _build_query(
        config,
        task,
        rollout_dir,
        context_text,
        context_images,
    )
    expected_frames = len(result_images) - 1
    for attempt in range(1, config.max_retries + 1):
        try:
            async with semaphore:
                _, scores = await client.predict_progress(
                    session=session,
                    prompt=prompt,
                    image_paths=image_paths,
                    expected_frames=expected_frames,
                )
            rows: list[dict[str, Any]] = [
                {"frame": result_images[0].name, "score": 0.0}
            ]
            rows.extend(
                {
                    "frame": image.name,
                    "score": scores[prompt_index],
                }
                for prompt_index, image in enumerate(result_images[1:], start=1)
            )
            LOGGER.info(
                "Scored %s/%s/%s (%d predicted frames)",
                task,
                rollout_dir.parent.name,
                rollout_dir.name,
                expected_frames,
            )
            return rollout_dir.name, rows
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            LOGGER.exception(
                "API attempt %d/%d failed for %s/%s/%s: %s",
                attempt,
                config.max_retries,
                task,
                rollout_dir.parent.name,
                rollout_dir.name,
                error,
            )
            if attempt == config.max_retries:
                raise RuntimeError(
                    f"Failed to score {task}/{rollout_dir.parent.name}/{rollout_dir.name}"
                ) from error
            await asyncio.sleep(config.retry_delay_seconds)
    raise AssertionError("Retry loop ended unexpectedly")


async def run_experiment(config: BaselineConfig) -> list[Path]:
    """
    Run one configured zero/few-shot experiment and return its result files.
    """

    if not config.data_root.is_dir():
        raise FileNotFoundError(config.data_root)
    output_dir = config.output_root / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    client = VLMAPIClient(
        VLMAPISettings.from_environment(max_tokens=config.max_response_tokens)
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    timeout = aiohttp.ClientTimeout(total=config.request_timeout_seconds)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    output_files: list[Path] = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for task in config.task_descriptions:
            task_dir = config.data_root / task
            if not task_dir.is_dir():
                raise FileNotFoundError(task_dir)
            context_text, context_images = _build_few_shot_context(config, task_dir)
            skipped_context_rollouts = {
                f"rollout_{index}" for index in range(config.in_context_examples)
            }

            for level in config.levels:
                level_dir = task_dir / level
                if not level_dir.is_dir():
                    raise FileNotFoundError(level_dir)
                rollout_dirs = sorted(
                    (path for path in level_dir.iterdir() if path.is_dir()),
                    key=_rollout_number,
                )
                if len(rollout_dirs) < config.rollouts_per_task:
                    raise ValueError(
                        f"{level_dir} contains {len(rollout_dirs)} rollouts, but "
                        f"{config.rollouts_per_task} are required"
                    )
                rollout_dirs = rollout_dirs[: config.rollouts_per_task]
                if level == "expert":
                    rollout_dirs = [
                        path
                        for path in rollout_dirs
                        if path.name not in skipped_context_rollouts
                    ]
                if not rollout_dirs:
                    raise ValueError(f"No evaluation rollouts remain in {level_dir}")

                predictions = await asyncio.gather(
                    *(
                        _predict_rollout(
                            client=client,
                            session=session,
                            semaphore=semaphore,
                            config=config,
                            task=task,
                            rollout_dir=rollout_dir,
                            context_text=context_text,
                            context_images=context_images,
                        )
                        for rollout_dir in rollout_dirs
                    )
                )
                result_file = output_dir / f"{task}_{level}_rewards.json"
                payload = {
                    "task": task,
                    "step": "eval",
                    "level": level,
                    "results": dict(predictions),
                }
                result_file.write_text(json.dumps(payload, indent=2) + "\n")
                output_files.append(result_file)
                LOGGER.info("Saved %s", result_file)
    return output_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero/few-shot VLM task-progress paper reproduction."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the dataset root used for queries and few-shot examples.",
    )
    parser.add_argument(
        "--in-context-examples",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Override the zero/one/two-shot setting in the YAML config.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the output directory name in the YAML config.",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    config = BaselineConfig.from_yaml(args.config)
    if args.data_root is not None:
        config = replace(config, data_root=args.data_root.expanduser())
    if args.in_context_examples is not None:
        config = replace(config, in_context_examples=args.in_context_examples)
    if args.experiment_name is not None:
        config = replace(config, experiment_name=args.experiment_name)
    config.validate()
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
