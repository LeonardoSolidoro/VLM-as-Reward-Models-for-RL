import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml


TASKS = ["PickCube-v1", "PushCube-v1", "PegInsertionSide-v1"]
LEVELS = ["expert", "partial", "random", "regressing"]
VIEW = "topview"
NUM_FRAMES = 20
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TINY_PER_TASK = 2


def load_config(repo_root: Path) -> Dict:
    config_path = repo_root / "configs" / "configs.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def rollout_number(path: Path) -> int:
    return int(path.name.split("_")[-1])


def rel_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def frame_path(rollout_path: Path, frame_idx: int, view: str) -> Path:
    return rollout_path / f"{view}_frame_{frame_idx:03d}.jpg"


def list_rollouts(camera_root: Path, task: str, level: str) -> List[Path]:
    level_root = camera_root / task / level
    if not level_root.exists():
        raise FileNotFoundError(f"Missing level folder: {level_root}")

    rollouts = [p for p in level_root.iterdir() if p.is_dir() and p.name.startswith("rollout_")]
    rollouts.sort(key=rollout_number)
    return rollouts


def validate_rollout(rollout_path: Path, view: str) -> None:
    missing = [
        frame_path(rollout_path, frame_idx, view)
        for frame_idx in range(NUM_FRAMES)
        if not frame_path(rollout_path, frame_idx, view).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{rollout_path} is missing {len(missing)} frame image(s)")

    rewards_path = rollout_path / "rewards.json"
    if not rewards_path.exists():
        raise FileNotFoundError(f"Missing rewards file: {rewards_path}")

    with rewards_path.open("r") as f:
        rewards = json.load(f)

    if len(rewards) != NUM_FRAMES:
        raise ValueError(f"{rewards_path} has {len(rewards)} rewards, expected {NUM_FRAMES}")


def load_reward(rewards: Sequence[float], frame_idx: int, rollout_path: Path) -> float:
    reward = float(rewards[frame_idx])

    if -1e-6 < reward < 0.0:
        reward = 0.0
    if 1.0 < reward < 1.0 + 1e-6:
        reward = 1.0

    if reward < 0.0 or reward > 1.0:
        raise ValueError(f"Reward outside [0, 1] in {rollout_path}: frame {frame_idx}, reward={reward}")

    return round(reward, 4)


def matching_pairs(static_root: Path, moving_root: Path, task: str, level: str, view: str) -> List[Tuple[Path, Path]]:
    static_rollouts = list_rollouts(static_root, task, level)
    moving_rollouts = list_rollouts(moving_root, task, level)
    moving_by_number = {rollout_number(path): path for path in moving_rollouts}

    pairs = []
    for static_rollout in static_rollouts:
        number = rollout_number(static_rollout)
        if number not in moving_by_number:
            raise FileNotFoundError(f"No matching moving rollout for {static_rollout}")

        moving_rollout = moving_by_number[number]
        validate_rollout(static_rollout, view)
        validate_rollout(moving_rollout, view)
        pairs.append((static_rollout, moving_rollout))

    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], seed: int) -> Dict[str, List[Tuple[Path, Path]]]:
    rng = random.Random(seed)
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)

    train_count = int(len(shuffled_pairs) * TRAIN_FRACTION)
    val_count = int(len(shuffled_pairs) * VAL_FRACTION)

    return {
        "train": shuffled_pairs[:train_count],
        "val": shuffled_pairs[train_count:train_count + val_count],
        "test": shuffled_pairs[train_count + val_count:],
    }


def build_record(
    task: str,
    task_description: str,
    level: str,
    static_rollout: Path,
    moving_rollout: Path,
    repo_root: Path,
    view: str,
    primary_view: str,
    seed: int,
) -> Dict:
    if primary_view == "moving":
        anchor_rollout = moving_rollout
        positive_rollout = static_rollout
    elif primary_view == "static":
        anchor_rollout = static_rollout
        positive_rollout = moving_rollout
    else:
        raise ValueError(f"Unsupported primary view: {primary_view}")

    with (anchor_rollout / "rewards.json").open("r") as f:
        rewards = json.load(f)

    frame_order = list(range(NUM_FRAMES))
    rng = random.Random(seed)
    rng.shuffle(frame_order)

    images = [rel_path(frame_path(anchor_rollout, frame_idx, view), repo_root) for frame_idx in frame_order]
    images_positive = [rel_path(frame_path(positive_rollout, frame_idx, view), repo_root) for frame_idx in frame_order]
    reward_values = [load_reward(rewards, frame_idx, anchor_rollout) for frame_idx in frame_order]

    rollout_idx = rollout_number(static_rollout)
    return {
        "id": f"{task}_{level}_rollout_{rollout_idx:06d}_{primary_view}",
        "task": task,
        "task_description": task_description,
        "level": level,
        "primary_view": primary_view,
        "trajectory_path_static": rel_path(static_rollout, repo_root),
        "trajectory_path_moving": rel_path(moving_rollout, repo_root),
        "frame_order": frame_order,
        "images": images,
        "images_positive": images_positive,
        "rewards": reward_values,
    }


def validate_records(rows: List[Dict], repo_root: Path) -> None:
    seen_ids = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError(f"Duplicate record id: {row_id}")
        seen_ids.add(row_id)

        if not (len(row["frame_order"]) == len(row["images"]) == len(row["images_positive"]) == len(row["rewards"]) == NUM_FRAMES):
            raise ValueError(f"Bad record lengths for {row_id}")

        for image_path in row["images"]:
            if not (repo_root / image_path).exists():
                raise FileNotFoundError(f"Missing image in {row_id}: {image_path}")
        for image_path in row["images_positive"]:
            if not (repo_root / image_path).exists():
                raise FileNotFoundError(f"Missing positive image in {row_id}: {image_path}")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/reward_contrastive")
    parser.add_argument("--output-root", default="finetune_data/reward_contrastive")
    parser.add_argument("--view", default=VIEW)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-moving-fraction", type=float, default=0.70)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--levels", nargs="+", default=LEVELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(repo_root)
    seed = int(config["seed"]) if args.seed is None else args.seed

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    static_root = data_root / "static"
    moving_root = data_root / "moving_mounted"

    task_descriptions = {
        task: config["tasks"][task]["description"]
        for task in args.tasks
    }

    split_rows: Dict[str, List[Dict]] = {"train": [], "val": [], "test": [], "tiny": []}
    counts: Dict[str, Dict[str, int]] = {}

    for task in args.tasks:
        counts[task] = {}
        for level in args.levels:
            pairs = matching_pairs(static_root, moving_root, task, level, args.view)
            counts[task][level] = len(pairs)
            level_seed = seed + sum(ord(ch) for ch in f"{task}:{level}")
            split_pairs_by_name = split_pairs(pairs, level_seed)

            for split_name, split_pairs_list in split_pairs_by_name.items():
                moving_count = int(len(split_pairs_list) * args.train_moving_fraction) if split_name == "train" else len(split_pairs_list)

                for pair_idx, (static_rollout, moving_rollout) in enumerate(split_pairs_list):
                    primary_view = "moving" if pair_idx < moving_count else "static"
                    record_seed = level_seed + rollout_number(static_rollout) * 17 + pair_idx
                    row = build_record(
                        task=task,
                        task_description=task_descriptions[task],
                        level=level,
                        static_rollout=static_rollout,
                        moving_rollout=moving_rollout,
                        repo_root=repo_root,
                        view=args.view,
                        primary_view=primary_view,
                        seed=record_seed,
                    )
                    split_rows[split_name].append(row)

    split_rows["tiny"] = split_rows["train"][:TINY_PER_TASK * len(args.tasks)]

    for rows in split_rows.values():
        validate_records(rows, repo_root)

    for split_name, rows in split_rows.items():
        write_jsonl(output_root / f"{split_name}.jsonl", rows)

    print("Created Qwen reward contrastive JSONL files:")
    for task in args.tasks:
        print(f"  {task}: {counts[task]}")
    for split_name, rows in split_rows.items():
        print(f"  {split_name}: {len(rows)} -> {rel_path(output_root / f'{split_name}.jsonl', repo_root)}")


if __name__ == "__main__":
    main()

