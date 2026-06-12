import json
import random
from pathlib import Path

import yaml


TASKS = ["PickCube-v1", "PushCube-v1", "PegInsertionSide-v1"]
VIEW = "topview"
LEVEL = "expert"
NUM_FRAMES = 20
QUERY_FRAMES = 19
TRAIN_PER_TASK = 400
VAL_PER_TASK = 50
TEST_PER_TASK = 50
TINY_PER_TASK = 2


def load_config(repo_root):
    config_path = repo_root / "configs" / "configs.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def rollout_number(path):
    return int(path.name.split("_")[-1])


def rel_path(path, repo_root):
    return path.relative_to(repo_root).as_posix()


def find_rollouts(data_root, task):
    task_root = data_root / task / LEVEL
    if not task_root.exists():
        raise FileNotFoundError(f"Missing task folder: {task_root}")

    rollouts = [p for p in task_root.iterdir() if p.is_dir() and p.name.startswith("rollout_")]
    rollouts.sort(key=rollout_number)
    return rollouts


def frame_path(rollout_path, frame_idx):
    return rollout_path / f"{VIEW}_frame_{frame_idx:03d}.jpg"


def check_rollout(rollout_path):
    frame_files = list(rollout_path.glob(f"{VIEW}_frame_*.jpg"))
    if len(frame_files) != NUM_FRAMES:
        return False, f"found {len(frame_files)} {VIEW} images"

    missing = [frame_path(rollout_path, i) for i in range(NUM_FRAMES) if not frame_path(rollout_path, i).exists()]
    if missing:
        return False, f"missing {len(missing)} expected frame image(s)"

    return True, ""


def build_record(task, task_description, rollout_path, repo_root, seed):
    ok, reason = check_rollout(rollout_path)
    if not ok:
        raise ValueError(f"Invalid rollout {rollout_path}: {reason}")

    frame_order = list(range(1, NUM_FRAMES))
    random.Random(seed).shuffle(frame_order)

    images = [rel_path(frame_path(rollout_path, idx), repo_root) for idx in frame_order]
    progress = [round(idx / (NUM_FRAMES - 1) * 100) for idx in frame_order]
    rollout_idx = rollout_number(rollout_path)

    return {
        "id": f"{task}_traj_{rollout_idx:06d}",
        "task": task,
        "task_description": task_description,
        "trajectory_path": rel_path(rollout_path, repo_root),
        "initial_image": rel_path(frame_path(rollout_path, 0), repo_root),
        "frame_order": frame_order,
        "images": images,
        "progress": progress,
    }


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def validate_records(rows, repo_root):
    seen_ids = set()
    for row in rows:
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate record id: {row['id']}")
        seen_ids.add(row["id"])

        if not (repo_root / row["initial_image"]).exists():
            raise FileNotFoundError(f"Missing initial image in {row['id']}: {row['initial_image']}")

        if not (len(row["frame_order"]) == len(row["images"]) == len(row["progress"]) == QUERY_FRAMES):
            raise ValueError(f"Bad record lengths for {row['id']}")

        if sorted(row["frame_order"]) != list(range(1, NUM_FRAMES)):
            raise ValueError(f"Bad frame_order for {row['id']}")

        for frame_idx, image_path, progress in zip(row["frame_order"], row["images"], row["progress"]):
            if not (repo_root / image_path).exists():
                raise FileNotFoundError(f"Missing image path in {row['id']}: {image_path}")

            expected_progress = round(frame_idx / (NUM_FRAMES - 1) * 100)
            if progress != expected_progress:
                raise ValueError(f"Bad progress in {row['id']}: got {progress}, expected {expected_progress}")


def validate_disjoint(split_rows):
    split_paths = {}
    for split_name, rows in split_rows.items():
        if split_name == "tiny":
            continue

        for row in rows:
            path = row["trajectory_path"]
            if path in split_paths:
                raise ValueError(f"Trajectory appears in both {split_paths[path]} and {split_name}: {path}")
            split_paths[path] = split_name


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root)
    seed = int(config.get("seed", 42))

    enable_moving_camera = config.get("enable_moving_camera")
    camera_type = "moving" if enable_moving_camera else "static"

    data_root = repo_root / config.get("data_root", "data") / camera_type
    output_root = repo_root / ("finetune_data_moving" if enable_moving_camera else "finetune_data")
    output_root.mkdir(exist_ok=True)

    task_descriptions = {
        task: config["tasks"][task]["description"]
        for task in TASKS
    }

    split_rows = {"train": [], "val": [], "test": [], "tiny": []}
    found_counts = {}
    skipped = {}

    for task in TASKS:
        rollouts = find_rollouts(data_root, task)
        found_counts[task] = len(rollouts)
        skipped[task] = []

        valid_rollouts = []
        for rollout_path in rollouts:
            ok, reason = check_rollout(rollout_path)
            if ok:
                valid_rollouts.append(rollout_path)
            else:
                skipped[task].append((rollout_path, reason))

        needed = TRAIN_PER_TASK + VAL_PER_TASK + TEST_PER_TASK
        if len(valid_rollouts) < needed:
            raise ValueError(f"{task} has {len(valid_rollouts)} valid rollouts, but {needed} are needed")

        rng = random.Random(seed + sum(ord(ch) for ch in task))
        rng.shuffle(valid_rollouts)

        split_rollouts = {
            "train": valid_rollouts[:TRAIN_PER_TASK],
            "val": valid_rollouts[TRAIN_PER_TASK:TRAIN_PER_TASK + VAL_PER_TASK],
            "test": valid_rollouts[TRAIN_PER_TASK + VAL_PER_TASK:needed],
        }
        # The 400/50/50 split uses all 500 rollouts, so tiny is a debug subset.
        split_rollouts["tiny"] = split_rollouts["train"][:TINY_PER_TASK]

        for split_name, rollout_paths in split_rollouts.items():
            for rollout_path in rollout_paths:
                frame_seed = seed + rollout_number(rollout_path) + sum(ord(ch) for ch in task)
                row = build_record(task, task_descriptions[task], rollout_path, repo_root, frame_seed)
                split_rows[split_name].append(row)

    for rows in split_rows.values():
        validate_records(rows, repo_root)
    validate_disjoint(split_rows)

    for split_name, rows in split_rows.items():
        write_jsonl(output_root / f"{split_name}.jsonl", rows)

    print("Created Qwen progress finetune JSONL files:")
    print("Trajectories found per task:")
    for task in TASKS:
        print(f"  {task}: {found_counts[task]} found, {len(skipped[task])} skipped")
        for rollout_path, reason in skipped[task][:5]:
            print(f"    skipped {rel_path(rollout_path, repo_root)}: {reason}")
        if len(skipped[task]) > 5:
            print(f"    ... {len(skipped[task]) - 5} more skipped")
    print("Trajectories used per split:")
    for split_name, rows in split_rows.items():
        print(f"  {split_name}: {len(rows)} trajectories -> {rel_path(output_root / (split_name + '.jsonl'), repo_root)}")
    print("Note: tiny is a deterministic debug subset of train because each task has exactly 500 trajectories.")


if __name__ == "__main__":
    main()
