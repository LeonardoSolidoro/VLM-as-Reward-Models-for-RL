import json
import math
import random
from pathlib import Path
import yaml

TASKS = ["PickCube-v1", "PushCube-v1", "PegInsertionSide-v1"]
VIEW_STATIC = "topview"
VIEW_MOVING = "topview" 
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

def parse_reward(reward: float) -> int:
    """Parses environment GT float reward into a 0-100 percentage.
    Very small noise values < 0.001 are floored to 0. Otherwise it rounds up.
    Handles floats, scientific notation natively parsed by json/float()."""
    val = float(reward)
    if val < 0.001:
        return 0
    return min(100, int(math.ceil(val * 100)))

def find_rollouts(data_root, task):
    task_root = data_root / task / LEVEL
    if not task_root.exists():
        raise FileNotFoundError(f"Missing task folder: {task_root}")

    rollouts = [p for p in task_root.iterdir() if p.is_dir() and p.name.startswith("rollout_")]
    rollouts.sort(key=rollout_number)
    return rollouts

def frame_path(rollout_path, view_name, frame_idx):
    return rollout_path / f"{view_name}_frame_{frame_idx:03d}.jpg"

def check_rollout(rollout_path_static, rollout_path_moving, view_name):
    # Check static
    frame_files_static = list(rollout_path_static.glob(f"{view_name}_frame_*.jpg"))
    if len(frame_files_static) != NUM_FRAMES:
        return False, f"found {len(frame_files_static)} {view_name} static images"

    missing_static = [frame_path(rollout_path_static, view_name, i) for i in range(NUM_FRAMES) if not frame_path(rollout_path_static, view_name, i).exists()]
    if missing_static:
        return False, f"missing {len(missing_static)} expected static frame image(s)"

    # Check moving
    frame_files_moving = list(rollout_path_moving.glob(f"{view_name}_frame_*.jpg"))
    if len(frame_files_moving) != NUM_FRAMES:
        return False, f"found {len(frame_files_moving)} {view_name} moving images"

    missing_moving = [frame_path(rollout_path_moving, view_name, i) for i in range(NUM_FRAMES) if not frame_path(rollout_path_moving, view_name, i).exists()]
    if missing_moving:
        return False, f"missing {len(missing_moving)} expected moving frame image(s)"

    rewards_path = rollout_path_static / "rewards.json"
    if not rewards_path.exists():
        return False, "missing rewards.json"
        
    try:
        with open(rewards_path, "r") as f:
            rewards = json.load(f)
        if len(rewards) != NUM_FRAMES:
            return False, f"rewards.json has {len(rewards)} entries, expected {NUM_FRAMES}"
    except Exception as e:
        return False, f"failed to parse rewards.json: {e}"

    return True, ""

def build_record(task, task_description, rollout_path_static, rollout_path_moving, repo_root, seed, view_name, primary_view_type):
    ok, reason = check_rollout(rollout_path_static, rollout_path_moving, view_name)
    if not ok:
        raise ValueError(f"Invalid rollout {rollout_path_static}: {reason}")

    frame_order = list(range(1, NUM_FRAMES))
    random.Random(seed).shuffle(frame_order)

    images_static = [rel_path(frame_path(rollout_path_static, view_name, idx), repo_root) for idx in frame_order]
    images_moving = [rel_path(frame_path(rollout_path_moving, view_name, idx), repo_root) for idx in frame_order]
    
    if primary_view_type == "moving":
        images_anchor = images_moving
        images_positive = images_static
        initial_image = rel_path(frame_path(rollout_path_moving, view_name, 0), repo_root)
    else:
        images_anchor = images_static
        images_positive = images_moving
        initial_image = rel_path(frame_path(rollout_path_static, view_name, 0), repo_root)
        
    with open(rollout_path_static / "rewards.json", "r") as f:
        rewards = json.load(f)
        
    progress = [parse_reward(rewards[idx]) for idx in frame_order]
    rollout_idx = rollout_number(rollout_path_static)

    return {
        "id": f"{task}_traj_{rollout_idx:06d}_{primary_view_type}_primary",
        "task": task,
        "task_description": task_description,
        "trajectory_path_static": rel_path(rollout_path_static, repo_root),
        "trajectory_path_moving": rel_path(rollout_path_moving, repo_root),
        "initial_image": initial_image,
        "frame_order": frame_order,
        "images": images_anchor,  # anchor images (primary for LLM)
        "images_positive": images_positive, # positive images (secondary for Contrastive)
        "progress": progress,
        "primary_view": primary_view_type,
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

        if not (len(row["frame_order"]) == len(row["images"]) == len(row["images_positive"]) == len(row["progress"]) == QUERY_FRAMES):
            raise ValueError(f"Bad record lengths for {row['id']}")

        if sorted(row["frame_order"]) != list(range(1, NUM_FRAMES)):
            raise ValueError(f"Bad frame_order for {row['id']}")

        for frame_idx, image_path, pos_image_path, progress in zip(row["frame_order"], row["images"], row["images_positive"], row["progress"]):
            if not (repo_root / image_path).exists():
                raise FileNotFoundError(f"Missing anchor image path in {row['id']}: {image_path}")
            if not (repo_root / pos_image_path).exists():
                raise FileNotFoundError(f"Missing positive image path in {row['id']}: {pos_image_path}")

            if not isinstance(progress, int) or not (0 <= progress <= 100):
                raise ValueError(f"Bad progress in {row['id']}: got {progress}, must be int between 0 and 100")

def validate_disjoint(split_rows):
    split_paths = {}
    for split_name, rows in split_rows.items():
        if split_name == "tiny":
            continue

        for row in rows:
            path = row["trajectory_path_static"]
            if path in split_paths and split_paths[path] != split_name:
                raise ValueError(f"Trajectory appears in both {split_paths[path]} and {split_name}: {path}")
            split_paths[path] = split_name

def main():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root)
    seed = int(config["seed"])

    data_root_static = repo_root / config["data_root"] / "static"
    data_root_moving = repo_root / config["data_root"] / "moving_mounted"
    
    output_root = repo_root / "finetune_data_contrastive" / "moving_mounted"
    output_root.mkdir(exist_ok=True)

    task_descriptions = {
        task: config["tasks"][task]["description"]
        for task in TASKS
    }

    view_name = config["views"][0]

    split_rows = {"train": [], "val": [], "test": [], "tiny": []}
    found_counts = {}
    skipped = {}

    for task in TASKS:
        rollouts_static = find_rollouts(data_root_static, task)
        rollouts_moving = find_rollouts(data_root_moving, task)
        
        moving_by_num = {rollout_number(p): p for p in rollouts_moving}
        
        found_counts[task] = 0
        skipped[task] = []

        valid_rollouts = []
        for r_static in rollouts_static:
            r_num = rollout_number(r_static)
            if r_num not in moving_by_num:
                skipped[task].append((r_static, "missing matching moving rollout"))
                continue
                
            r_moving = moving_by_num[r_num]
            ok, reason = check_rollout(r_static, r_moving, view_name)
            if ok:
                valid_rollouts.append((r_static, r_moving))
                found_counts[task] += 1
            else:
                skipped[task].append((r_static, reason))

        needed = TRAIN_PER_TASK + VAL_PER_TASK + TEST_PER_TASK
        if len(valid_rollouts) < needed:
            raise ValueError(f"{task} has {len(valid_rollouts)} valid paired rollouts, but {needed} are needed")

        rng = random.Random(seed + sum(ord(ch) for ch in task))
        rng.shuffle(valid_rollouts)

        split_rollouts = {
            "train": valid_rollouts[:TRAIN_PER_TASK],
            "val": valid_rollouts[TRAIN_PER_TASK:TRAIN_PER_TASK + VAL_PER_TASK],
            "test": valid_rollouts[TRAIN_PER_TASK + VAL_PER_TASK:needed],
        }
        split_rollouts["tiny"] = split_rollouts["train"][:TINY_PER_TASK]

        for split_name, rollout_pairs in split_rollouts.items():
            if split_name == "train":
                num_moving = int(len(rollout_pairs) * 0.7)
            else:
                num_moving = len(rollout_pairs)
            
            moving_pairs = rollout_pairs[:num_moving]
            static_pairs = rollout_pairs[num_moving:]
            
            for r_static, r_moving in moving_pairs:
                frame_seed = seed + rollout_number(r_static) + sum(ord(ch) for ch in task)
                row = build_record(task, task_descriptions[task], r_static, r_moving, repo_root, frame_seed, view_name, "moving")
                split_rows[split_name].append(row)
                
            for r_static, r_moving in static_pairs:
                frame_seed = seed + rollout_number(r_static) + sum(ord(ch) for ch in task)
                row = build_record(task, task_descriptions[task], r_static, r_moving, repo_root, frame_seed, view_name, "static")
                split_rows[split_name].append(row)

    for rows in split_rows.values():
        validate_records(rows, repo_root)
    validate_disjoint(split_rows)

    for split_name, rows in split_rows.items():
        write_jsonl(output_root / f"{split_name}.jsonl", rows)

    print("Created Qwen contrastive finetune JSONL files:")
    print("Trajectories paired per task:")
    for task in TASKS:
        print(f"  {task}: {found_counts[task]} found, {len(skipped[task])} skipped")
    print("Trajectories used per split:")
    for split_name, rows in split_rows.items():
        print(f"  {split_name}: {len(rows)} trajectories -> {rel_path(output_root / (split_name + '.jsonl'), repo_root)}")

if __name__ == "__main__":
    main()
