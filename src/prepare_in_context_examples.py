import os
import json
import shutil
import random
import numpy as np
import yaml

from utilities import resolve_camera_data_root, set_all_seeds

""" 
Create in-context examples that later optionally get inserted into the VLM prompt. 
"""

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_ROOT = resolve_camera_data_root(
    config.get("data_root"),
    config.get("enable_moving_camera", True),
)


def prepare_single_in_context_example(
    data_root,
    task,
    rollout_id,
    target_root,
    frames_in_context,
    experiment_shuffle_frames,
):
    source_dir = os.path.join(data_root, task, "expert", rollout_id)
    target_dir = os.path.join(data_root, target_root, task)
    
    if not os.path.exists(source_dir):
        print(f"Warning: {source_dir} not found. Skipping task {task}.")
        return
        
    os.makedirs(target_dir, exist_ok = True)
    
    # Load rewards to get the total number of frames
    rewards_path = os.path.join(source_dir, "rewards.json")
    if not os.path.exists(rewards_path):
        print(f"Warning: {rewards_path} not found. Skipping task {task}.")
        return

    with open(rewards_path, "r") as f:
        rewards = json.load(f)
    num_frames = len(rewards)
    
    # Calculate ground-truth task completion percentages t / T-1
    percentages = np.zeros(num_frames)
    if num_frames > 1:
        for t in range(num_frames):
            percentages[t] = (t / (num_frames - 1)) * 100.0
            
    percentages = np.clip(percentages, 0, 100).astype(int)
    
    # Sample `frames_in_context` frames for the in-context example
    all_indices = list(range(num_frames))
    
    if num_frames > frames_in_context:
        first_frame = all_indices[0]
        last_frame = all_indices[-1]
        middle_frames = all_indices[1:-1]
        
        # Keep first and last frame fixed, sample frames_in_context - 2 from the middle
        sampled_middle = random.sample(middle_frames, frames_in_context - 2)
        sampled_indices = [first_frame] + sorted(sampled_middle + [last_frame])

    else:
        sampled_indices = all_indices
        
    # Shuffle all but the first frame conditionally
    if len(sampled_indices) > 1 and experiment_shuffle_frames:
        rest = sampled_indices[1:]
        random.shuffle(rest)
        final_indices = [sampled_indices[0]] + rest
    else:
        final_indices = sampled_indices
        
    # Save the selected and shuffled data
    in_context_data = []
    
    # We will copy all views just in case, but keep track of the mapping
    views = ["topview", "corner", "corner2"]
    
    # Copy images for each view into the target destination
    for new_idx, original_idx in enumerate(final_indices):
        perc = int(percentages[original_idx])
        
        # Create metadata for one frame
        frame_info = {
            "prompt_index": new_idx,         # New index after sampling/shuffling
            "original_index": original_idx,  # Original frame index in rollout_i
            "percentage": perc,
            "source_rollout": rollout_id,
            "images": {}
        }
        
        for view in views:
            src_img = os.path.join(source_dir, f"{view}_frame_{original_idx:03d}.jpg")
            dst_img = f"{view}_frame_{new_idx:03d}.jpg"
            dst_path = os.path.join(target_dir, dst_img)
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_path)
                frame_info["images"][view] = dst_img
                
        in_context_data.append(frame_info)
        
    with open(os.path.join(target_dir, "in_context_data.json"), "w") as f:
        json.dump(in_context_data, f, indent=4)
        
    print(f"Prepared {target_root} for {task} at {target_dir} with {len(in_context_data)} frames.")


def prepare_all_in_context_examples():
    frames_in_context = config.get("frames_in_context")
    experiment_shuffle_frames = config.get("experiment_shuffle_frames")
    num_in_context_examples = config.get("experiment_in_context_examples")

    if not os.path.exists(DATA_ROOT):
        print(f"Error: {DATA_ROOT} not found.")
        return
    
    # Clear old in-context examples
    for i in range(num_in_context_examples):
        ic_dir = os.path.join(DATA_ROOT, f"in-context-example-{i}")
        if os.path.exists(ic_dir):
            print(f"Clearing existing in-context examples directory: {ic_dir}")
            shutil.rmtree(ic_dir)
    
    # Find all tasks
    tasks = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
        and not d.startswith("in-context-example")
    ]
    
    for task in tasks:
        for i in range(num_in_context_examples):
            rollout_id = f"rollout_{i}"
            target_root = f"in-context-example-{i}"

            prepare_single_in_context_example(
                data_root = DATA_ROOT,
                task = task,
                rollout_id = rollout_id,
                target_root = target_root,
                frames_in_context = frames_in_context,
                experiment_shuffle_frames = experiment_shuffle_frames,
            )


if __name__ == "__main__":
    set_all_seeds(config.get("seed"))

    prepare_all_in_context_examples()