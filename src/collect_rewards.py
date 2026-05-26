import os
import json
import glob
import yaml
import asyncio
import aiohttp
import random
from reward_function import get_reward_score
from utilities import set_all_seeds

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

seed = config.get("seed")
set_all_seeds(seed)

# Extract task descriptions from config
TASK_DESCRIPTIONS = {name: cfg["description"] for name, cfg in config["tasks"].items()}

def load_in_context_example(task, view):
    ic_path = os.path.join("data", "metaworld", "in-context-example", task, "in_context_data.json")
    if not os.path.exists(ic_path):
        return "", []
    with open(ic_path, "r") as f:
        ic_data = json.load(f)
        
    ic_str = "In-context Example (Expert Demo):\n"
    ic_images = []
    
    for i, frame in enumerate(ic_data):
        ic_str += f"Frame {i}: [IMG]\n"
        ic_str += f"Task Completion Percentage: <score>{frame['percentage']}%</score>\n"
        
        img_name = frame["images"].get(view)
        if img_name:
            img_path = os.path.join("data", "metaworld", "in-context-example", task, img_name)
            ic_images.append(img_path)
            
    return ic_str, ic_images

async def process_rollout(session, semaphore, task, level, rollout, rollout_path, prompt_template, views, frames_in_context, ic_str, ic_images, combined_results, experiment_shuffle_frames):
    async with semaphore:
        print(f"Processing {task} | {level} | {rollout}...")
    
    view = views[0] if views else "topview"
    
    # Get all frames for the specific view
    all_files = glob.glob(os.path.join(rollout_path, f"{view}_*.jpg"))
    frame_indices = set()
    for f in all_files:
        basename = os.path.basename(f)
        try:
            parts = basename.replace(".jpg", "").split("_frame_")
            frame_indices.add(int(parts[1]))
        except:
            continue
    
    if not frame_indices:
        return
        
    sorted_indices = sorted(list(frame_indices))
    
    # Guarantee the first and last frames are always included
    first_frame = sorted_indices[0]
    last_frame = sorted_indices[-1]
    
    middle_frames = sorted_indices[1:-1]
    
    # Subsample remaining frames from the middle
    if len(middle_frames) > frames_in_context - 2:
        sampled_others = random.sample(middle_frames, frames_in_context - 2)
        sampled_others.append(last_frame)
    else:
        sampled_others = middle_frames + [last_frame]

    # Shuffle the remaining frames conditionally
    if experiment_shuffle_frames:
        random.shuffle(sampled_others)
    else:
        sampled_others = sorted(sampled_others)
    
    # Assembly: First frame is passed separately in the prompt as [IMG]
    image_paths = list(ic_images)
    image_paths.append(os.path.join(rollout_path, f"{view}_frame_{first_frame:03d}.jpg"))
    
    frames_list_str = ""
    for i, idx in enumerate(sampled_others):
        # We start enumeration at 1 for the predicted frames
        frames_list_str += f"Frame {i+1}: [IMG]\n"
        path = os.path.join(rollout_path, f"{view}_frame_{idx:03d}.jpg")
        image_paths.append(path)
    
    prompt = prompt_template.format(
        task_description=TASK_DESCRIPTIONS.get(task, task),
        in_context_example=ic_str,
        frames_list=frames_list_str
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        explanation, scores_dict = await get_reward_score(session, prompt, image_paths)
        if explanation is not None and scores_dict is not None:
            # The first frame is explicitly 0% as per the prompt
            results_list = [{
                "frame": f"{view}_frame_{first_frame:03d}.jpg",
                "score": 0.0
            }]
            
            # The remaining frames were enumerated starting at 1
            for i, original_idx in enumerate(sampled_others):
                score = scores_dict.get(i + 1, None)
                if score is not None:
                    results_list.append({
                        "frame": f"{view}_frame_{original_idx:03d}.jpg",
                        "score": score
                    })
                else:
                    print(f"Error: Frame {view}_frame_{original_idx:03d}.jpg (prompt index {i+1}) was dropped for {task} | {rollout} due to missing score in VLM output!")
            
            
            #print(f"\n--- DEBUG: Raw VLM Output for {task} | {rollout} ---")
            #print(explanation)
            #print("---------------------------------------------------\n")

            combined_results[level][rollout] = results_list
            return
        print(f"Attempt {attempt + 1} failed for {task} | {rollout}. Retrying...")
        await asyncio.sleep(30)

    print(f"Failed to get reward for {task} | {rollout} after {max_retries} attempts.")

async def run_pipeline():
    data_root = config.get("data_root")
    output_root = config.get("output_root")
    os.makedirs(output_root, exist_ok=True)

    experiment_name = config.get("experiment_name", "exp_default")
    views = config.get("views", ["topview"])
    frames_in_context = config.get("frames_in_context", 30)
    prompt_template = config["reward_prompt_template"]
    experiment_levels = config.get("experiment_levels", None)
    experiment_shuffle_frames = config.get("experiment_shuffle_frames", True)
    experiment_in_context_examples = config.get("experiment_in_context_examples", 1)

    view = views[0] if views else "topview"

    if not os.path.exists(data_root):
        print(f"Error: Data path {data_root} does not exist.")
        return
        
    tasks_to_process = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d != "in-context-example"]
    
    # Limit to 1 concurrent API request to prevent overwhelming the VLM server
    semaphore = asyncio.Semaphore(1)
    
    # Disable default 5-minute timeout for massive generation requests
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for task in tasks_to_process:
            if experiment_in_context_examples > 0:
                ic_str, ic_images = load_in_context_example(task, view)
            else:
                ic_str, ic_images = "", []
                
            task_path = os.path.join(data_root, task)
            levels = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
            
            if experiment_levels:
                levels = [lvl for lvl in levels if lvl in experiment_levels]
            
            combined_results = {}
            rollout_tasks = []
            
            for level in levels:
                combined_results[level] = {}
                level_path = os.path.join(task_path, level)
                rollouts = [d for d in os.listdir(level_path) if os.path.isdir(os.path.join(level_path, d))]
                
                for rollout in rollouts:
                    # Skip rollout_0 if we are using it for in-context examples
                    if rollout == "rollout_0" and experiment_in_context_examples > 0:
                        print(f"Skipping {task} | {level} | {rollout} (used as in-context example)")
                        continue
                        
                    rollout_path = os.path.join(level_path, rollout)
                    combined_results[level][rollout] = None
                    
                    rollout_tasks.append(process_rollout(
                        session, semaphore, task, level, rollout, rollout_path, prompt_template,
                        views, frames_in_context, ic_str, ic_images, combined_results,
                        experiment_shuffle_frames
                    ))
            
            if rollout_tasks:
                await asyncio.gather(*rollout_tasks)

            # Save results
            for level, results in combined_results.items():
                # Clean up None results
                clean_results = {k: v for k, v in results.items() if v is not None}
                output_file = os.path.join(output_root, experiment_name, f"{task}_{level}_rewards.json")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump({
                        "task": task,
                        "step": "eval",
                        "level": level,
                        "results": clean_results
                    }, f, indent=4)
                print(f"Saved results of experiment {experiment_name} for {task} ({level}) to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
