import os
import json
import glob
import yaml
import asyncio
import aiohttp
import random
import ssl
import certifi
from vlm_api import get_reward_score
from utilities import resolve_camera_data_root, set_all_seeds
from prepare_in_context_examples import prepare_all_in_context_examples

""" 
Run the VLM-based progress inference pipeline.

This script loads trajectory image data, optionally prepares in-context examples,
samples frames from each rollout, builds a multimodal prompt with image
placeholders, sends the prompt and images to a VLM API, parses the predicted task
completion scores, and saves the per-frame scores as JSON files for later metric
evaluation.
"""

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

seed = config.get("seed")
set_all_seeds(seed)

DATA_ROOT = resolve_camera_data_root(
    config.get("data_root"),
    config.get("enable_moving_camera", True),
)

# Extract task descriptions from config
TASK_DESCRIPTIONS = {name: cfg["description"] for name, cfg in config["tasks"].items()}

def load_in_context_example(task, view, num_in_context_examples):
    ic_str = ""
    ic_images = []

    for example_idx in range(num_in_context_examples):
        ic_root = f"in-context-example-{example_idx}"
        ic_path = os.path.join(DATA_ROOT, ic_root, task, "in_context_data.json")

        if not os.path.exists(ic_path):
            print(f"Warning: In-context example file not found for task {task} at {ic_path}. Skipping this example.")
            continue

        with open(ic_path, "r") as f:
            ic_data = json.load(f)

        ic_str += f"In-context Example {example_idx} (Expert Demo):\n"

        for i, frame in enumerate(ic_data):
            ic_str += f"Frame {i}: [IMG]\n"
            ic_str += f"Task Completion Percentage: <score>{frame['percentage']}%</score>\n"

            img_name = frame["images"].get(view)
            if img_name:
                img_path = os.path.join(DATA_ROOT, ic_root, task, img_name)
                ic_images.append(img_path)

        ic_str += "\n"

    return ic_str, ic_images

async def process_rollout(session, semaphore, task, level, rollout, rollout_path, prompt_template, view, frames_in_context, ic_str, ic_images, combined_results, experiment_shuffle_frames):
    """ 
    Process one rollout. 
    Job: One rollout folder -> find image frames -> sample frames -> optionally shuffle them -> built prompt + image list 
    -> query VLM -> parse scores -> store scores in combined_results. 
    """
    print(f"Processing {task} | {level} | {rollout}...")
        
    # Get all frames for the specific view
    all_files = glob.glob(os.path.join(rollout_path, f"{view}_*.jpg"))

    # Extract the frame number from each filename and add it to frame_indices
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
    
    # Optionally shuffle all frames except for the first frame
    first_frame = sorted_indices[0]
    sampled_others = sorted_indices[1:]
    
    if experiment_shuffle_frames:
        random.shuffle(sampled_others)
    else:
        sampled_others = sorted(sampled_others)
            
    # Assembly: First frame is passed separately in the prompt as [IMG]
    image_paths = list(ic_images) # Start image list with in-context images
    image_paths.append(os.path.join(rollout_path, f"{view}_frame_{first_frame:03d}.jpg")) # Add the first anchor frame
    
    # Build query frame prompt text
    frames_list_str = ""
    for i, idx in enumerate(sampled_others):
        # We start enumeration at 1 for the predicted frames
        frames_list_str += f"Frame {i+1}: [IMG]\n"
        path = os.path.join(rollout_path, f"{view}_frame_{idx:03d}.jpg")
        image_paths.append(path)
    
    prompt = prompt_template.format(
        task_description = TASK_DESCRIPTIONS.get(task, task),
        in_context_example = ic_str,
        frames_list = frames_list_str
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        async with semaphore: # Only 2 VLM API requests can run at the same time
            explanation, scores_dict = await get_reward_score(session, prompt, image_paths)
        if explanation is not None and scores_dict is not None:
            if len(scores_dict) != len(sampled_others):
                print(f"Warning: Number of scores extracted ({len(scores_dict)}) does not match number of predicted frames ({len(sampled_others)}).")
            # The first frame is explicitly 0% as per the prompt
            results_list = [{
                "frame": f"{view}_frame_{first_frame:03d}.jpg",
                "score": 0.0
            }]
            
            # Map VLM scores back to original frame filenames
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

            # Store result for this rollout
            combined_results[level][rollout] = results_list
            print(f"Successfully processed {task} | {level} | {rollout} with {len(results_list)} frames scored.")
            return
        print(f"Attempt {attempt + 1} failed for {task} | {rollout}. Retrying...")
        await asyncio.sleep(60)

    print(f"Failed to get reward for {task} | {rollout} after {max_retries} attempts.")

async def run_pipeline():
    output_root = config.get("output_root")
    os.makedirs(output_root, exist_ok = True)

    # Load experiment settings
    experiment_name = config.get("experiment_name")
    view = config.get("views")[0]
    frames_in_context = config.get("frames_in_context")
    prompt_template = config["reward_prompt_template"]
    experiment_levels = config.get("experiment_levels")
    experiment_shuffle_frames = config.get("experiment_shuffle_frames")
    experiment_in_context_examples = config.get("experiment_in_context_examples")

    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data path {DATA_ROOT} does not exist.")
        return
        
    if experiment_in_context_examples > 0:
        print("Preparing in-context examples...")
        prepare_all_in_context_examples()

    # Find all tasks to process
    tasks_to_process = [
            d for d in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, d))
            and not d.startswith("in-context-example")
        ]    
    
    # Limit to 2 concurrent API request to prevent overwhelming the VLM server
    semaphore = asyncio.Semaphore(2)
    
    # Disable default 5-minute timeout for massive generation requests
    timeout = aiohttp.ClientTimeout(total = None)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(timeout = timeout, connector = connector) as session:
        for task in tasks_to_process:
            if experiment_in_context_examples > 0:
                ic_str, ic_images = load_in_context_example(task, view, experiment_in_context_examples)
            else:
                ic_str, ic_images = "", []
                
            task_path = os.path.join(DATA_ROOT, task)
            levels = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
            
            if experiment_levels:
                levels = [lvl for lvl in levels if lvl in experiment_levels]
            
            combined_results = {}
            rollout_tasks = []
            
            for level in levels:
                combined_results[level] = {}
                level_path = os.path.join(task_path, level)
                rollouts = [d for d in os.listdir(level_path) if os.path.isdir(os.path.join(level_path, d))]
                in_context_rollouts = {
                    f"rollout_{i}" for i in range(experiment_in_context_examples)
                }
                for rollout in rollouts:
                    # Skip the rollouts being used as in-context examples
                    if level == "expert" and rollout in in_context_rollouts:
                        print(f"Skipping {task} | {level} | {rollout} (used as in-context example)")
                        continue
                        
                    rollout_path = os.path.join(level_path, rollout)
                    combined_results[level][rollout] = None
                    
                    rollout_tasks.append(process_rollout(
                        session, semaphore, task, level, rollout, rollout_path, prompt_template,
                        view, frames_in_context, ic_str, ic_images, combined_results,
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
