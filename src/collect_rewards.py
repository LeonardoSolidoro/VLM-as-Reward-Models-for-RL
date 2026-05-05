import os
import json
import glob
import yaml
import asyncio
import aiohttp
from reward_function import get_reward_score

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Extract task descriptions from config
TASK_DESCRIPTIONS = {name: cfg["description"] for name, cfg in config["tasks"].items()}

async def process_rollout(session, task, level, rollout, rollout_path, prompt, use_initial, step_sizes, interval, combined_results):
    print(f"Processing {task} | {level} | {rollout}...")
    
    # Get all frames, sorted
    frames = sorted(glob.glob(os.path.join(rollout_path, "frame_*.jpg")))
    if not frames:
        return

    tasks = []
    
    # helper to wrap get_reward_score and store results
    async def get_and_store(step_key, frame2_path, frame1_path):
        max_retries = 5
        semaphore = asyncio.Semaphore(5)  # Limit concurrency to 5 requests
        async with semaphore:
            for attempt in range(max_retries):
                explanation, score = await get_reward_score(session, frame1_path, frame2_path, prompt)
                if explanation is not None and score is not None:
                    combined_results[step_key][level][rollout].append({
                        "frame": os.path.basename(frame2_path),
                        "score": score,
                        "explanation": explanation
                    })
                    return
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying {task} | {rollout} | {os.path.basename(frame2_path)} in {wait_time}s... (Attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(wait_time)
            
            print(f"Failed to get reward for {task} | {rollout} | {os.path.basename(frame2_path)} after {max_retries} attempts.")

    # Evaluation loop every 'interval' frames
    for i in range(0, len(frames), interval):
        frame2_path = frames[i]
        
        # 1. Fixed Initial Frame logic
        if use_initial:
            frame1_path = frames[0]
            tasks.append(get_and_store("initial", frame2_path, frame1_path))

        # 2. Dynamic Step Size logic
        for step in step_sizes:
            idx1 = max(0, i - step)
            if idx1 == i and i > 0:
                continue
                
            frame1_path = frames[idx1]
            tasks.append(get_and_store(f"step_{step}", frame2_path, frame1_path))
    
    if tasks:
        await asyncio.gather(*tasks)

async def run_pipeline():
    data_root = config.get("data_root")
    output_root = config.get("output_root")
    output_root = os.path.join(output_root, config.get("camera_name"))
    os.makedirs(output_root, exist_ok=True)

    sampling_cfg = config.get("sampling")
    step_sizes = sampling_cfg.get("step_sizes")
    use_initial = sampling_cfg.get("use_initial_frame")
    interval = sampling_cfg.get("evaluation_interval")

    # Find all tasks in the data directory
    tasks_to_process = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    async with aiohttp.ClientSession() as session:
        for task in tasks_to_process:
            task_desc = TASK_DESCRIPTIONS.get(task)
            prompt = config["reward_prompt_template"].format(task_description=task_desc)

            task_path = os.path.join(data_root, task)
            levels = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
            
            # We will separate results by comparison step (initial vs step_X)
            combined_results = {}
            steps = []
            if use_initial: 
                steps.append("initial")
            for s in step_sizes: 
                steps.append(f"step_{s}")
            
            for t in steps:
                combined_results[t] = {}

            rollout_tasks = []
            for level in levels:
                level_path = os.path.join(task_path, level)
                rollouts = [d for d in os.listdir(level_path) if os.path.isdir(os.path.join(level_path, d))]
                
                for t in steps:
                    combined_results[t][level] = {}
                
                for rollout in rollouts:
                    rollout_path = os.path.join(level_path, rollout)
                    for t in steps:
                        combined_results[t][level][rollout] = []
                    
                    rollout_tasks.append(process_rollout(
                        session, task, level, rollout, rollout_path, prompt, 
                        use_initial, step_sizes, interval, combined_results
                    ))
            
            if rollout_tasks:
                await asyncio.gather(*rollout_tasks)

            # Save results into distinct files for each task and step size
            for t in steps:
                output_file = os.path.join(output_root, f"{task}_{t}_rewards.json")
                # Sort frames in each rollout before saving to ensure consistency
                for level in combined_results[t]:
                    for rollout in combined_results[t][level]:
                        combined_results[t][level][rollout].sort(key=lambda x: x["frame"])
                
                with open(output_file, "w") as f:
                    json.dump({
                        "task": task,
                        "step": t,
                        "description": task_desc,
                        "results": combined_results[t]
                    }, f, indent=4)
                print(f"Saved results for {task} ({t}) to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
