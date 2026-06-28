import os
import json
import asyncio
import aiohttp
import yaml
import glob
import random
import re
from dotenv import load_dotenv

from collect_rewards import load_in_context_example, TASK_DESCRIPTIONS
from prepare_in_context import prepare_in_context_example
from vlm_api import encode_image

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

VLLM_API_URL = os.getenv("VLLM_API_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME")

async def main():
    # Hardcoded test case
    task = "PegInsertionSide-v1"
    level = "expert"
    rollout = "rollout_2"
    
    data_root = config.get("data_root")
    if config.get("enable_moving_camera"):
        data_root = os.path.join(data_root, "moving")
    rollout_path = os.path.join(data_root, task, level, rollout)
    
    if not os.path.exists(rollout_path):
        print(f"Path not found: {rollout_path}")
        print("Please make sure you are running from the project root and data_root points to the right place.")
        return

    view = config.get("views")[0]
    frames_in_context = config.get("frames_in_context")
    prompt_template = config["reward_prompt_template"]
    experiment_shuffle_frames = config.get("experiment_shuffle_frames")
    experiment_in_context_examples = config.get("experiment_in_context_examples")

    if experiment_in_context_examples > 0:
        prepare_in_context_example()

    if experiment_in_context_examples > 0:
        ic_str, ic_images = load_in_context_example(task, view)

    all_files = glob.glob(os.path.join(rollout_path, f"{view}_*.jpg"))
    frame_indices = set()
    for f in all_files:
        basename = os.path.basename(f)
        try:
            parts = basename.replace(".jpg", "").split("_frame_")
            frame_indices.add(int(parts[1]))
        except:
            continue
    
    sorted_indices = sorted(list(frame_indices))
    if not sorted_indices:
        print("No images found in rollout.")
        return

    first_frame = sorted_indices[0]
    last_frame = sorted_indices[-1]
    middle_frames = sorted_indices[1:-1]
    
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

    image_paths = list(ic_images)
    image_paths.append(os.path.join(rollout_path, f"{view}_frame_{first_frame:03d}.jpg"))
    
    frames_list_str = ""
    for i, idx in enumerate(sampled_others):
        frames_list_str += f"Frame {i+1}: [IMG]\n"
        path = os.path.join(rollout_path, f"{view}_frame_{idx:03d}.jpg")
        image_paths.append(path)
    
    prompt = prompt_template.format(
        task_description=TASK_DESCRIPTIONS.get(task, task),
        in_context_example=ic_str,
        frames_list=frames_list_str
    )
    
    text_chunks = prompt.split("[IMG]")
    content = []
    
    for i, img_path in enumerate(image_paths):
        if text_chunks[i]:
            content.append({"type": "text", "text": text_chunks[i]})
        # Omit base64 string for clean printing
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,<BASE64_STRING_OMITTED> | PATH: {os.path.dirname(img_path)}/{os.path.basename(img_path)}"}})
        
    if len(text_chunks) > len(image_paths) and text_chunks[-1]:
        content.append({"type": "text", "text": text_chunks[-1]})

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }

    print("=" * 80)
    print("DEBUG: STRUCTURED INPUT PROMPT (PAYLOAD)")
    print("=" * 80)
    print(json.dumps(payload, indent=2))
    print("=" * 80)

    print(f"\nSending request to {VLLM_API_URL} for model {MODEL_NAME}...")
    
    # Build real payload with actual base64
    real_content = []
    for i, img_path in enumerate(image_paths):
        if text_chunks[i]:
            real_content.append({"type": "text", "text": text_chunks[i]})
        real_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}})
        
    if len(text_chunks) > len(image_paths) and text_chunks[-1]:
        real_content.append({"type": "text", "text": text_chunks[-1]})

    real_payload = dict(payload)
    real_payload["messages"][0]["content"] = real_content
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VLLM_API_KEY}"
    }

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(VLLM_API_URL, headers=headers, json=real_payload) as response:
                response.raise_for_status()
                response_data = await response.json()
                response_text = response_data['choices'][0]['message']['content'].strip()
                
                print("\n" + "=" * 80)
                print("DEBUG: API RESPONSE TEXT")
                print("=" * 80)
                print(response_text)
                print("=" * 80)
                
                # Remove <think>...</think> tags if they exist to prevent regex confusion
                response_text_no_thoughts = re.sub(r"<think>.*?</think>", "", response_text, flags=re.IGNORECASE | re.DOTALL)
                
                scores_dict = {}
                frame_blocks = re.findall(r"Frame\s+(\d+):.*?(?:<score>|Score:)\s*(\d+(?:\.\d+)?)\s*%?\s*(?:</score>)?", response_text_no_thoughts, re.IGNORECASE | re.DOTALL)
                
                if frame_blocks:
                    print("\nDEBUG: Parsed using Regex A (Frame X: ... <score>Y%)")
                    for idx_str, score_str in frame_blocks:
                        scores_dict[int(idx_str)] = float(score_str)
                else:
                    raw_scores = re.findall(r"<score>\s*(\d+(?:\.\d+)?)\s*%?\s*</score>", response_text_no_thoughts, re.IGNORECASE)
                    if raw_scores:
                        print("\nDEBUG: Parsed using Fallback Regex B (<score>Y%)")
                        for i, score_str in enumerate(raw_scores):
                            scores_dict[i+1] = float(score_str)
                    else:
                        print("\nDEBUG: Failed to parse any scores from the output!")
                
                print("\nDEBUG: EXTRACTED SCORES_DICT:")
                print(json.dumps(scores_dict, indent=4))
                
        except Exception as e:
            print(f"Error calling VLM API: {e}")

if __name__ == "__main__":
    asyncio.run(main())
