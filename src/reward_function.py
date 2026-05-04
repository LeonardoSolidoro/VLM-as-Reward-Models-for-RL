import base64
import json
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_reward_score(frame1_path, frame2_path, task_description):
    """
    Computes a reward score for two frames based on the task description.
    """
    
    # Minimal prompt to enforce score output
    prompt = (
        f"Task: {task_description}\n"
        "You are a reward model for RL training that evaluates the progress towards the task goal based on two images. "
        "You are given two frames: the first frame is the initial state used as reference to identify the task objective and the second frame is the current state. "
        "Output a brief explanation of the score and a single number from 0.0 to 10.0 representing the reward score for the current state. 10.0 means the task is fully completed, while 0.0 means no progress has been made. "
        "You can use decimal points for more precision."
        "For example: 'The robot has not yet picked up the cube: 1.0'"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame1_path)}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame2_path)}"}}
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    
    response = requests.post(VLLM_API_URL, json=payload)
    response_data = response.json()
    
    try:
        content = response_data['choices'][0]['message']['content'].strip()
        score = content.split(' ')[-1].strip()
        return content, float(score)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Full response data: {response_data}")
        return None, 0.0

if __name__ == "__main__":
    # Example usage:
    frame1 = "/home/leonardo/Projects/VLM-as-Reward-Models-for-RL/data/metaworld/drawer-open-v3/expert/rollout_0/frame_000.jpg"
    frame2 = "/home/leonardo/Projects/VLM-as-Reward-Models-for-RL/data/metaworld/drawer-open-v3/expert/rollout_0/frame_030.jpg"
    task = "Reach the drawer handle and open the drawer fully."
    content, score = get_reward_score(frame1, frame2, task)
    print(f"Reward Score: {score}")
    print(f"Explanation: {content}")