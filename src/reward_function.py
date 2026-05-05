import base64
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

VLLM_API_URL = os.getenv("VLLM_API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def get_reward_score(session, frame1_path, frame2_path, prompt):
    """
    Computes a reward score for two frames based on the task description asynchronously.
    """
    
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
    
    try:
        async with session.post(VLLM_API_URL, json=payload) as response:
            response.raise_for_status()
            response_data = await response.json()
            content = response_data['choices'][0]['message']['content'].strip()
            # Extract score by finding the last number in the content
            import re
            scores = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            score = float(scores[-1])
        
            return content, score
    except Exception as e:
        print(f"Error calling VLM API: {e}")
        return None, None

async def main():
    # Example usage:
    frame1 = "/home/leonardo/Projects/VLM-as-Reward-Models-for-RL/data/metaworld/door-open-v3/expert/rollout_0/frame_000.jpg"
    frame2 = "/home/leonardo/Projects/VLM-as-Reward-Models-for-RL/data/metaworld/door-open-v3/expert/rollout_0/frame_020.jpg"
    task_description = "Reach the door handle and open the door completely."
    prompt = (
        f"Task: {task_description}\n"
        "You are a reward model for RL training that evaluates the progress towards the task goal based on two images. "
        "You are given two frames: the first frame is the initial state used as reference to identify the task objective and the second frame is the current state. "
        "Output a brief explanation of the score and a single number from 0.0 to 10.0 representing the reward score for the current state. 10.0 means the task is fully completed, while 0.0 means no progress has been made. "
        "You can use decimal points for more precision."
        "For example: 'The robot has not yet picked up the cube: 1.0'"
    )
    
    async with aiohttp.ClientSession() as session:
        content, score = await get_reward_score(session, frame1, frame2, prompt)
        print(f"Reward Score: {score}")
        print(f"Explanation: {content}")

if __name__ == "__main__":
    asyncio.run(main())