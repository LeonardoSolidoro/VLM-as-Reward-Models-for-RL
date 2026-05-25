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

async def get_reward_score(session, prompt, image_paths):
    """
    Computes a reward score based on a sequence of images and a prompt.
    """
    
    content = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}})

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
    
    try:
        async with session.post(VLLM_API_URL, json=payload) as response:
            response.raise_for_status()
            response_data = await response.json()
            content = response_data['choices'][0]['message']['content'].strip()
            
            # Extract scores for each frame
            import re
            
            scores_dict = {}
            # Look for Frame X: ... <score>Y%</score>
            # We can just find all instances of <score>Y%</score> or similar.
            # However, since they are associated with Frame X, let's find the frame indices and scores.
            frame_blocks = re.findall(r"Frame\s+(\d+):.*?(?:<score>|Score:)\s*(\d+(?:\.\d+)?)\s*%?\s*(?:</score>)?", content, re.IGNORECASE | re.DOTALL)
            
            if frame_blocks:
                for idx_str, score_str in frame_blocks:
                    scores_dict[int(idx_str)] = float(score_str)
            else:
                # Fallback: just find all <score>...</score>
                raw_scores = re.findall(r"<score>\s*(\d+(?:\.\d+)?)\s*%?\s*</score>", content, re.IGNORECASE)
                for i, score_str in enumerate(raw_scores):
                    scores_dict[i] = float(score_str)
            
            return content, scores_dict
    except Exception as e:
        print(f"Error calling VLM API: {e}")
        return None, None

if __name__ == "__main__":
    pass