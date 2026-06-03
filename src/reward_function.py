import base64
import asyncio
import json
import aiohttp
import os
import re
import ssl
from pathlib import Path
from dotenv import load_dotenv

try:
    import certifi
except ImportError:
    certifi = None

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

VLM_API_URL = os.getenv("VLM_API_URL") or os.getenv("VLLM_API_URL")
VLM_API_KEY = os.getenv("VLM_API_KEY") or os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME")
VLM_SSL_NO_VERIFY = os.getenv("VLM_SSL_NO_VERIFY", "").lower() in ("1", "true", "yes")


def build_ssl_context():
    if VLM_SSL_NO_VERIFY:
        return False
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def get_reward_score(session, prompt, image_paths):
    """
    Computes a reward score based on a sequence of images and a prompt.
    """

    if not VLM_API_URL or not MODEL_NAME:
        print("Error: set VLM_API_URL and MODEL_NAME in your .env.")
        return None, None
    
    # Split the prompt by the exact placeholder used in configs.yaml
    text_chunks = prompt.split("[IMG]")
    
    num_tags = len(text_chunks) - 1
    num_imgs = len(image_paths)
    if num_tags != num_imgs:
        print("CRITICAL ERROR: Number of text placeholders DOES NOT MATCH number of images!")
    
    content = []
    # Interleave text chunks and image objects
    for i, img_path in enumerate(image_paths):
        if text_chunks[i]:  # Add the text before the image
            content.append({"type": "text", "text": text_chunks[i]})
        
        # Add the image exactly where the placeholder was
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}})
        
    # Add any remaining text after the last image
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
    
    headers = {
        "Content-Type": "application/json",
    }
    if VLM_API_KEY and VLM_API_KEY != "EMPTY":
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    try:
        ssl_context = build_ssl_context()
        async with session.post(
            VLM_API_URL,
            headers=headers,
            json=payload,
            ssl=ssl_context,
        ) as response:
            response.raise_for_status()
            response_data = await response.json()
            content = response_data['choices'][0]['message']['content'].strip()

            #print(f"Received response from VLM API:\n{content}\n")
            
            # Remove <think>...</think> tags if they exist to prevent regex confusion
            content_no_thoughts = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
            
            # Extract scores for each frame
            scores_dict = {}
            # Look for Frame X: ... <score>Y%</score>
            # We can just find all instances of <score>Y%</score> or similar.
            # However, since they are associated with Frame X, let's find the frame indices and scores.
            frame_blocks = re.findall(r"Frame\s+(\d+):.*?(?:<score>|Score:)\s*(\d+(?:\.\d+)?)\s*%?\s*(?:</score>)?", content_no_thoughts, re.IGNORECASE | re.DOTALL)
            
            if frame_blocks:
                for idx_str, score_str in frame_blocks:
                    scores_dict[int(idx_str)] = float(score_str)
            else:
                print("Warning: No frame-specific scores found in the response. Attempting fallback parsing.")
                # Fallback: just find all <score>...</score>
                raw_scores = re.findall(r"<score>\s*(\d+(?:\.\d+)?)\s*%?\s*</score>", content_no_thoughts, re.IGNORECASE)

                if raw_scores:
                    for i, score_str in enumerate(raw_scores):
                        scores_dict[i] = float(score_str)
                else:
                    print("Warning: No scores found in the response at all.")
                    
            return content, scores_dict
    except Exception as e:
        print(f"Error calling VLM API: {e}")
        return None, None

if __name__ == "__main__":
    pass