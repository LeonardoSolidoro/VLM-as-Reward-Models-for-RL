import base64
import asyncio
import json
import aiohttp
import os
import re
from dotenv import load_dotenv

""" 
Call the VLM API to predict progress scores from a prompt and trajectory images.

This file takes a text prompt with [IMG] placeholders and a list of image paths,
inserts the images into the prompt, sends the request to the VLM server, and
extracts the predicted task completion scores from the model response.
"""

load_dotenv()

VLM_API_URL = os.getenv("VLM_API_URL")
VLM_API_KEY = os.getenv("VLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def get_reward_score(session, prompt, image_paths):
    """
    Computes a reward score based on a sequence of images and a prompt.
    """
    
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
        "Authorization": f"Bearer {VLM_API_KEY}"
    }

    try:
        async with session.post(VLM_API_URL, headers = headers, json = payload) as response:
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