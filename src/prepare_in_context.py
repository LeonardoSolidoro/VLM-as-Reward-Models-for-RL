import os
import json
import shutil
import random
import numpy as np

def prepare_in_context_example():
    data_root = os.path.join("data", "metaworld")
    if not os.path.exists(data_root):
        print(f"Error: {data_root} not found.")
        return
        
    tasks = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d != "in-context-example"]
    
    for task in tasks:
        source_dir = os.path.join(data_root, task, "expert", "rollout_0")
        target_dir = os.path.join(data_root, "in-context-example", task)
        
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} not found. Skipping task {task}.")
            continue
            
        os.makedirs(target_dir, exist_ok=True)
        
        # Load rewards
        rewards_path = os.path.join(source_dir, "rewards.json")
        with open(rewards_path, "r") as f:
            rewards = json.load(f)
            
        # Map accumulated rewards to 0-100%
        cum_rewards = np.cumsum(rewards)
        max_reward = cum_rewards[-1]
        
        if max_reward > 0:
            percentages = (cum_rewards / max_reward) * 100.0
        else:
            percentages = np.zeros_like(cum_rewards)
            
        percentages = np.clip(percentages, 0, 100).astype(int)
        
        # We want to sample 30 frames.
        num_frames = len(rewards)
        # The frames are 0-indexed.
        all_indices = list(range(num_frames))
        
        if len(all_indices) > 30:
            # Keep first frame fixed, sample 29 from the rest
            sampled_indices = [0] + sorted(random.sample(all_indices[1:], 29))
        else:
            sampled_indices = all_indices
            
        # Shuffle all but the first frame
        if len(sampled_indices) > 1:
            rest = sampled_indices[1:]
            random.shuffle(rest)
            final_indices = [sampled_indices[0]] + rest
        else:
            final_indices = sampled_indices
            
        # Save the selected and shuffled data
        in_context_data = []
        
        # We will copy all views just in case, but keep track of the mapping
        views = ["topview", "corner", "corner2"]
        
        for new_idx, original_idx in enumerate(final_indices):
            perc = int(percentages[original_idx])
            
            frame_info = {
                "prompt_index": new_idx,
                "original_index": original_idx,
                "percentage": perc,
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
            
        print(f"Prepared in-context example for {task} at {target_dir} with {len(in_context_data)} frames.")

if __name__ == "__main__":
    # Fix seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    prepare_in_context_example()
