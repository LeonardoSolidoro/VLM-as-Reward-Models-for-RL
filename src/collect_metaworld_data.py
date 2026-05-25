import os
import random
import json
import cv2
import numpy as np
import metaworld
import yaml
from pathlib import Path
from metaworld import policies
from utilities import set_all_seeds

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Mapping tasks to their expert policies using names from config
POLICY_MAP = {}
for task_name, task_cfg in config["tasks"].items():
    policy_attr = task_cfg.get("policy")
    if policy_attr and hasattr(policies, policy_attr):
        POLICY_MAP[task_name] = getattr(policies, policy_attr)

def get_images(env, camera_names=['topview', 'corner', 'corner2']):
        images = {}
        for cam in camera_names:
            # MetaWorld V3 uses gymnasium's MujocoRenderer.
            # We need to set the camera_id before calling render().
            # Map camera names to IDs based on the model.
            cam_id = env.model.camera(cam).id
            env.mujoco_renderer.camera_id = cam_id
            
            img = env.render()
            
            # The renderer might return upside-down images for some cameras
            if cam in ['corner', 'corner2']:
                img = cv2.flip(img, 0)

            # Resize if needed (the original code had 448x448)
            if img.shape[:2] != (448, 448):
                img = cv2.resize(img, (448, 448))
                
            images[cam] = img
        return images

def collect_rollout(env_name, level, max_steps=200, camera_names=['topview', 'corner', 'corner2']):
    """Collects a single rollout from Meta-World."""
    # Use MT1 for single task environments in Meta-World
    mt1 = metaworld.MT1(env_name)
    # We instantiate the environment class directly from train_classes
    env = mt1.train_classes[env_name](render_mode='rgb_array')
    
    # Use a random task from MT1 train_tasks to randomize initial conditions (e.g. object positions)
    # The policy read the goal position from the observation, so it handles any task instance
    idx = np.random.randint(len(mt1.train_tasks))
    task = mt1.train_tasks[idx]
    env.set_task(task)
    
    obs, info = env.reset()
    
    # Initialize policy
    policy_cls = POLICY_MAP.get(env_name)
    if policy_cls is None:
        raise ValueError(f"No policy found for task {env_name}")
    policy = policy_cls()
    
    frames = []
    rewards = []
    successes = []

    # Logic for partial/regressing
    if level == "partial":
        max_steps = np.random.randint(20, 80)

    for t in range(max_steps):
        # 1. Render frame
        images = get_images(env, camera_names)
        frames.append(images)
        
        # 2. Get action
        if level == 'expert':
            action = policy.get_action(obs)
        elif level == 'partial':
            action = policy.get_action(obs)
        else: # random
            action = env.action_space.sample()
            
        # 3. Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 4. Record metadata
        rewards.append(float(reward))
        successes.append(float(info['success']))
        
        if terminated or truncated or info.get('success'):
            break
            
    env.close()

    print(f"Collected {len(frames)} frames. Collected {len(rewards)} rewards. Success: {info.get('success', False)}")
    return frames, rewards, successes

def save_rollout(frames, rewards, successes, base_path):
    """Saves frames as JPGs and metadata as JSON."""
    base_path = Path(base_path)
    os.makedirs(base_path, exist_ok=True)
    
    # Save frames
    for i, frame_dict in enumerate(frames):
        for cam, img in frame_dict.items():
            # img is already RGB from mujoco render
            cv2.imwrite(str(base_path / f"{cam}_frame_{i:03d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    with open(base_path / "rewards.json", "w") as f:
        # Convert float32 to float
        serializable_rewards = [float(r) for r in rewards]
        json.dump(serializable_rewards, f)

    # Save successes
    with open(base_path / "successes.json", "w") as f:
        json.dump(successes, f)
    
    print(f"Saved rollout to {base_path}")

def main():
    seed = config.get("seed")
    set_all_seeds(seed)
    
    tasks = list(POLICY_MAP.keys())
    levels = config.get("levels")
    rollouts_per_setting = config.get("rollouts_per_setting")
    camera_names = config.get("camera_names")
    
    data_root = config.get("data_root")
    
    for task in tasks:
        for level in levels:
            for i in range(rollouts_per_setting):
                print(f"Collecting {task} - {level} - rollout {i}")
                save_path = os.path.join(data_root, task, level, f"rollout_{i}")
                if os.path.exists(save_path):
                    continue
                
                frames, rewards, successes = collect_rollout(task, level=level, camera_names=camera_names)
                save_rollout(frames, rewards, successes, save_path)

if __name__ == "__main__":
#    frames, rewards, successes = collect_rollout("peg-insert-side-v3", level="expert", camera_names=config.get("camera_names"))
    main()
