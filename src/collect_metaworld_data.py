import os
import json
import cv2
import numpy as np
import metaworld
import yaml
from metaworld import policies

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

def collect_rollout(env_name, level, max_steps=200, render_size=(448, 448), camera_name='corner'):
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
    
    # Set camera angle after reset to ensure renderer is initialized
    # Gymnasium's MujocoRenderer uses camera_id to determine the view
    for i in range(env.model.ncam):
        if env.model.camera(i).name == camera_name:
            env.mujoco_renderer.camera_id = i
            break
    
    # Initialize policy
    policy_cls = POLICY_MAP.get(env_name)
    if policy_cls is None:
        raise ValueError(f"No policy found for task {env_name}")
    policy = policy_cls()
    
    frames = []
    metadata = {
        "rewards": [],
        "success": [],
        "actions": []
    }
    
    for t in range(max_steps):
        # 1. Render frame (topview is usually the default or camera 0)
        # metaworld render returns a (H, W, 3) rgb array
        # In newer Gymnasium-based MetaWorld, we can use render() directly if render_mode='rgb_array'
        img = env.render()
        # Mujoco rendering is sometimes vertically flipped (upside down)
        img = np.flipud(img)
        # Resize if needed
        if img.shape[:2] != render_size:
            img = cv2.resize(img, render_size)
        frames.append(img)
        
        # 2. Get action
        if level == 'expert':
            action = policy.get_action(obs)
        elif level == 'near-expert':
            # Add Gaussian noise to expert action to make it "near-expert"
            action = policy.get_action(obs) + np.random.normal(0, 1, size=4)
            action = np.clip(action, -1, 1)
        else: # random
            action = env.action_space.sample()
            
        # 3. Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 4. Record metadata
        metadata["rewards"].append(float(reward))
        metadata["success"].append(float(info['success']))
        metadata["actions"].append(action.tolist())
        
        if terminated or truncated:
            break
            
    env.close()
    return frames, metadata

def save_rollout(frames, metadata, base_path):
    """Saves frames as JPGs and metadata as JSON."""
    os.makedirs(base_path, exist_ok=True)
    
    # Save frames
    for i, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(base_path, f"frame_{i:03d}.jpg"), frame_bgr)
        
    # Save metadata
    with open(os.path.join(base_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def main():
    tasks = list(POLICY_MAP.keys())
    levels = config.get("levels")
    rollouts_per_setting = config.get("rollouts_per_setting")
    camera_name = config.get("camera_name")
    render_size = config.get("render_size")
    max_steps = config.get("max_steps")
    
    data_root = config.get("data_root")
    
    for task in tasks:
        for level in levels:
            print(f"Collecting {task} - {level}...")
            for i in range(rollouts_per_setting):
                save_path = os.path.join(data_root, task, level, f"rollout_{i}")
                if os.path.exists(save_path):
                    continue
                
                try:
                    frames, metadata = collect_rollout(task, level=level, camera_name=camera_name, render_size=render_size, max_steps=max_steps)
                    save_rollout(frames, metadata, save_path)
                    print(f"  Saved rollout {i}")
                except Exception as e:
                    print(f"  Error collecting rollout {i}: {e}")

if __name__ == "__main__":
    main()
