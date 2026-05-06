import os
import json
import cv2
import numpy as np
import gymnasium as gym
import metaworld
from metaworld.policies import (
    SawyerReachV3Policy,
    SawyerPushV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerButtonPressV3Policy,
    SawyerPegInsertionSideV3Policy,
    SawyerPickPlaceV3Policy
)

# Mapping tasks to their expert policies
POLICY_MAP = {
    'reach-v3': SawyerReachV3Policy,
    'push-v3': SawyerPushV3Policy,
    'drawer-open-v3': SawyerDrawerOpenV3Policy,
    'door-open-v3': SawyerDoorOpenV3Policy,
    'button-press-v3': SawyerButtonPressV3Policy,
    'peg-insert-side-v3': SawyerPegInsertionSideV3Policy,
    'pick-place-v3': SawyerPickPlaceV3Policy
}

def collect_rollout(
    env_name,
    level = 'expert',
    max_steps = 500,
    render_size = (512, 512),
    camera_name = 'corner'
):
    """Collects a single rollout from Meta-World."""

    # env_name = task name, e.g. "reach-v3", mt1 contains environment class in mt1.train_classes[env_name] 
    # and a list of task variants in mt1.train_tasks. Different task variants mean different sampled instances of the same task, 
    # e.g. different object or robot starting positions
    mt1 = metaworld.MT1(env_name) 

    # Instantiate the environment class 
    env = mt1.train_classes[env_name](render_mode = 'rgb_array')
    
    # Pick a random task instance to randomize 
    idx = np.random.randint(len(mt1.train_tasks))
    task = mt1.train_tasks[idx]

    # Load the sampled task configuration into the environment, i.e. tell the environment which task instance to use for this rollout, 
    # what the inital conditions should be, and what goal/task parameters should be active for this rollout
    env.set_task(task)
    
    obs, info = env.reset()
    
    # Select the camera view, e.g. "corner", "overhead", "front", ... 
    for i in range(env.model.ncam):
        if env.model.camera(i).name == camera_name:
            env.mujoco_renderer.camera_id = i
            break
    
    # Initialize the expert policy for this task
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
        # 1) Render frame from the current simulator state
        # Meta-World returns (H, W, 3) RGB arrays when render_mode is 'rgb_array'
        img = env.render()

        # Mujoco rendering can be upside down depending on setup.
        img = np.flipud(img)

        # Resize to a consistent dataset resolution.
        if img.shape[:2] != render_size:
            img = cv2.resize(img, render_size)

        frames.append(img)
        
        # 2) Choose an action based on the requested rollout level
        if level == 'expert':
            action = policy.get_action(obs)
            action = np.clip(action, -1, 1)
        elif level == 'near-expert':
            # Add Gaussian noise to expert actions for "near-expert" behavior
            action = policy.get_action(obs) + np.random.normal(0, 0.2, size = 4)
            action = np.clip(action, -1, 1)
        else:  # random
            action = env.action_space.sample()
            
        # 3) Step the environment forward with the chosen action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 4) Record rewards, success flags, and actions for later analysis
        metadata["rewards"].append(float(reward))
        metadata["success"].append(float(info['success']))
        metadata["actions"].append(action.tolist())
        
        if terminated or truncated:
            break
            
    env.close()
    return frames, metadata

def save_rollout(frames, metadata, base_path):
    """Saves frames as JPGs and metadata as JSON."""
    os.makedirs(base_path, exist_ok = True)
    
    # Save each frame as a JPG on disk
    for i, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(base_path, f"frame_{i:03d}.jpg"), frame_bgr)
        
    # Save rollout metadata alongside the frames
    with open(os.path.join(base_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent = 4)

def main():
    # Dataset configuration
    tasks = list(POLICY_MAP.keys())
    levels = ['expert', 'near-expert', 'random']
    rollouts_per_setting = 5  # Adjust as needed for full dataset
    
    data_root = "data/metaworld"
    
    # Iterate over tasks and behavior levels, skipping rollouts that already exist
    for task in tasks:
        for level in levels:
            print(f"Collecting {task} - {level}...")
            for i in range(rollouts_per_setting):
                save_path = os.path.join(data_root, task, level, f"rollout_{i}")
                if os.path.exists(save_path):
                    continue
                
                try:
                    frames, metadata = collect_rollout(task, level = level)
                    save_rollout(frames, metadata, save_path)
                    print(f"  Saved rollout {i}")
                except Exception as e:
                    print(f"  Error collecting rollout {i}: {e}")

if __name__ == "__main__":
    main()
