import json
import os
import shutil
from typing import Any, Dict, List, Tuple, Union

import cv2
import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np
import torch
import yaml

from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import sapien_utils
import sapien
from utilities import set_all_seeds

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_ROOT = os.path.join(PROJECT_ROOT, "h5")

TASKS = ["PickCube-v1", "PushCube-v1", "PegInsertionSide-v1"]


def render(env: gym.Env) -> np.ndarray:
    """Renders the environment and returns an RGB image."""
    image = env.render_rgb_array(camera_name="render_camera")

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # image shape: (Batch, Height, Width, Channels) or (Height, Width, Channels)
    if image.ndim == 4:
        image = image[0]

    return image.astype(np.uint8)


def save_image(path: str, image: np.ndarray) -> None:
    """Saves an image to disk."""
    ok = cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def sample_indices(n: int, num_frames: int) -> np.ndarray:
    """Returns exactly num_frames evenly spaced indices from 0 to n-1."""
    return np.round(np.linspace(0, n - 1, num_frames)).astype(int)


def get_task_camera_target(env: gym.Env, task: str) -> np.ndarray:
    """Computes the target point the camera should look at."""
    if task == "PickCube-v1":
        obj = env.cube.pose.p[0].cpu().numpy()
        goal = env.goal_site.pose.p[0].cpu().numpy()
        return 0.6 * obj + 0.4 * goal

    if task == "PushCube-v1":
        obj = env.obj.pose.p[0].cpu().numpy()
        goal = env.goal_region.pose.p[0].cpu().numpy()
        return 0.5 * obj + 0.5 * goal

    if task == "PegInsertionSide-v1":
        obj = env.peg.pose.p[0].cpu().numpy()
        goal = env.goal_pose.p[0].cpu().numpy()
        return 0.5 * obj + 0.5 * goal

    return np.array([0.0, 0.0, 0.1])


def update_wrist_follow_camera(env: gym.Env, task: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """Updates the 'render_camera' to follow the robot wrist."""
    wrist_link_name = "panda_hand"
    wrist_link = env.agent.robot.links_map[wrist_link_name]

    wrist_position = wrist_link.pose.p[0].cpu().numpy()

    if task == "PickCube-v1":
        eye = wrist_position + np.array([0.10, -0.10, 0.28])
    else:
        eye = wrist_position + np.array([0.065, -0.065, 0.25])

    target = get_task_camera_target(env, task)
    pose = sapien_utils.look_at(eye=eye, target=target)

    cam = env.scene.human_render_cameras["render_camera"].camera
    cam.set_local_pose(pose.sp)

    return wrist_link_name, eye, target


def make_env(task: str) -> gym.Env:
    """Creates a configured ManiSkill3 environment."""
    return gym.make(
        task,
        obs_mode="state",
        render_mode="rgb_array",
        reward_mode="normalized_dense",
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
    )


def load_episodes(task: str, num_rollouts: int) -> List[Dict[str, Any]]:
    """Loads a specified number of successful expert episodes from the JSON file."""
    json_path = os.path.join(DEMO_ROOT, task, "motionplanning", "trajectory.json")

    with open(json_path, "r") as f:
        data = json.load(f)
        try:
            episodes = data["episodes"]
        except KeyError as e:
            print(f"Error loading {json_path}: Missing 'episodes' key.")
            raise e

    valid_episodes = []
    for ep in episodes:
        try:
            is_success = ep["success"]
        except KeyError as e:
            print(f"Missing 'success' key in episode dict: {ep}")
            raise e
            
        if is_success:
            valid_episodes.append(ep)
            if len(valid_episodes) >= num_rollouts:
                break

    return valid_episodes


def extract_reward(env: gym.Env) -> float:
    """Extracts the dense reward from the current environment state."""
    obs = env.unwrapped.get_obs()
    info = env.unwrapped.get_info()
    
    # action shape: (Batch, ActionDim) - passed as zeros since it's an offline static frame evaluation.
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    return float(env.unwrapped.get_reward(obs=obs, action=action, info=info))


def _export_split(
    env: gym.Env, 
    task: str, 
    states: Union[h5py.Dataset, List[Union[torch.Tensor, np.ndarray, Dict]]], 
    sampled_indices: np.ndarray, 
    out_dir: str, 
    rollout_idx: int, 
    views: List[str], 
    enable_wrist_follow_camera: bool, 
    total_steps: int,
    is_list: bool = False
) -> None:
    """Renders and saves a specific subset of frames from an episode trajectory."""
    rollout_dir = os.path.join(out_dir, f"rollout_{rollout_idx}")
    os.makedirs(rollout_dir, exist_ok=True)
    
    sampled_rewards = []
    
    for frame_idx, state_idx in enumerate(sampled_indices):
        if is_list:
            state = states[state_idx]
        else:
            state = trajectory_utils.index_dict(states, state_idx)
            
        env.unwrapped.set_state_dict(state)
        
        if enable_wrist_follow_camera:
            update_wrist_follow_camera(env.unwrapped, task)
            
        image = render(env.unwrapped)
        reward = extract_reward(env)
        sampled_rewards.append(reward)
        
        for view in views:
            path = os.path.join(rollout_dir, f"{view}_frame_{frame_idx:03d}.jpg")
            save_image(path, image)
            
    with open(os.path.join(rollout_dir, "rewards.json"), "w") as f:
        json.dump(sampled_rewards, f, indent=4)
        
    metadata = {
        "total_steps": total_steps,
        "frame_steps": sampled_indices.tolist()
    }
    with open(os.path.join(rollout_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def export_task(
    task: str, 
    data_root: str, 
    views: List[str], 
    num_rollouts: int, 
    num_frames: int, 
    split_percentages: List[float], 
    enable_wrist_follow_camera: bool,
    seed: int
) -> None:
    """Exports expert, partial, and random trajectory data for a specific task."""
    h5_path = os.path.join(DEMO_ROOT, task, "motionplanning", "trajectory.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing ManiSkill demo file: {h5_path}")

    camera_type = "moving_mounted_rl" if enable_wrist_follow_camera else "static_rl"

    splits = ["expert", "partial", "random", "regressing"]
    split_dirs = {}
    for split in splits:
        d = os.path.join(data_root, camera_type, task, split)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        split_dirs[split] = d

    env = make_env(task)
    
    num_expert = int(num_rollouts * split_percentages[0])
    num_partial = int(num_rollouts * split_percentages[1])
    num_random = int(num_rollouts * split_percentages[2])
    num_regressing = num_rollouts - num_expert - num_partial - num_random
    
    episodes = load_episodes(task, num_expert + num_partial + num_regressing)

    if len(episodes) < num_expert + num_partial + num_regressing:
        print(f"Warning: Not enough expert episodes ({len(episodes)}) to cover {num_expert} expert, {num_partial} partial, and {num_regressing} regressing rollouts. Will use all available.")

    with h5py.File(h5_path, "r") as h5:
        # Generate Expert Rollouts
        for rollout_idx in range(num_expert):
            if rollout_idx >= len(episodes):
                break
                
            print(f"{task}: exporting expert rollout_{rollout_idx}")
            episode = episodes[rollout_idx]
            
            try:
                episode_id = episode['episode_id']
                reset_kwargs = episode['reset_kwargs']
            except KeyError as e:
                print(f"Missing required key in episode: {e}")
                raise e

            traj = h5[f"traj_{episode_id}"]
            states = traj["env_states"]
            
            env.reset(**reset_kwargs)
            actual_max_steps = len(traj["actions"]) + 1
            expert_indices = sample_indices(actual_max_steps, num_frames)
            _export_split(env, task, states, expert_indices, split_dirs["expert"], rollout_idx, views, enable_wrist_follow_camera, actual_max_steps - 1)

        # Generate Partial Rollouts
        start_idx = num_expert
        for i in range(num_partial):
            rollout_idx = start_idx + i
            if rollout_idx >= len(episodes):
                break
                
            print(f"{task}: exporting partial rollout_{rollout_idx}")
            episode = episodes[rollout_idx]
            
            try:
                episode_id = episode['episode_id']
                reset_kwargs = episode['reset_kwargs']
            except KeyError as e:
                print(f"Missing required key in episode: {e}")
                raise e
                
            traj = h5[f"traj_{episode_id}"]
            states = traj["env_states"]
            
            env.reset(**reset_kwargs)
            actual_max_steps = len(traj["actions"]) + 1
            # Cap the max step to 2/3 between num_frames and the end, guaranteeing a short/early failure
            max_cutoff = num_frames + max(1, int((actual_max_steps - num_frames) * 2 / 3))
            N = np.random.randint(num_frames, max_cutoff + 1)
            partial_indices = sample_indices(N, num_frames)
            _export_split(env, task, states, partial_indices, split_dirs["partial"], rollout_idx, views, enable_wrist_follow_camera, actual_max_steps - 1)

        # Generate Random Rollouts
        start_idx = num_expert + num_partial
        for i in range(num_random):
            rollout_idx = start_idx + i
            print(f"{task}: exporting random rollout_{rollout_idx}")
            
            env.reset(seed=seed + rollout_idx)
            env.action_space.seed(seed + rollout_idx)
            random_states = [env.unwrapped.get_state_dict()]
            random_action_momentum = env.action_space.sample()
            target_action = env.action_space.sample()
            step_count = 0
            while True:
                if step_count % 10 == 0:
                    target_action = env.action_space.sample()

                # Action shape: (Batch, ActionDim)
                random_action_momentum = 0.90 * random_action_momentum + 0.10 * target_action
                action = np.clip(random_action_momentum, env.action_space.low, env.action_space.high)
                _, _, term, trunc, _ = env.step(action)
                random_states.append(env.unwrapped.get_state_dict())
                step_count += 1
                if term or trunc:
                    break
            
            random_indices = sample_indices(len(random_states), num_frames)
            _export_split(env, task, random_states, random_indices, split_dirs["random"], rollout_idx, views, enable_wrist_follow_camera, len(random_states) - 1, is_list=True)

        # Generate Regressing Rollouts
        start_idx = num_expert + num_partial + num_random
        for i in range(num_regressing):
            rollout_idx = start_idx + i
            episode_idx = num_expert + num_partial + i
            if episode_idx >= len(episodes):
                break
                
            print(f"{task}: exporting regressing rollout_{rollout_idx}")
            episode = episodes[episode_idx]
            
            try:
                episode_id = episode['episode_id']
                reset_kwargs = episode['reset_kwargs']
            except KeyError as e:
                print(f"Missing required key in episode: {e}")
                raise e
                
            traj = h5[f"traj_{episode_id}"]
            states = traj["env_states"]
            
            env.reset(**reset_kwargs)
            actual_max_steps = len(traj["actions"]) + 1
            
            # T is turnaround point between num_frames and 80%
            min_T = num_frames
            max_T = max(min_T, int(0.8 * actual_max_steps))
            T = np.random.randint(min_T, max_T + 1) if min_T < max_T else min_T
            
            # Randomize the rendered frame index where the turnaround happens (between 20% and 80% of the sequence)
            min_forward = max(1, int(num_frames * 0.20))
            max_forward = max(min_forward, int(num_frames * 0.80))
            forward_count = np.random.randint(min_forward, max_forward + 1)
            backward_count = num_frames - forward_count
            
            forward_idx = np.round(np.linspace(0, T, forward_count)).astype(int)
            
            # Create a pool of all available integers up to T that aren't used by the forward pass
            available_pool = [x for x in range(T + 1) if x not in forward_idx]
            
            # As a mathematical failsafe, expand the pool to actual_max_steps if we still need more frames
            if len(available_pool) < backward_count:
                extra = [x for x in range(T + 1, actual_max_steps) if x not in forward_idx]
                available_pool.extend(extra)
                
            # Sample backward_count elements evenly from the pool, starting from highest down to lowest
            indices = np.round(np.linspace(len(available_pool) - 1, 0, backward_count)).astype(int)
            final_backward_idx = [available_pool[i] for i in indices]
            
            regressing_indices = np.concatenate([forward_idx, final_backward_idx]).astype(int)
            
            _export_split(env, task, states, regressing_indices, split_dirs["regressing"], rollout_idx, views, enable_wrist_follow_camera, actual_max_steps - 1)

    env.close()
    print(f"{task}: saved mixed rollouts to {data_root}/{camera_type}/{task}/")


def main() -> None:
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {CONFIG_PATH}")
        raise e

    try:
        seed = config["seed"]
        data_root = config["data_root"]
        views = config["views"]
        num_rollouts = config["num_rollouts"]
        num_frames = config["num_frames"]
        split_percentages = config["split_percentages"]
        enable_wrist_follow_camera = config["enable_wrist_follow_camera"]
    except KeyError as e:
        print(f"Missing required configuration key: {e}")
        raise e

    set_all_seeds(seed)

    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(CONFIG_PATH), "..", data_root)

    data_root = os.path.abspath(data_root)

    print(f"Writing RL trajectories to: {data_root}")
    camera_type = "moving_mounted_rl" if enable_wrist_follow_camera else "static_rl"
    print(f"Camera type: {camera_type}")
    print(f"Total Rollouts: {num_rollouts} (Split: Expert={split_percentages[0]*100}%, Partial={split_percentages[1]*100}%, Random={split_percentages[2]*100}%, Regressing={split_percentages[3]*100}%)")
    print(f"Frames per rollout: {num_frames}")

    for task in TASKS:
        export_task(task, data_root, views, num_rollouts, num_frames, split_percentages, enable_wrist_follow_camera, seed)


if __name__ == "__main__":
    main()
