import json
import os
import shutil

import cv2
import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import yaml
from mani_skill.trajectory import utils as trajectory_utils

from utilities import set_all_seeds


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
DEMO_ROOT = os.path.expanduser("~/.maniskill/demos")
TASKS = ["PegInsertionSide-v1", "PickCube-v1", "PushCube-v1"]
NUM_ROLLOUTS = 50
NUM_FRAMES = 20


def render(env):
    image = env.render_rgb_array(camera_name = "render_camera")
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    return image.astype(np.uint8)


def save_image(path, image):
    ok = cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def sample_indices(n):
    return np.round(np.linspace(0, n - 1, NUM_FRAMES)).astype(int)


def make_env(task):
    return gym.make(
        task,
        obs_mode = "none",
        control_mode = "pd_joint_pos",
        render_mode = "rgb_array",
        reward_mode = "none",
        sim_backend = "physx_cpu",
        render_backend = "sapien_cpu",
    )


def load_episodes(task):
    json_path = os.path.join(DEMO_ROOT, task, "motionplanning", "trajectory.json")
    with open(json_path, "r") as f:
        episodes = json.load(f)["episodes"]
    return [ep for ep in episodes if ep.get("success", True)][:NUM_ROLLOUTS]


def export_task(task, data_root, views):
    h5_path = os.path.join(DEMO_ROOT, task, "motionplanning", "trajectory.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing ManiSkill demo file: {h5_path}")

    expert_dir = os.path.join(data_root, task, "expert")
    if os.path.exists(expert_dir):
        shutil.rmtree(expert_dir)
    os.makedirs(expert_dir, exist_ok=True)

    env = make_env(task)
    episodes = load_episodes(task)

    with h5py.File(h5_path, "r") as h5:
        for rollout_idx, episode in enumerate(episodes):
            print(f"{task}: exporting rollout_{rollout_idx}")
            env.reset(**episode["reset_kwargs"])

            traj = h5[f"traj_{episode['episode_id']}"]
            states = traj["env_states"]
            rewards = traj["rewards"][:]
            rollout_dir = os.path.join(expert_dir, f"rollout_{rollout_idx}")
            os.makedirs(rollout_dir, exist_ok=True)

            sampled_rewards = []
            for frame_idx, state_idx in enumerate(sample_indices(len(rewards) + 1)):
                state = trajectory_utils.index_dict(states, state_idx)
                env.unwrapped.set_state_dict(state)
                image = render(env.unwrapped)

                if state_idx == 0 or len(rewards) == 0:
                    sampled_rewards.append(0.0)
                else:
                    reward_idx = min(state_idx - 1, len(rewards) - 1)
                    sampled_rewards.append(float(rewards[reward_idx]))

                for view in views:
                    path = os.path.join(rollout_dir, f"{view}_frame_{frame_idx:03d}.jpg")
                    save_image(path, image)

            with open(os.path.join(rollout_dir, "rewards.json"), "w") as f:
                json.dump(sampled_rewards, f, indent=4)

    env.close()
    print(f"{task}: saved {len(episodes)} rollouts to {expert_dir}")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    set_all_seeds(config.get("seed", 0))
    data_root = config.get("data_root", "data")
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(CONFIG_PATH), "..", data_root)
    data_root = os.path.abspath(data_root)
    views = config.get("views", ["topview"])
    print(f"Writing data to: {data_root}")

    for task in TASKS:
        export_task(task, data_root, views)


if __name__ == "__main__":
    main()
