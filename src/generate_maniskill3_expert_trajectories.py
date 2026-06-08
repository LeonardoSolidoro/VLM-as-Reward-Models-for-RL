import json
import os
import shutil
from turtle import forward

import cv2
import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np
import torch
import yaml

from mani_skill.trajectory import utils as trajectory_utils
import sapien
from scipy.spatial.transform import Rotation as R

from utilities import set_all_seeds


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
DEMO_ROOT = os.path.expanduser("~/.maniskill/demos")

TASKS = ["PegInsertionSide-v1", "PickCube-v1", "PushCube-v1"]


def render(env):
    # Remder the current environment state from render_camera view, i.e. right of workspace, slightly in front, above table looking downward
    image = env.render_rgb_array(camera_name = "render_camera")

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Remove the batch dimension if available
    if image.ndim == 4:
        image = image[0]

    return image.astype(np.uint8)


def save_image(path, image):
    ok = cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def sample_indices(n, num_frames):
    # Sample NUM_FRAMES evenly spaced state indices from the full trajectory
    return np.round(np.linspace(0, n - 1, num_frames)).astype(int)


def camera_pose_from_spherical(target, az_deg, el_deg, radius):
    """ 
    target: The point the camera is looking at 
    az_deg: Horizontal angle around target
    el_deg: Verticle angle above table
    radius: Distance from target
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    # Convert spherical coordinates to camera position (Cartesian)
    # Camera position is placed on a sphere around the target point
    x = target[0] + radius * np.cos(el) * np.cos(az)
    y = target[1] + radius * np.cos(el) * np.sin(az)
    z = target[2] + radius * np.sin(el)

    position = np.array([x, y, z])

    # Direction from camera to target
    forward = target - position
    forward /= np.linalg.norm(forward)

    # Build the camera coordinate frame
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rotation_matrix = np.stack(
        [forward, left, up],
        axis = 1,
    )

    # Quaternion in the format ManiSkill3 expects: (w, x, y, z)
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = np.array(
        [
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ]
    )

    return sapien.Pose(
        p = position,
        q = quat_wxyz,
    )


def update_moving_camera(
    env,
    frame_idx,
    num_frames,
    base_az,
    base_el,
    base_r,
):
    cam = env.scene.human_render_cameras["render_camera"].camera

    t = frame_idx / max(num_frames - 1, 1)

    # Make camera move left / right smoothly
    azimuth = (
        base_az
        + 15.0 * np.sin(2.0 * np.pi * t)
        + np.random.uniform(-2.0, 2.0)
    )

    # Make camera move up / down smoothly
    elevation = (
        base_el
        + 8.0 * np.sin(np.pi * t)
        + np.random.uniform(-1.0, 1.0)
    )

    # Make camera move closer / farther smoothly
    radius = (
        base_r
        + 0.08 * np.sin(1.5 * np.pi * t)
    )

    # Roughly the workspace center
    target = np.array([0.0, 0.0, 0.2])

    # Move the camera to the new pose
    pose = camera_pose_from_spherical(
        target,
        azimuth,
        elevation,
        radius,
    )
    cam.set_local_pose(pose)


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


def load_episodes(task, num_rollouts):
    json_path = os.path.join(
        DEMO_ROOT,
        task,
        "motionplanning",
        "trajectory.json",
    )

    with open(json_path, "r") as f:
        episodes = json.load(f)["episodes"]

    return [ep for ep in episodes if ep.get("success", True)][:num_rollouts]


def export_task(task, data_root, views, num_rollouts, num_frames, enable_moving_camera):
    h5_path = os.path.join(
        DEMO_ROOT,
        task,
        "motionplanning",
        "trajectory.h5",
    )

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Missing ManiSkill demo file: {h5_path}"
        )

    camera_type = "moving" if enable_moving_camera else "static"
    expert_dir = os.path.join(data_root, camera_type, task, "expert")

    if os.path.exists(expert_dir):
        shutil.rmtree(expert_dir)

    os.makedirs(expert_dir, exist_ok=True)

    env = make_env(task)
    episodes = load_episodes(task, num_rollouts)

    with h5py.File(h5_path, "r") as h5:
        for rollout_idx, episode in enumerate(episodes):
            print(f"{task}: exporting rollout_{rollout_idx}")

            env.reset(**episode["reset_kwargs"])

            base_az = np.random.uniform(-45.0, 45.0)
            base_el = np.random.uniform(25.0, 55.0)
            base_r = np.random.uniform(0.9, 1.3)

            traj = h5[f"traj_{episode['episode_id']}"]

            states = traj["env_states"]
            rewards = traj["rewards"][:]

            rollout_dir = os.path.join(
                expert_dir,
                f"rollout_{rollout_idx}",
            )

            os.makedirs(rollout_dir, exist_ok=True)

            sampled_rewards = []

            for frame_idx, state_idx in enumerate(
                sample_indices(len(rewards) + 1, num_frames)
            ):
                state = trajectory_utils.index_dict(
                    states,
                    state_idx,
                )

                env.unwrapped.set_state_dict(state)

                if enable_moving_camera:
                    update_moving_camera(
                        env.unwrapped,
                        frame_idx,
                        num_frames,
                        base_az,
                        base_el,
                        base_r,
                    )

                image = render(env.unwrapped)

                if state_idx == 0 or len(rewards) == 0:
                    sampled_rewards.append(0.0)
                else:
                    reward_idx = min(
                        state_idx - 1,
                        len(rewards) - 1,
                    )
                    sampled_rewards.append(
                        float(rewards[reward_idx])
                    )

                for view in views:
                    path = os.path.join(
                        rollout_dir,
                        f"{view}_frame_{frame_idx:03d}.jpg",
                    )

                    save_image(path, image)

            with open(
                os.path.join(
                    rollout_dir,
                    "rewards.json",
                ),
                "w",
            ) as f:
                json.dump(
                    sampled_rewards,
                    f,
                    indent = 4,
                )

    env.close()

    print(
        f"{task}: saved {len(episodes)} rollouts to {expert_dir}"
    )


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    set_all_seeds(config.get("seed", 0))

    data_root = config.get("data_root", "data")

    if not os.path.isabs(data_root):
        data_root = os.path.join(
            os.path.dirname(CONFIG_PATH),
            "..",
            data_root,
        )

    data_root = os.path.abspath(data_root)

    views = config.get("views", ["moving"])

    num_rollouts = config.get("num_rollouts", 50)
    num_frames = config.get("num_frames", 20)
    enable_moving_camera = config.get("enable_moving_camera", True)

    print(f"Writing data to: {data_root}")
    print(f"Camera type: {'moving' if enable_moving_camera else 'static'}")
    print(f"Rollouts: {num_rollouts}")
    print(f"Frames per rollout: {num_frames}")

    for task in TASKS:
        export_task(
            task,
            data_root,
            views,
            num_rollouts,
            num_frames,
            enable_moving_camera,
        )


if __name__ == "__main__":
    main()