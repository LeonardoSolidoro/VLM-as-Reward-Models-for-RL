import json
import os
import shutil

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
from scipy.spatial.transform import Rotation as R

from utilities import set_all_seeds


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_ROOT = os.path.join(PROJECT_ROOT, "h5")

TASKS = ["PickCube-v1", "PushCube-v1", "PegInsertionSide-v1"]

ENABLE_WRIST_FOLLOW_CAMERA = True


def render(env):
    image = env.render_rgb_array(camera_name="render_camera")

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if image.ndim == 4:
        image = image[0]

    return image.astype(np.uint8)


def save_image(path, image):
    ok = cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def sample_indices(n, num_frames):
    return np.round(np.linspace(0, n - 1, num_frames)).astype(int)


def camera_pose_from_spherical(target, az_deg, el_deg, radius):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    x = target[0] + radius * np.cos(el) * np.cos(az)
    y = target[1] + radius * np.cos(el) * np.sin(az)
    z = target[2] + radius * np.sin(el)

    position = np.array([x, y, z])

    forward = target - position
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)

    rotation_matrix = np.stack([forward, left, up], axis=1)

    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return sapien.Pose(p=position, q=quat_wxyz)


def update_moving_camera(env, frame_idx, num_frames, base_az, base_el, base_r):
    cam = env.scene.human_render_cameras["render_camera"].camera

    t = frame_idx / max(num_frames - 1, 1)

    azimuth = base_az + 10.0 * np.sin(np.pi * t) + np.random.uniform(-0.5, 0.5)
    elevation = base_el + 10.0 * np.sin(2.0 * np.pi * t) + np.random.uniform(-0.5, 0.5)
    radius = base_r + 0.02 * np.sin(np.pi * t)

    target = np.array([
        0.0 + 0.10 * np.sin(2.0 * np.pi * t) + np.random.uniform(-0.005, 0.005),
        0.0 + 0.10 * np.cos(2.6 * np.pi * t) + np.random.uniform(-0.005, 0.005),
        0.2 + np.random.uniform(-0.005, 0.005),
    ])

    pose = camera_pose_from_spherical(target, azimuth, elevation, radius)
    cam.set_local_pose(pose)


def get_task_camera_target(env, task):
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


def update_wrist_follow_camera(env, task):
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


def make_env(task):
    return gym.make(
        task,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
    )


def load_episodes(task, num_rollouts):
    json_path = os.path.join(DEMO_ROOT, task, "motionplanning", "trajectory.json")

    with open(json_path, "r") as f:
        episodes = json.load(f)["episodes"]

    return [ep for ep in episodes if ep.get("success", True)][:num_rollouts]


def export_task(task, data_root, views, num_rollouts, num_frames, enable_moving_camera):
    h5_path = os.path.join(DEMO_ROOT, task, "motionplanning", "trajectory.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing ManiSkill demo file: {h5_path}")

    if ENABLE_WRIST_FOLLOW_CAMERA:
        camera_type = "moving_mounted"
    else:
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

            base_az = np.random.uniform(-90.0, -150.0)
            base_el = np.random.uniform(45.0, 60.0)
            base_r = np.random.uniform(0.9, 1.1)

            traj = h5[f"traj_{episode['episode_id']}"]
            states = traj["env_states"]
            rewards = traj["rewards"][:]

            rollout_dir = os.path.join(expert_dir, f"rollout_{rollout_idx}")
            os.makedirs(rollout_dir, exist_ok=True)

            sampled_rewards = []
            printed_camera_debug = False

            for frame_idx, state_idx in enumerate(sample_indices(len(rewards) + 1, num_frames)):
                state = trajectory_utils.index_dict(states, state_idx)
                env.unwrapped.set_state_dict(state)

                if ENABLE_WRIST_FOLLOW_CAMERA:
                    wrist_link_name, camera_eye, camera_target = update_wrist_follow_camera(
                        env.unwrapped,
                        task,
                    )

                    if not printed_camera_debug:
                        print(
                            f"{task}: wrist-follow camera "
                            f"link={wrist_link_name}, "
                            f"eye={camera_eye.tolist()}, "
                            f"target={camera_target.tolist()}"
                        )
                        printed_camera_debug = True

                elif enable_moving_camera:
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
                    reward_idx = min(state_idx - 1, len(rewards) - 1)
                    sampled_rewards.append(float(rewards[reward_idx]))

                for view in views:
                    path = os.path.join(
                        rollout_dir,
                        f"{view}_frame_{frame_idx:03d}.jpg",
                    )
                    save_image(path, image)

            with open(os.path.join(rollout_dir, "rewards.json"), "w") as f:
                json.dump(sampled_rewards, f, indent=4)

    env.close()
    print(f"{task}: saved {len(episodes)} rollouts to {expert_dir}")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    set_all_seeds(config.get("seed"))

    data_root = config.get("data_root")

    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(CONFIG_PATH), "..", data_root)

    data_root = os.path.abspath(data_root)

    views = config.get("views")
    num_rollouts = config.get("num_rollouts")
    num_frames = config.get("num_frames")
    enable_moving_camera = config.get("enable_moving_camera")

    print(f"Writing data to: {data_root}")

    if ENABLE_WRIST_FOLLOW_CAMERA:
        camera_type = "moving_mounted"
    else:
        camera_type = "moving" if enable_moving_camera else "static"

    print(f"Camera type: {camera_type}")
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