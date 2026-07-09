import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import cv2
import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np
import torch
import yaml
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import sapien_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities import set_all_seeds


TASKS = ["PickCube-v1", "PushCube-v1", "PegInsertionSide-v1"]
LEVELS = ["expert", "partial", "random", "regressing"]
DEFAULT_SPLIT_PERCENTAGES = [0.20, 0.25, 0.35, 0.20]
DEFAULT_NUM_ROLLOUTS = 500
DEFAULT_NUM_FRAMES = 20
DEFAULT_VIEW = "topview"


def load_config(repo_root: Path) -> Dict[str, Any]:
    config_path = repo_root / "configs" / "configs.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_demo_root(repo_root: Path, demo_root_arg: str) -> Path:
    demo_root = Path(demo_root_arg)
    if not demo_root.is_absolute():
        demo_root = repo_root / demo_root
    if demo_root.exists():
        return demo_root

    h5_root = repo_root / "h5"
    if h5_root.exists():
        return h5_root

    demos_root = repo_root / "demos"
    if demos_root.exists():
        return demos_root

    raise FileNotFoundError(f"No demo root found. Checked {demo_root}, {h5_root}, and {demos_root}")


def sample_indices(num_states: int, num_frames: int) -> np.ndarray:
    return np.round(np.linspace(0, num_states - 1, num_frames)).astype(int)


def render(env: gym.Env) -> np.ndarray:
    image = env.render_rgb_array(camera_name="render_camera")
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    return image.astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def make_env(task: str) -> gym.Env:
    return gym.make(
        task,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        reward_mode="normalized_dense",
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
    )


def get_task_camera_target(env: gym.Env, task: str) -> np.ndarray:
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
    raise ValueError(f"Unsupported task: {task}")


def update_wrist_follow_camera(env: gym.Env, task: str) -> None:
    wrist_link = env.agent.robot.links_map["panda_hand"]
    wrist_position = wrist_link.pose.p[0].cpu().numpy()
    if task == "PickCube-v1":
        eye = wrist_position + np.array([0.10, -0.10, 0.28])
    else:
        eye = wrist_position + np.array([0.065, -0.065, 0.25])

    target = get_task_camera_target(env, task)
    pose = sapien_utils.look_at(eye=eye, target=target)
    cam = env.scene.human_render_cameras["render_camera"].camera
    cam.set_local_pose(pose.sp)


def extract_reward(env: gym.Env) -> float:
    obs = env.unwrapped.get_obs()
    info = env.unwrapped.get_info()
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    reward = env.unwrapped.get_reward(obs=obs, action=action, info=info)
    return float(reward.cpu().item() if hasattr(reward, "cpu") else reward)


def load_successful_episodes(demo_root: Path, task: str, num_rollouts: int) -> List[Dict[str, Any]]:
    json_path = demo_root / task / "motionplanning" / "trajectory.json"
    with json_path.open("r") as f:
        data = json.load(f)

    episodes = []
    for episode in data["episodes"]:
        if episode["success"]:
            episodes.append(episode)
            if len(episodes) >= num_rollouts:
                break
    return episodes


def export_states(
    env: gym.Env,
    task: str,
    states: Union[h5py.Dataset, List[Dict[str, Any]]],
    sampled_indices: Sequence[int],
    rollout_dir: Path,
    view: str,
    camera_mode: str,
    total_steps: int,
    states_are_list: bool = False,
) -> None:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rewards = []

    for frame_idx, state_idx in enumerate(sampled_indices):
        state = states[state_idx] if states_are_list else trajectory_utils.index_dict(states, int(state_idx))
        env.unwrapped.set_state_dict(state)

        if camera_mode == "moving_mounted":
            update_wrist_follow_camera(env.unwrapped, task)
        elif camera_mode != "static":
            raise ValueError(f"Unsupported camera mode: {camera_mode}")

        image = render(env.unwrapped)
        reward = extract_reward(env)
        if reward < 0.0 or reward > 1.0:
            raise ValueError(f"Normalized dense reward outside [0, 1]: {reward}")

        rewards.append(round(reward, 6))
        save_image(rollout_dir / f"{view}_frame_{frame_idx:03d}.jpg", image)

    with (rollout_dir / "rewards.json").open("w") as f:
        json.dump(rewards, f, indent=2)

    metadata = {
        "task": task,
        "camera_mode": camera_mode,
        "reward_mode": "normalized_dense",
        "total_steps": total_steps,
        "source_state_indices": [int(idx) for idx in sampled_indices],
    }
    with (rollout_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def rollout_counts(num_rollouts: int, split_percentages: Sequence[float]) -> Dict[str, int]:
    if len(split_percentages) != len(LEVELS):
        raise ValueError(f"Expected {len(LEVELS)} split percentages, got {len(split_percentages)}")
    if abs(sum(split_percentages) - 1.0) > 1e-6:
        raise ValueError(f"Split percentages must sum to 1.0, got {sum(split_percentages)}")

    expert = int(num_rollouts * split_percentages[0])
    partial = int(num_rollouts * split_percentages[1])
    random_count = int(num_rollouts * split_percentages[2])
    regressing = num_rollouts - expert - partial - random_count
    return {"expert": expert, "partial": partial, "random": random_count, "regressing": regressing}


def export_task_camera(
    task: str,
    camera_mode: str,
    data_root: Path,
    demo_root: Path,
    view: str,
    num_rollouts: int,
    num_frames: int,
    split_percentages: Sequence[float],
    seed: int,
) -> None:
    task_root = data_root / camera_mode / task
    if task_root.exists():
        shutil.rmtree(task_root)
    for level in LEVELS:
        (task_root / level).mkdir(parents=True, exist_ok=True)

    counts = rollout_counts(num_rollouts, split_percentages)
    needed_expert_episodes = counts["expert"] + counts["partial"] + counts["regressing"]
    episodes = load_successful_episodes(demo_root, task, needed_expert_episodes)
    if len(episodes) < needed_expert_episodes:
        raise ValueError(f"{task} has only {len(episodes)} successful episodes, need {needed_expert_episodes}")

    env = make_env(task)
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in task))
    h5_path = demo_root / task / "motionplanning" / "trajectory.h5"

    with h5py.File(h5_path, "r") as h5:
        for rollout_idx in range(counts["expert"]):
            episode = episodes[rollout_idx]
            traj = h5[f"traj_{episode['episode_id']}"]
            env.reset(**episode["reset_kwargs"])
            num_states = len(traj["actions"]) + 1
            sampled = sample_indices(num_states, num_frames)
            export_states(
                env=env,
                task=task,
                states=traj["env_states"],
                sampled_indices=sampled,
                rollout_dir=task_root / "expert" / f"rollout_{rollout_idx}",
                view=view,
                camera_mode=camera_mode,
                total_steps=num_states - 1,
            )

        partial_start = counts["expert"]
        for i in range(counts["partial"]):
            rollout_idx = partial_start + i
            episode = episodes[rollout_idx]
            traj = h5[f"traj_{episode['episode_id']}"]
            env.reset(**episode["reset_kwargs"])
            num_states = len(traj["actions"]) + 1
            max_cutoff = num_frames + max(1, int((num_states - num_frames) * 2 / 3))
            cutoff = int(rng.integers(num_frames, max_cutoff + 1))
            sampled = sample_indices(cutoff, num_frames)
            export_states(
                env=env,
                task=task,
                states=traj["env_states"],
                sampled_indices=sampled,
                rollout_dir=task_root / "partial" / f"rollout_{rollout_idx}",
                view=view,
                camera_mode=camera_mode,
                total_steps=num_states - 1,
            )

        random_start = counts["expert"] + counts["partial"]
        for i in range(counts["random"]):
            rollout_idx = random_start + i
            env.reset(seed=seed + rollout_idx)
            env.action_space.seed(seed + rollout_idx)
            random_states = [env.unwrapped.get_state_dict()]
            momentum_action = env.action_space.sample()
            target_action = env.action_space.sample()

            while True:
                if len(random_states) % 10 == 0:
                    target_action = env.action_space.sample()
                momentum_action = 0.90 * momentum_action + 0.10 * target_action
                action = np.clip(momentum_action, env.action_space.low, env.action_space.high)
                _, _, terminated, truncated, _ = env.step(action)
                random_states.append(env.unwrapped.get_state_dict())

                terminated_bool = bool(terminated.cpu().item()) if hasattr(terminated, "cpu") else bool(terminated)
                truncated_bool = bool(truncated.cpu().item()) if hasattr(truncated, "cpu") else bool(truncated)
                if terminated_bool or truncated_bool:
                    break

            sampled = sample_indices(len(random_states), num_frames)
            export_states(
                env=env,
                task=task,
                states=random_states,
                sampled_indices=sampled,
                rollout_dir=task_root / "random" / f"rollout_{rollout_idx}",
                view=view,
                camera_mode=camera_mode,
                total_steps=len(random_states) - 1,
                states_are_list=True,
            )

        regressing_start = counts["expert"] + counts["partial"] + counts["random"]
        regressing_episode_start = counts["expert"] + counts["partial"]
        for i in range(counts["regressing"]):
            rollout_idx = regressing_start + i
            episode = episodes[regressing_episode_start + i]
            traj = h5[f"traj_{episode['episode_id']}"]
            env.reset(**episode["reset_kwargs"])
            num_states = len(traj["actions"]) + 1

            min_turn = num_frames
            max_turn = max(min_turn, int(0.8 * num_states))
            turn = int(rng.integers(min_turn, max_turn + 1)) if min_turn < max_turn else min_turn
            forward_count = int(rng.integers(max(1, int(num_frames * 0.20)), max(1, int(num_frames * 0.80)) + 1))
            backward_count = num_frames - forward_count

            forward_indices = np.round(np.linspace(0, turn, forward_count)).astype(int)
            available = [idx for idx in range(turn + 1) if idx not in set(forward_indices.tolist())]
            if len(available) < backward_count:
                available.extend([idx for idx in range(turn + 1, num_states) if idx not in set(forward_indices.tolist())])
            backward_positions = np.round(np.linspace(len(available) - 1, 0, backward_count)).astype(int)
            backward_indices = [available[pos] for pos in backward_positions]
            sampled = np.concatenate([forward_indices, np.asarray(backward_indices, dtype=int)])

            export_states(
                env=env,
                task=task,
                states=traj["env_states"],
                sampled_indices=sampled,
                rollout_dir=task_root / "regressing" / f"rollout_{rollout_idx}",
                view=view,
                camera_mode=camera_mode,
                total_steps=num_states - 1,
            )

    env.close()
    print(f"{task} {camera_mode}: saved rollouts to {task_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/reward_contrastive")
    parser.add_argument("--demo-root", default="h5")
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--split-percentages", nargs=4, type=float, default=DEFAULT_SPLIT_PERCENTAGES)
    parser.add_argument("--camera-modes", nargs="+", choices=["static", "moving_mounted"], default=["static", "moving_mounted"])
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(repo_root)
    seed = int(config["seed"]) if args.seed is None else args.seed
    set_all_seeds(seed)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root

    demo_root = resolve_demo_root(repo_root, args.demo_root)
    print(f"Writing reward contrastive trajectories to: {data_root}")
    print(f"Demo root: {demo_root}")
    print(f"Rollout mix [expert, partial, random, regressing]: {args.split_percentages}")

    for camera_mode in args.camera_modes:
        for task in args.tasks:
            export_task_camera(
                task=task,
                camera_mode=camera_mode,
                data_root=data_root,
                demo_root=demo_root,
                view=args.view,
                num_rollouts=args.num_rollouts,
                num_frames=args.num_frames,
                split_percentages=args.split_percentages,
                seed=seed,
            )


if __name__ == "__main__":
    main()

