"""
Collect online PickCube transitions that expose reward-model false positives.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from vlm_reward.rl.checkpoints import (
    load_actor_checkpoint,
    validate_actor_for_environment,
)
from vlm_reward.rl.core import PolicyNetwork, set_seed
from vlm_reward.rl.environments import get_state_dict, state_to_numpy
from vlm_reward.models.reward_predictor import QwenRewardHeadPredictor


def scalar_float(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar {name}, got tensor shape {tuple(value.shape)}")
        return float(value.item())
    return float(value)


def scalar_bool(value: Any, name: str) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar {name}, got tensor shape {tuple(value.shape)}")
        return bool(value.item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar {name}, got array shape {value.shape}")
        return bool(value.item())
    return bool(value)


def save_rgb(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="JPEG", quality=jpeg_quality)


def portable_path(path: Path) -> str:
    absolute_path = path.resolve()
    if absolute_path == PROJECT_ROOT or PROJECT_ROOT in absolute_path.parents:
        return absolute_path.relative_to(PROJECT_ROOT).as_posix()
    return str(absolute_path)


def load_actor(
    checkpoint_path: Path,
    state_dim: int,
    action_dim: int,
    max_action: float,
    hidden_dim: int,
    device: torch.device,
) -> PolicyNetwork:
    actor, metadata = load_actor_checkpoint(checkpoint_path, device)
    validate_actor_for_environment(
        metadata=metadata,
        task="PickCube-v1",
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )
    if metadata["actor_hidden_dim"] != hidden_dim:
        raise ValueError(
            f"Checkpoint actor width {metadata['actor_hidden_dim']} does not match "
            f"--actor-hidden-dim {hidden_dim}"
        )
    return actor


def flush_records(
    pending_records: List[Dict[str, Any]],
    predictor: QwenRewardHeadPredictor,
    prediction_batch_size: int,
    train_handle: Any,
    val_handle: Any,
    test_handle: Any,
) -> Dict[str, float]:
    if not pending_records:
        return {"count": 0.0, "absolute_error_sum": 0.0, "hard_negative_count": 0.0}

    images = [record["image_array"] for record in pending_records]
    predictions = predictor.predict(images, batch_size=prediction_batch_size)
    if len(predictions) != len(pending_records):
        raise ValueError(f"Expected {len(pending_records)} predictions, got {len(predictions)}")

    absolute_error_sum = 0.0
    hard_negative_count = 0
    for record, prediction in zip(pending_records, predictions):
        predicted_reward = float(prediction)
        true_reward = float(record["reward"])
        absolute_error = abs(predicted_reward - true_reward)
        hard_negative = predicted_reward >= 0.25 and true_reward <= 0.10
        absolute_error_sum += absolute_error
        hard_negative_count += int(hard_negative)

        output_record = {
            "task": record["task"],
            "episode_id": record["episode_id"],
            "step": record["step"],
            "previous_image": record["previous_image"],
            "image": record["image"],
            "reward": true_reward,
            "predicted_reward": predicted_reward,
            "absolute_error": absolute_error,
            "hard_negative": hard_negative,
            "success": record["success"],
            "terminated": record["terminated"],
            "truncated": record["truncated"],
            "source_checkpoint": record["source_checkpoint"],
            "policy_mode": record["policy_mode"],
        }
        if record["split"] == "train":
            handle = train_handle
        elif record["split"] == "val":
            handle = val_handle
        elif record["split"] == "test":
            handle = test_handle
        else:
            raise ValueError(f"Unsupported split: {record['split']}")
        handle.write(json.dumps(output_record) + "\n")

    return {
        "count": float(len(pending_records)),
        "absolute_error_sum": absolute_error_sum,
        "hard_negative_count": float(hard_negative_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickCube-v1", choices=["PickCube-v1"])
    parser.add_argument("--actor-checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--reward-head-checkpoint",
        default="finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube/best_partial_balanced_reward_head.pth",
    )
    parser.add_argument("--output-dir", default="finetune_data/reward_head_online_pickcube")
    parser.add_argument("--episodes-per-checkpoint", type=int, default=100)
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--episodes-per-prediction-batch", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_checkpoint <= 0:
        raise ValueError("--episodes-per-checkpoint must be positive")
    if args.prediction_batch_size <= 0 or args.episodes_per_prediction_batch <= 0:
        raise ValueError("Prediction batch sizes must be positive")
    if args.validation_fraction <= 0.0 or args.validation_fraction >= 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if args.test_fraction <= 0.0 or args.test_fraction >= 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16 if args.bf16 else torch.float16
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    reward_head_checkpoint = Path(args.reward_head_checkpoint).expanduser()
    if not reward_head_checkpoint.is_absolute():
        reward_head_checkpoint = PROJECT_ROOT / reward_head_checkpoint
    reward_head_checkpoint = reward_head_checkpoint.resolve()
    actor_checkpoint_paths = []
    for checkpoint_text in args.actor_checkpoints:
        checkpoint_path = Path(checkpoint_text).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        actor_checkpoint_paths.append(checkpoint_path.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = output_dir / "train.jsonl"
    val_manifest = output_dir / "val.jsonl"
    test_manifest = output_dir / "test.jsonl"
    if train_manifest.exists() or val_manifest.exists() or test_manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifests in {output_dir}. Remove or choose another output directory."
        )

    env = gym.make(
        args.task,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        reward_mode="normalized_dense",
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
    )
    initial_obs, _ = env.reset(seed=args.seed)
    state_dim = int(initial_obs.shape[-1])
    action_dim = int(env.action_space.shape[-1])
    max_action = float(env.action_space.high.reshape(-1)[0])

    predictor = QwenRewardHeadPredictor(
        checkpoint_path=str(reward_head_checkpoint),
        device=device,
        dtype=dtype,
    )
    if predictor.task != args.task:
        raise ValueError(f"Reward head task {predictor.task} does not match {args.task}")

    pending_records: List[Dict[str, Any]] = []
    total_count = 0.0
    total_absolute_error = 0.0
    total_hard_negatives = 0.0
    global_episode_id = 0

    with (
        train_manifest.open("w") as train_handle,
        val_manifest.open("w") as val_handle,
        test_manifest.open("w") as test_handle,
    ):
        for checkpoint_path in actor_checkpoint_paths:
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            actor = load_actor(
                checkpoint_path=checkpoint_path,
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                hidden_dim=args.actor_hidden_dim,
                device=device,
            )
            checkpoint_name = checkpoint_path.stem

            for local_episode_idx in range(args.episodes_per_checkpoint):
                episode_seed = args.seed + global_episode_id
                obs, _ = env.reset(seed=episode_seed)
                env.action_space.seed(episode_seed)
                policy_mode = "deterministic" if local_episode_idx % 2 == 0 else "stochastic"
                split_bucket = global_episode_id % 100
                validation_end = int(args.validation_fraction * 100)
                test_end = validation_end + int(args.test_fraction * 100)
                if split_bucket < validation_end:
                    split = "val"
                elif split_bucket < test_end:
                    split = "test"
                else:
                    split = "train"
                episode_name = f"{checkpoint_name}_{policy_mode}_{global_episode_id:06d}"
                episode_dir = output_dir / "images" / episode_name

                state_dict = get_state_dict(env, obs, args.task, use_moving_mounted_camera=False)
                previous_image = state_dict["image"]
                previous_path = episode_dir / "frame_000.jpg"
                save_rgb(previous_path, previous_image, args.jpeg_quality)

                done = False
                step = 0
                while not done:
                    state_tensor = torch.from_numpy(state_to_numpy(state_dict["state"])).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if policy_mode == "deterministic":
                            mean, _ = actor(state_tensor)
                            action_tensor = torch.tanh(mean) * actor.max_action
                        else:
                            action_tensor, _ = actor.sample(state_tensor)
                    action = action_tensor.squeeze(0).cpu().numpy()

                    next_obs, reward_value, terminated, truncated, info = env.step(action)
                    terminated_bool = scalar_bool(terminated, "terminated")
                    truncated_bool = scalar_bool(truncated, "truncated")
                    success = scalar_bool(info["success"], "success")
                    reward = scalar_float(reward_value, "reward")
                    if reward < 0.0 or reward > 1.0:
                        raise ValueError(f"Reward outside [0, 1]: {reward}")

                    next_state_dict = get_state_dict(
                        env,
                        next_obs,
                        args.task,
                        use_moving_mounted_camera=False,
                    )
                    next_image = next_state_dict["image"]
                    next_path = episode_dir / f"frame_{step + 1:03d}.jpg"
                    save_rgb(next_path, next_image, args.jpeg_quality)

                    pending_records.append(
                        {
                            "task": args.task,
                            "episode_id": episode_name,
                            "step": step,
                            "previous_image": portable_path(previous_path),
                            "image": portable_path(next_path),
                            "image_array": next_image,
                            "reward": reward,
                            "success": success,
                            "terminated": terminated_bool,
                            "truncated": truncated_bool,
                            "source_checkpoint": str(checkpoint_path),
                            "policy_mode": policy_mode,
                            "split": split,
                        }
                    )
                    previous_path = next_path
                    state_dict = next_state_dict
                    step += 1
                    done = terminated_bool or truncated_bool

                global_episode_id += 1
                if global_episode_id % args.episodes_per_prediction_batch == 0:
                    stats = flush_records(
                        pending_records,
                        predictor,
                        args.prediction_batch_size,
                        train_handle,
                        val_handle,
                        test_handle,
                    )
                    total_count += stats["count"]
                    total_absolute_error += stats["absolute_error_sum"]
                    total_hard_negatives += stats["hard_negative_count"]
                    pending_records = []
                    print(
                        f"Collected {global_episode_id} episodes | transitions={int(total_count)} | "
                        f"MAE={total_absolute_error / total_count:.4f} | "
                        f"hard_negatives={int(total_hard_negatives)}"
                    )

        stats = flush_records(
            pending_records,
            predictor,
            args.prediction_batch_size,
            train_handle,
            val_handle,
            test_handle,
        )
        total_count += stats["count"]
        total_absolute_error += stats["absolute_error_sum"]
        total_hard_negatives += stats["hard_negative_count"]

    env.close()
    if total_count == 0:
        raise RuntimeError("Collector produced no transitions")
    print(f"Saved train manifest: {train_manifest}")
    print(f"Saved validation manifest: {val_manifest}")
    print(f"Saved test manifest: {test_manifest}")
    print(
        f"Final transitions={int(total_count)} | MAE={total_absolute_error / total_count:.4f} | "
        f"hard_negatives={int(total_hard_negatives)}"
    )


if __name__ == "__main__":
    main()
