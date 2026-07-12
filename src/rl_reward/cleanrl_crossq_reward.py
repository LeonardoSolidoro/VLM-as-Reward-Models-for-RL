import argparse
import copy
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.finetune_reward_contrastive.qwen_reward_contrastive_dataset import (
    REWARD_PROMPT_TEMPLATE,
    TASK_REWARD_GUIDANCE,
    build_frames_list,
    build_user_content,
)
from src.rl.cleanrl_crossq import (
    PolicyNetwork,
    QNetwork,
    SimpleReplayBuffer,
    evaluate_policy,
    get_state_dict,
    set_seed,
)
from src.rl_reward.qwen_reward_head import QwenRewardHeadPredictor


CHECKPOINT_VERSION = 2


class StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf: str) -> None:
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self) -> None:
        pass

def parse_reward_percentages(text: str) -> List[float]:
    score_matches = re.findall(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</score>", text, flags=re.IGNORECASE)
    if score_matches:
        return [float(value) / 100.0 for value in score_matches]

    percent_matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
    return [float(value) / 100.0 for value in percent_matches]


def clip_reward(value: float, episode_idx: int, transition_idx: int) -> float:
    if value < 0.0 or value > 1.0:
        print(
            f"[VLM Reward] Clipping predicted reward for episode {episode_idx}, "
            f"transition {transition_idx}: {value:.4f}"
        )
    return float(np.clip(value, 0.0, 1.0))


def state_to_numpy(state: Any) -> np.ndarray:
    if hasattr(state, "cpu"):
        state = state.cpu().numpy()
    return np.array(state).flatten()


def success_to_bool(success: Any) -> bool:
    if isinstance(success, torch.Tensor):
        return bool(success.detach().any().item())
    if isinstance(success, np.ndarray):
        return bool(np.any(success))
    return bool(success)


def diagonal_gaussian_kl(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_log_std: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(current || reference) for diagonal pre-squash Gaussians.

    All inputs have shape (Batch, ActionDim); the result is a scalar tensor.
    """
    if not (
        mean.shape == log_std.shape == reference_mean.shape == reference_log_std.shape
    ):
        raise ValueError(
            "Policy KL tensor shapes must match, got "
            f"mean={tuple(mean.shape)}, log_std={tuple(log_std.shape)}, "
            f"reference_mean={tuple(reference_mean.shape)}, "
            f"reference_log_std={tuple(reference_log_std.shape)}"
        )
    variance_ratio = torch.exp(2.0 * (log_std - reference_log_std))
    squared_mean_distance = (mean - reference_mean).pow(2) * torch.exp(
        -2.0 * reference_log_std
    )
    kl_per_dimension = (
        reference_log_std
        - log_std
        + 0.5 * (variance_ratio + squared_mean_distance - 1.0)
    )
    return kl_per_dimension.sum(dim=1).mean()


def atomic_torch_save(checkpoint: Dict[str, Any], path: str) -> None:
    """Write a checkpoint completely before replacing the destination file."""
    temporary_path = f"{path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def compact_actor_checkpoint(
    actor: torch.nn.Module,
    log_alpha: torch.Tensor | None,
    args: argparse.Namespace,
    state_dim: int,
    action_dim: int,
    max_action: float,
    global_step: int,
    success_rate: float,
    avg_reward: float,
) -> Dict[str, Any]:
    checkpoint: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "actor_export",
        "task": args.task,
        "actor_state_dict": actor.state_dict(),
        "actor_hidden_dim": args.actor_hidden_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "global_step": global_step,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "seed": args.seed,
        "fixed_alpha": args.alpha,
    }
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    return checkpoint


def get_step_state_dict(env: gym.Env, obs: Any, task: str, moving_camera: bool, need_image: bool) -> Dict[str, Any]:
    if need_image:
        return get_state_dict(env, obs, task, moving_camera)
    return {"state": obs, "image": None}


def annotate_reward_episodes(
    episodes_data: List[List[Dict[str, Any]]],
    model: torch.nn.Module,
    processor: Any,
    task: str,
    task_description: str,
    device: torch.device,
    context_len: int,
    generation_batch_size: int,
    reward_scale: float,
    base_episode_idx: int,
) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
    chunk_messages = []
    chunk_metadata = []

    for local_ep_idx, episode_data in enumerate(episodes_data):
        episode_idx = base_episode_idx + local_ep_idx
        for start in range(0, len(episode_data), context_len):
            chunk = episode_data[start:start + context_len]
            images = [Image.fromarray(item["next_image"]) for item in chunk]
            prompt = REWARD_PROMPT_TEMPLATE.format(
                task=task,
                task_description=task_description,
                reward_guidance=TASK_REWARD_GUIDANCE[task],
                frames_list=build_frames_list(len(images)),
            )
            chunk_messages.append([{"role": "user", "content": build_user_content(prompt, images)}])
            chunk_metadata.append({"episode_idx": episode_idx, "start": start, "count": len(chunk)})

    predictions: Dict[Tuple[int, int], float] = {}
    for batch_start in range(0, len(chunk_messages), generation_batch_size):
        batch_messages = chunk_messages[batch_start:batch_start + generation_batch_size]
        batch_metadata = chunk_metadata[batch_start:batch_start + generation_batch_size]
        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={"padding": True, "return_tensors": "pt"},
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=400, do_sample=False)

        prompt_len = inputs["input_ids"].shape[1]
        generated_texts = processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)

        for text, metadata in zip(generated_texts, batch_metadata):
            rewards = parse_reward_percentages(text)
            expected = metadata["count"]
            if len(rewards) != expected:
                raise ValueError(
                    f"Expected {expected} reward predictions, got {len(rewards)} for "
                    f"episode {metadata['episode_idx']} chunk starting at {metadata['start']}.\n{text}"
                )

            for offset, reward in enumerate(rewards):
                transition_idx = metadata["start"] + offset
                predictions[(metadata["episode_idx"], transition_idx)] = clip_reward(
                    reward,
                    metadata["episode_idx"],
                    transition_idx,
                )

    annotated_transitions = []
    abs_errors = []
    for local_ep_idx, episode_data in enumerate(episodes_data):
        episode_idx = base_episode_idx + local_ep_idx
        for transition_idx, item in enumerate(episode_data):
            predicted_reward = predictions[(episode_idx, transition_idx)] * reward_scale
            annotated_transitions.append(
                (item["state"], item["action"], predicted_reward, item["next_state"], item["done"])
            )
            abs_errors.append(abs(predicted_reward - item["task_reward"]))

    if abs_errors:
        print(f"[VLM Reward] Batch env-reward MAE diagnostic: {np.mean(abs_errors):.4f}")

    return annotated_transitions


def annotate_reward_head_episodes(
    episodes_data: List[List[Dict[str, Any]]],
    predictor: QwenRewardHeadPredictor,
    batch_size: int,
    reward_scale: float,
    env_success_override: bool,
) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
    images = [item["next_image"] for episode in episodes_data for item in episode]
    inference_start = time.perf_counter()
    predicted_rewards = predictor.predict(images, batch_size=batch_size)
    inference_seconds = time.perf_counter() - inference_start
    images_per_second = len(images) / inference_seconds
    print(
        f"[Reward Head] Predicted {len(images)} frames in {inference_seconds:.2f}s "
        f"({images_per_second:.1f} frames/s)"
    )
    expected_predictions = sum(len(episode) for episode in episodes_data)
    if len(predicted_rewards) != expected_predictions:
        raise ValueError(
            f"Expected {expected_predictions} head predictions, got {len(predicted_rewards)}"
        )

    annotated_transitions = []
    absolute_errors = []
    signed_errors = []
    used_rewards = []
    environment_rewards = []
    prediction_idx = 0
    success_override_count = 0
    for episode in episodes_data:
        for item in episode:
            if env_success_override and item["success"]:
                predicted_reward = 1.0
                success_override_count += 1
            else:
                predicted_reward = clip_reward(
                    float(predicted_rewards[prediction_idx]),
                    episode_idx=-1,
                    transition_idx=prediction_idx,
                ) * reward_scale
            annotated_transitions.append(
                (item["state"], item["action"], predicted_reward, item["next_state"], item["done"])
            )
            absolute_errors.append(abs(predicted_reward - item["task_reward"]))
            signed_errors.append(predicted_reward - item["task_reward"])
            used_rewards.append(predicted_reward)
            environment_rewards.append(item["task_reward"])
            prediction_idx += 1

    if absolute_errors:
        print(
            "[Reward Head] Batch diagnostics | "
            f"MAE={np.mean(absolute_errors):.4f} | "
            f"signed_error={np.mean(signed_errors):+.4f} | "
            f"head_mean={np.mean(used_rewards):.4f} | "
            f"env_mean={np.mean(environment_rewards):.4f} | "
            f"head_std={np.std(used_rewards):.4f}"
        )
    if env_success_override:
        print(f"[Reward Head] Applied {success_override_count} environment-success override(s)")
    return annotated_transitions


def load_vlm(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs = {"device_map": args.device}
    if args.quant_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        if not args.quant_4bit:
            model = model.merge_and_unload()
    model.eval()
    return model, processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PushCube-v1")
    parser.add_argument("--max-steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument(
        "--random-steps",
        type=int,
        default=None,
        help="Random-action steps; defaults to --learning-starts",
    )
    parser.add_argument(
        "--deterministic-prefill-steps",
        type=int,
        default=0,
        help="Initial steps collected with the initialized actor's deterministic mean",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")

    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--target-success-rate", type=float, default=0.90)
    parser.add_argument("--minimum-steps-before-convergence", type=int, default=0)
    parser.add_argument(
        "--bootstrap-target-success-rate",
        type=float,
        default=None,
        help="Stop a native-reward bootstrap run at the first evaluation meeting this rate",
    )
    parser.add_argument("--save-dir", type=str, default="finetuning_output/cleanrl_crossq_reward/weights")

    parser.add_argument("--use-env-rewards", action="store_true")
    parser.add_argument("--moving-mounted-camera", action="store_true")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights")
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--vlm-context-len", type=int, default=20)
    parser.add_argument("--vlm-batch-size", type=int, default=1, help="Number of completed episodes to annotate together")
    parser.add_argument("--vlm-generation-batch-size", type=int, default=1, help="Number of VLM prompts to generate at once")
    parser.add_argument("--vlm-reward-scale", type=float, default=1.0)
    parser.add_argument("--4bit-quant", dest="quant_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--reward-head-checkpoint",
        type=str,
        default=None,
        help="Use the direct visual reward head instead of autoregressive VLM generation",
    )
    parser.add_argument("--reward-head-batch-size", type=int, default=128)
    parser.add_argument(
        "--env-success-override",
        action="store_true",
        help="Set reward-head rewards to exactly 1.0 when info['success'] is true",
    )
    parser.add_argument(
        "--init-actor-checkpoint",
        type=str,
        default=None,
        help="Initialize only the actor and optional alpha value from a checkpoint",
    )
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--restore-alpha-from-init",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-initial-success-rate", type=float, default=0.0)
    parser.add_argument("--initial-eval-episodes", type=int, default=20)

    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--actor-learning-rate",
        type=float,
        default=None,
        help="Actor learning rate; defaults to --learning-rate",
    )
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--bootstrap-at-done", type=str, default="always", choices=["always", "never"])
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--policy-delay", type=int, default=3)
    parser.add_argument(
        "--critic-only-warmup-updates",
        type=int,
        default=0,
        help="Number of critic updates before actor and temperature updates begin",
    )
    parser.add_argument("--batch-renorm-warmup-steps", type=int, default=None)
    parser.add_argument("--actor-max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_steps is None:
        args.random_steps = args.learning_starts
    if args.actor_learning_rate is None:
        args.actor_learning_rate = args.learning_rate
    if args.use_env_rewards and args.reward_head_checkpoint is not None:
        raise ValueError("--use-env-rewards and --reward-head-checkpoint are mutually exclusive")
    if args.env_success_override and args.reward_head_checkpoint is None:
        raise ValueError("--env-success-override requires --reward-head-checkpoint")
    if args.reward_head_batch_size <= 0:
        raise ValueError(f"--reward-head-batch-size must be positive, got {args.reward_head_batch_size}")
    if args.resume and args.init_actor_checkpoint is not None:
        raise ValueError("--resume and --init-actor-checkpoint are mutually exclusive")
    if args.reference_kl_weight < 0.0:
        raise ValueError("--reference-kl-weight must be non-negative")
    if args.reference_kl_weight > 0.0 and args.init_actor_checkpoint is None and not args.resume:
        raise ValueError("Positive --reference-kl-weight requires --init-actor-checkpoint or --resume")
    if args.restore_alpha_from_init and args.init_actor_checkpoint is not None and args.alpha is not None:
        raise ValueError("Cannot restore alpha from an initialization checkpoint with fixed --alpha")
    if args.random_steps < 0 or args.deterministic_prefill_steps < 0:
        raise ValueError("Exploration and deterministic prefill steps must be non-negative")
    if args.random_steps > args.deterministic_prefill_steps and args.deterministic_prefill_steps > 0:
        raise ValueError("--random-steps cannot exceed --deterministic-prefill-steps")
    if args.deterministic_prefill_steps > args.learning_starts:
        raise ValueError("--deterministic-prefill-steps cannot exceed --learning-starts")
    if args.deterministic_prefill_steps > 0 and args.init_actor_checkpoint is None and not args.resume:
        raise ValueError("Deterministic prefill requires --init-actor-checkpoint or --resume")
    if args.learning_starts < args.batch_size:
        raise ValueError("--learning-starts must be at least --batch-size")
    if args.batch_renorm_warmup_steps is not None and args.batch_renorm_warmup_steps <= 0:
        raise ValueError("--batch-renorm-warmup-steps must be positive")
    if args.actor_max_grad_norm <= 0.0:
        raise ValueError("--actor-max-grad-norm must be positive")
    if args.learning_rate <= 0.0 or args.actor_learning_rate <= 0.0:
        raise ValueError("Learning rates must be positive")
    if args.critic_only_warmup_updates < 0:
        raise ValueError("--critic-only-warmup-updates must be non-negative")
    for name, value in [
        ("target success rate", args.target_success_rate),
        ("minimum initial success rate", args.minimum_initial_success_rate),
    ]:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if args.bootstrap_target_success_rate is not None:
        if not args.use_env_rewards:
            raise ValueError("--bootstrap-target-success-rate requires --use-env-rewards")
        if args.bootstrap_target_success_rate < 0.0 or args.bootstrap_target_success_rate > 1.0:
            raise ValueError("--bootstrap-target-success-rate must be in [0, 1]")
    if (
        args.minimum_initial_success_rate > 0.0
        and args.init_actor_checkpoint is None
        and not args.resume
    ):
        raise ValueError(
            "--minimum-initial-success-rate requires --init-actor-checkpoint or --resume"
        )
    if args.initial_eval_episodes <= 0 or args.eval_episodes <= 0:
        raise ValueError("Evaluation episode counts must be positive")
    if args.minimum_steps_before_convergence < 0:
        raise ValueError("--minimum-steps-before-convergence must be non-negative")
    args.save_dir = os.path.join(args.save_dir, args.task)
    os.makedirs(args.save_dir, exist_ok=True)
    existing_artifacts = [
        os.path.join(args.save_dir, name)
        for name in os.listdir(args.save_dir)
    ]
    if not args.resume and existing_artifacts:
        raise FileExistsError(
            "Refusing to mix a new run with existing artifacts. Remove or rename the run "
            f"directory first: {existing_artifacts}"
        )
    latest_checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if args.resume and not os.path.exists(latest_checkpoint_path):
        raise FileNotFoundError(
            f"Cannot resume; checkpoint does not exist: {latest_checkpoint_path}"
        )
    eval_actor_dir = os.path.join(args.save_dir, "eval_actors")
    os.makedirs(eval_actor_dir, exist_ok=True)

    log_file = os.path.join(args.save_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

    print("=" * 50)
    print("REWARD-BASED CROSSQ TRAINING CONFIGURATION")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    need_images = not args.use_env_rewards

    env_kwargs = {
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pos",
        "render_mode": "rgb_array" if need_images else None,
        "sim_backend": "physx_cpu",
        "render_backend": "sapien_cpu",
        "reward_mode": "normalized_dense",
    }
    env = gym.make(args.task, **env_kwargs)
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    state_dim = obs.shape[-1]
    action_dim = env.action_space.shape[-1]
    max_action = float(env.action_space.high.flatten()[0])

    if need_images and args.reward_head_checkpoint is not None:
        reward_head_dtype = (
            torch.float32 if device.type == "cpu" else torch.bfloat16 if args.bf16 else torch.float16
        )
        reward_head_predictor = QwenRewardHeadPredictor(
            checkpoint_path=args.reward_head_checkpoint,
            device=device,
            dtype=reward_head_dtype,
        )
        if reward_head_predictor.task != args.task:
            raise ValueError(
                f"Reward head was trained for {reward_head_predictor.task}, but RL task is {args.task}"
            )
        model = None
        processor = None
        task_description = ""
        print(f"Using direct visual reward head: {args.reward_head_checkpoint}")
        print(
            "Replay reward source: frozen adapted visual encoder + attention pooler + "
            "reward head. Native dense reward is diagnostic-only."
        )
        if args.env_success_override:
            print("Sparse ManiSkill success is used only to override successful rewards to 1.0.")
    elif need_images:
        reward_head_predictor = None
        model, processor = load_vlm(args)
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/configs.yaml"))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        task_description = config["tasks"][args.task]["description"]
    else:
        reward_head_predictor = None
        model = None
        processor = None
        task_description = ""
        print("Bypassing VLM and using native normalized dense rewards.")

    warmup_steps = (
        args.batch_renorm_warmup_steps
        if args.batch_renorm_warmup_steps is not None
        else int(args.max_steps * 0.04)
    )
    actor = PolicyNetwork(state_dim, action_dim, max_action, hidden_dim=args.actor_hidden_dim).to(device)
    critic = QNetwork(
        state_dim,
        action_dim,
        hidden_dim=args.critic_hidden_dim,
        warmup_steps=warmup_steps,
    ).to(device)

    initialization_checkpoint_path = args.init_actor_checkpoint
    reference_actor: PolicyNetwork | None = None
    initial_log_alpha_value = 0.0
    if args.init_actor_checkpoint is not None:
        if not os.path.exists(args.init_actor_checkpoint):
            raise FileNotFoundError(args.init_actor_checkpoint)
        initialization_checkpoint = torch.load(
            args.init_actor_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if "task" in initialization_checkpoint and initialization_checkpoint["task"] != args.task:
            raise ValueError(
                f"Initialization checkpoint task is {initialization_checkpoint['task']}, "
                f"but the requested task is {args.task}"
            )
        if (
            "actor_hidden_dim" in initialization_checkpoint
            and initialization_checkpoint["actor_hidden_dim"] != args.actor_hidden_dim
        ):
            raise ValueError(
                "Initialization actor width does not match --actor-hidden-dim: "
                f"{initialization_checkpoint['actor_hidden_dim']} != {args.actor_hidden_dim}"
            )
        if "state_dim" in initialization_checkpoint and initialization_checkpoint["state_dim"] != state_dim:
            raise ValueError(
                f"Initialization state dimension {initialization_checkpoint['state_dim']} "
                f"does not match environment dimension {state_dim}"
            )
        if "action_dim" in initialization_checkpoint and initialization_checkpoint["action_dim"] != action_dim:
            raise ValueError(
                f"Initialization action dimension {initialization_checkpoint['action_dim']} "
                f"does not match environment dimension {action_dim}"
            )
        if (
            "max_action" in initialization_checkpoint
            and not np.isclose(initialization_checkpoint["max_action"], max_action)
        ):
            raise ValueError(
                f"Initialization max action {initialization_checkpoint['max_action']} "
                f"does not match environment max action {max_action}"
            )
        actor.load_state_dict(initialization_checkpoint["actor_state_dict"], strict=True)
        reference_actor = copy.deepcopy(actor)
        reference_actor.eval()
        for parameter in reference_actor.parameters():
            parameter.requires_grad = False
        if args.alpha is None and args.restore_alpha_from_init:
            if (
                "log_alpha" not in initialization_checkpoint
                or initialization_checkpoint["log_alpha"] is None
            ):
                raise KeyError(
                    "Initialization checkpoint has no automatic-temperature log_alpha; use "
                    "--no-restore-alpha-from-init or provide --alpha"
                )
            initial_log_alpha_value = float(
                torch.as_tensor(initialization_checkpoint["log_alpha"])
                .detach()
                .reshape(-1)[0]
                .item()
            )
        print(f"Initialized actor only from {args.init_actor_checkpoint}")
        print(
            "Frozen reference policy created from the imported actor. Critic, replay "
            "buffer, and all optimizers remain freshly initialized."
        )
        del initialization_checkpoint

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_learning_rate,
        betas=(args.adam_beta1, 0.999),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, 0.999),
    )

    if args.alpha is None:
        target_entropy = args.target_entropy if args.target_entropy is not None else -action_dim
        log_alpha = torch.tensor(
            [initial_log_alpha_value],
            dtype=torch.float32,
            requires_grad=True,
            device=device,
        )
        alpha_optimizer = torch.optim.Adam(
            [log_alpha],
            lr=args.learning_rate,
            betas=(args.adam_beta1, 0.999),
        )
    else:
        alpha = torch.tensor(args.alpha, dtype=torch.float32, device=device)
        log_alpha = None
        alpha_optimizer = None
        target_entropy = None

    buffer = SimpleReplayBuffer(capacity=args.buffer_size)
    obs, _ = env.reset()
    state_dict = get_step_state_dict(env, obs, args.task, args.moving_mounted_camera, need_images)
    episode_data = []
    unannotated_episodes = []

    global_step = 0
    total_training_updates = 0
    total_episodes_completed = 0
    best_success_rate = 0.0
    best_eval_reward = float("-inf")
    consecutive_successes = 0
    last_eval_step = 0
    eval_history: List[Dict[str, float]] = []
    latest_success_rate = 0.0
    latest_eval_reward = float("-inf")
    last_sac_actor_loss = float("nan")
    last_reference_kl = float("nan")
    last_total_actor_loss = float("nan")
    last_reference_action_mse = float("nan")
    actor_updates_started = False

    if args.resume:
        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {checkpoint['checkpoint_version']}; "
                f"expected {CHECKPOINT_VERSION}"
            )
        if checkpoint["checkpoint_type"] != "full_training_state":
            raise ValueError(
                "--resume requires a full training checkpoint, got "
                f"{checkpoint['checkpoint_type']}"
            )
        if checkpoint["task"] != args.task:
            raise ValueError(
                f"Resume checkpoint task {checkpoint['task']} does not match {args.task}"
            )
        if checkpoint["state_dim"] != state_dim or checkpoint["action_dim"] != action_dim:
            raise ValueError(
                "Resume checkpoint dimensions do not match the environment: "
                f"saved=({checkpoint['state_dim']}, {checkpoint['action_dim']}), "
                f"current=({state_dim}, {action_dim})"
            )
        if checkpoint["effective_batch_renorm_warmup_steps"] != warmup_steps:
            raise ValueError(
                "Effective BatchRenorm warmup changed on resume: "
                f"saved={checkpoint['effective_batch_renorm_warmup_steps']}, "
                f"requested={warmup_steps}. Pass an explicit, unchanged "
                "--batch-renorm-warmup-steps."
            )
        saved_args = checkpoint["args"]
        immutable_resume_arguments = [
            "use_env_rewards",
            "reward_head_checkpoint",
            "env_success_override",
            "moving_mounted_camera",
            "vlm_reward_scale",
            "vlm_batch_size",
            "vlm_generation_batch_size",
            "reward_head_batch_size",
            "model_id",
            "adapter_dir",
            "vlm_context_len",
            "bf16",
            "quant_4bit",
            "seed",
            "deterministic",
            "actor_hidden_dim",
            "critic_hidden_dim",
            "batch_renorm_warmup_steps",
            "learning_starts",
            "random_steps",
            "deterministic_prefill_steps",
            "batch_size",
            "buffer_size",
            "utd_ratio",
            "learning_rate",
            "actor_learning_rate",
            "adam_beta1",
            "gamma",
            "alpha",
            "target_entropy",
            "bootstrap_at_done",
            "policy_delay",
            "critic_only_warmup_updates",
            "reference_kl_weight",
            "actor_max_grad_norm",
        ]
        for argument_name in immutable_resume_arguments:
            if saved_args[argument_name] != vars(args)[argument_name]:
                raise ValueError(
                    f"Resume argument --{argument_name.replace('_', '-')} changed: "
                    f"saved={saved_args[argument_name]!r}, "
                    f"requested={vars(args)[argument_name]!r}"
                )
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        total_training_updates = checkpoint["total_training_updates"]
        total_episodes_completed = checkpoint["total_episodes_completed"]
        best_success_rate = checkpoint["best_success_rate"]
        best_eval_reward = checkpoint["best_eval_reward"]
        consecutive_successes = checkpoint["consecutive_successes"]
        last_eval_step = checkpoint["last_eval_step"]
        buffer.buffer.extend(checkpoint["buffer"])
        unannotated_episodes = checkpoint["unannotated_episodes"]
        eval_history = checkpoint["eval_history"]
        latest_success_rate = checkpoint["latest_success_rate"]
        latest_eval_reward = checkpoint["latest_eval_reward"]
        initialization_checkpoint_path = checkpoint["initialization_checkpoint_path"]
        last_sac_actor_loss = checkpoint["last_sac_actor_loss"]
        last_reference_kl = checkpoint["last_reference_kl"]
        last_total_actor_loss = checkpoint["last_total_actor_loss"]
        last_reference_action_mse = checkpoint["last_reference_action_mse"]
        actor_updates_started = checkpoint["actor_updates_started"]
        if checkpoint["current_episode"]:
            raise ValueError(
                "The latest checkpoint contains a partial environment episode and cannot "
                "be resumed safely"
            )
        saved_reference_state = checkpoint["reference_actor_state_dict"]
        if saved_reference_state is not None:
            reference_actor = PolicyNetwork(
                state_dim,
                action_dim,
                max_action,
                hidden_dim=args.actor_hidden_dim,
            ).to(device)
            reference_actor.load_state_dict(saved_reference_state, strict=True)
            reference_actor.eval()
            for parameter in reference_actor.parameters():
                parameter.requires_grad = False
        elif args.reference_kl_weight > 0.0:
            raise ValueError("Anchored resume checkpoint has no frozen reference actor")
        if args.alpha is None:
            log_alpha_value = checkpoint["log_alpha"].detach().reshape(-1)[0].to(device)
            with torch.no_grad():
                log_alpha.copy_(log_alpha_value.reshape_as(log_alpha))
            alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in checkpoint["cuda_rng_state"]]
            )
        print(f"Resumed from {latest_checkpoint_path} at step {global_step}")
        del checkpoint

    def build_training_checkpoint(success_rate: float, avg_reward: float) -> Dict[str, Any]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_type": "full_training_state",
            "task": args.task,
            "args": vars(args).copy(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "actor_hidden_dim": args.actor_hidden_dim,
            "critic_hidden_dim": args.critic_hidden_dim,
            "effective_batch_renorm_warmup_steps": warmup_steps,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "reference_actor_state_dict": (
                None if reference_actor is None else reference_actor.state_dict()
            ),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": (
                None if alpha_optimizer is None else alpha_optimizer.state_dict()
            ),
            "log_alpha": None if log_alpha is None else log_alpha.detach().cpu(),
            "fixed_alpha": args.alpha,
            "global_step": global_step,
            "total_training_updates": total_training_updates,
            "total_episodes_completed": total_episodes_completed,
            "latest_success_rate": success_rate,
            "latest_eval_reward": avg_reward,
            "best_success_rate": best_success_rate,
            "best_eval_reward": best_eval_reward,
            "consecutive_successes": consecutive_successes,
            "last_eval_step": last_eval_step,
            "last_sac_actor_loss": last_sac_actor_loss,
            "last_reference_kl": last_reference_kl,
            "last_total_actor_loss": last_total_actor_loss,
            "last_reference_action_mse": last_reference_action_mse,
            "actor_updates_started": actor_updates_started,
            "buffer": buffer.buffer,
            "current_episode": episode_data,
            "unannotated_episodes": unannotated_episodes,
            "eval_history": eval_history,
            "initialization_checkpoint_path": initialization_checkpoint_path,
            "reference_kl_weight": args.reference_kl_weight,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            ),
        }

    if args.init_actor_checkpoint is not None:
        initial_success_rate, initial_avg_reward = evaluate_policy(
            actor,
            args.task,
            device,
            args.initial_eval_episodes,
            base_seed=args.eval_seed,
        )
        latest_success_rate = initial_success_rate
        latest_eval_reward = initial_avg_reward
        best_success_rate = initial_success_rate
        best_eval_reward = initial_avg_reward
        eval_history.append(
            {
                "global_step": 0.0,
                "success_rate": initial_success_rate,
                "avg_reward": initial_avg_reward,
            }
        )
        print(
            f"Initial policy evaluation | success={initial_success_rate * 100:.1f}% | "
            f"avg_reward={initial_avg_reward:.2f}"
        )
        initial_checkpoint = build_training_checkpoint(
            initial_success_rate,
            initial_avg_reward,
        )
        atomic_torch_save(
            initial_checkpoint,
            os.path.join(args.save_dir, "initial_agent.pth"),
        )
        atomic_torch_save(initial_checkpoint, latest_checkpoint_path)
        atomic_torch_save(
            initial_checkpoint,
            os.path.join(args.save_dir, "best_agent.pth"),
        )
        atomic_torch_save(
            initial_checkpoint,
            os.path.join(args.save_dir, "best_return_agent.pth"),
        )
        initial_actor_export = compact_actor_checkpoint(
            actor,
            log_alpha,
            args,
            state_dim,
            action_dim,
            max_action,
            global_step=0,
            success_rate=initial_success_rate,
            avg_reward=initial_avg_reward,
        )
        atomic_torch_save(
            initial_actor_export,
            os.path.join(args.save_dir, "initial_actor_export.pth"),
        )
        atomic_torch_save(
            initial_actor_export,
            os.path.join(args.save_dir, "best_actor_export.pth"),
        )
        atomic_torch_save(
            initial_actor_export,
            os.path.join(eval_actor_dir, "actor_step_0000000_initial.pth"),
        )
        if initial_success_rate < args.minimum_initial_success_rate:
            env.close()
            raise RuntimeError(
                f"Initial policy success {initial_success_rate:.3f} is below required "
                f"{args.minimum_initial_success_rate:.3f}"
            )

    random_action_momentum = env.action_space.sample()
    target_action = env.action_space.sample()
    termination_reason = "maximum environment steps reached"

    while global_step < args.max_steps:
        if global_step < args.random_steps:
            if global_step % 10 == 0:
                target_action = env.action_space.sample()
            random_action_momentum = 0.90 * random_action_momentum + 0.10 * target_action
            action = np.clip(random_action_momentum, env.action_space.low, env.action_space.high)
        elif global_step < args.deterministic_prefill_steps:
            flat_state_ts = torch.as_tensor(
                state_to_numpy(state_dict["state"]),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                mean, _ = actor(flat_state_ts)
                action_tensor = torch.tanh(mean) * actor.max_action
            action = action_tensor.squeeze(0).cpu().numpy()
        else:
            flat_state_ts = torch.as_tensor(
                state_to_numpy(state_dict["state"]),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.sample(flat_state_ts)
            action = action.cpu().numpy().flatten()

        next_obs, env_reward, terminated, truncated, info = env.step(action)
        if hasattr(env_reward, "cpu"):
            env_reward = env_reward.cpu().item()
        terminated_bool = bool(terminated.cpu().item()) if hasattr(terminated, "cpu") else bool(terminated)
        truncated_bool = bool(truncated.cpu().item()) if hasattr(truncated, "cpu") else bool(truncated)
        done = terminated_bool or truncated_bool
        success = success_to_bool(info["success"])

        next_state_dict = get_step_state_dict(env, next_obs, args.task, args.moving_mounted_camera, need_images)
        episode_data.append(
            {
                "state": state_to_numpy(state_dict["state"]),
                "image": state_dict["image"],
                "action": action,
                "next_state": state_to_numpy(next_state_dict["state"]),
                "next_image": next_state_dict["image"],
                "done": terminated_bool,
                "task_reward": float(env_reward),
                "success": success,
            }
        )

        state_dict = next_state_dict
        global_step += 1

        if done:
            total_episodes_completed += 1
            unannotated_episodes.append(episode_data)
            episode_data = []
            obs, _ = env.reset()
            state_dict = get_step_state_dict(env, obs, args.task, args.moving_mounted_camera, need_images)

            if global_step < args.random_steps:
                random_action_momentum = env.action_space.sample()
                target_action = env.action_space.sample()

            if len(unannotated_episodes) >= args.vlm_batch_size:
                if args.use_env_rewards:
                    annotated_transitions = []
                    for episode in unannotated_episodes:
                        for item in episode:
                            annotated_transitions.append(
                                (item["state"], item["action"], item["task_reward"] * args.vlm_reward_scale, item["next_state"], item["done"])
                            )
                elif reward_head_predictor is not None:
                    print(f"Annotating {len(unannotated_episodes)} episode(s) with reward head at step {global_step}...")
                    annotated_transitions = annotate_reward_head_episodes(
                        episodes_data=unannotated_episodes,
                        predictor=reward_head_predictor,
                        batch_size=args.reward_head_batch_size,
                        reward_scale=args.vlm_reward_scale,
                        env_success_override=args.env_success_override,
                    )
                else:
                    print(f"Annotating {len(unannotated_episodes)} episode(s) at step {global_step}...")
                    annotated_transitions = annotate_reward_episodes(
                        episodes_data=unannotated_episodes,
                        model=model,
                        processor=processor,
                        task=args.task,
                        task_description=task_description,
                        device=device,
                        context_len=args.vlm_context_len,
                        generation_batch_size=args.vlm_generation_batch_size,
                        reward_scale=args.vlm_reward_scale,
                        base_episode_idx=total_episodes_completed - len(unannotated_episodes),
                    )

                for transition in annotated_transitions:
                    buffer.add(*transition)

                updates_to_do = len(annotated_transitions) * args.utd_ratio
                minimum_replay_size = max(args.batch_size, args.learning_starts)
                if len(buffer) >= minimum_replay_size:
                    for _ in range(updates_to_do):
                        b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(args.batch_size)
                        state_batch = torch.as_tensor(b_states, dtype=torch.float32, device=device)
                        action_batch = torch.as_tensor(b_actions, dtype=torch.float32, device=device)
                        reward_batch = torch.as_tensor(
                            b_rewards,
                            dtype=torch.float32,
                            device=device,
                        ).view(-1, 1)
                        next_state_batch = torch.as_tensor(
                            b_next_states,
                            dtype=torch.float32,
                            device=device,
                        )
                        done_batch = torch.as_tensor(
                            b_dones,
                            dtype=torch.float32,
                            device=device,
                        ).view(-1, 1)

                        current_alpha = log_alpha.exp() if args.alpha is None else alpha

                        with torch.no_grad():
                            next_action, next_log_prob = actor.sample(next_state_batch)

                        cat_states = torch.cat([state_batch, next_state_batch], dim=0)
                        cat_actions = torch.cat([action_batch, next_action], dim=0)
                        all_q1, all_q2 = critic(cat_states, cat_actions)
                        current_q1, next_q1 = torch.split(all_q1, args.batch_size)
                        current_q2, next_q2 = torch.split(all_q2, args.batch_size)
                        next_q_min = torch.min(next_q1, next_q2)
                        target_q = (next_q_min - current_alpha.detach() * next_log_prob).detach()
                        if args.bootstrap_at_done == "always":
                            target_q = reward_batch + args.gamma * target_q
                        else:
                            target_q = reward_batch + (1 - done_batch) * args.gamma * target_q

                        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()
                        total_training_updates += 1

                        if (
                            total_training_updates > args.critic_only_warmup_updates
                            and total_training_updates % args.policy_delay == 0
                        ):
                            for param in critic.parameters():
                                param.requires_grad = False

                            pi_action, pi_log_prob = actor.sample(state_batch)
                            critic.eval()
                            pi_q1, pi_q2 = critic(state_batch, pi_action)
                            critic.train()
                            sac_actor_loss = (
                                current_alpha.detach() * pi_log_prob - torch.min(pi_q1, pi_q2)
                            ).mean()

                            if reference_actor is not None:
                                policy_mean, policy_log_std = actor(state_batch)
                                with torch.no_grad():
                                    reference_mean, reference_log_std = reference_actor(state_batch)
                                reference_kl = torch.clamp(
                                    diagonal_gaussian_kl(
                                        policy_mean,
                                        policy_log_std,
                                        reference_mean,
                                        reference_log_std,
                                    ),
                                    min=0.0,
                                )
                                policy_mean_action = torch.tanh(policy_mean) * actor.max_action
                                reference_mean_action = (
                                    torch.tanh(reference_mean) * reference_actor.max_action
                                )
                                reference_action_mse = F.mse_loss(
                                    policy_mean_action,
                                    reference_mean_action,
                                )
                            else:
                                reference_kl = torch.zeros((), device=device)
                                reference_action_mse = torch.zeros((), device=device)

                            actor_loss = (
                                sac_actor_loss + args.reference_kl_weight * reference_kl
                            )

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                actor.parameters(),
                                args.actor_max_grad_norm,
                            )
                            actor_optimizer.step()
                            actor_updates_started = True

                            last_sac_actor_loss = float(sac_actor_loss.detach().item())
                            last_reference_kl = float(reference_kl.detach().item())
                            last_total_actor_loss = float(actor_loss.detach().item())
                            last_reference_action_mse = float(
                                reference_action_mse.detach().item()
                            )

                            for param in critic.parameters():
                                param.requires_grad = True

                            if args.alpha is None:
                                alpha_loss = -(log_alpha.exp() * (pi_log_prob + target_entropy).detach()).mean()
                                alpha_optimizer.zero_grad()
                                alpha_loss.backward()
                                alpha_optimizer.step()

                        if total_training_updates % args.learning_starts == 0:
                            alpha_value = current_alpha.item()
                            if actor_updates_started:
                                print(
                                    f"Update {total_training_updates} | Buffer {len(buffer)} | "
                                    f"Alpha {alpha_value:.5f} | "
                                    f"Avg batch reward {np.mean(b_rewards):.3f} | "
                                    f"SAC actor loss {last_sac_actor_loss:.4f} | "
                                    f"Reference KL {last_reference_kl:.6f} | "
                                    f"Total actor loss {last_total_actor_loss:.4f} | "
                                    f"Reference action MSE {last_reference_action_mse:.6f}"
                                )
                            else:
                                print(
                                    f"Update {total_training_updates} | Buffer {len(buffer)} | "
                                    f"Alpha {alpha_value:.5f} | "
                                    f"Avg batch reward {np.mean(b_rewards):.3f} | "
                                    "Critic-only warmup active"
                                )

                unannotated_episodes = []

            if global_step - last_eval_step >= args.eval_freq and global_step >= args.learning_starts:
                print(f"\n--- Evaluation at {global_step} steps ---")
                success_rate, avg_reward = evaluate_policy(
                    actor,
                    args.task,
                    device,
                    args.eval_episodes,
                    base_seed=args.eval_seed,
                )
                print(
                    f"Evaluation Success Rate: {success_rate * 100:.1f}% | "
                    f"Avg Total Reward: {avg_reward:.2f}"
                )

                success_improved = success_rate > best_success_rate
                reward_improved = avg_reward > best_eval_reward
                best_success_rate = max(best_success_rate, success_rate)
                best_eval_reward = max(best_eval_reward, avg_reward)
                latest_success_rate = success_rate
                latest_eval_reward = avg_reward
                last_eval_step = global_step
                eval_history.append(
                    {
                        "global_step": float(global_step),
                        "success_rate": success_rate,
                        "avg_reward": avg_reward,
                    }
                )

                if success_rate >= args.target_success_rate:
                    consecutive_successes += 1
                else:
                    consecutive_successes = 0

                checkpoint_dict = build_training_checkpoint(success_rate, avg_reward)
                atomic_torch_save(checkpoint_dict, latest_checkpoint_path)

                actor_export = compact_actor_checkpoint(
                    actor,
                    log_alpha,
                    args,
                    state_dim,
                    action_dim,
                    max_action,
                    global_step,
                    success_rate,
                    avg_reward,
                )
                success_per_mille = int(round(success_rate * 1000.0))
                actor_snapshot_path = os.path.join(
                    eval_actor_dir,
                    f"actor_step_{global_step:07d}_success_{success_per_mille:04d}.pth",
                )
                atomic_torch_save(actor_export, actor_snapshot_path)

                if success_improved:
                    atomic_torch_save(
                        checkpoint_dict,
                        os.path.join(args.save_dir, "best_agent.pth"),
                    )
                    atomic_torch_save(
                        actor_export,
                        os.path.join(args.save_dir, "best_actor_export.pth"),
                    )
                if reward_improved:
                    atomic_torch_save(
                        checkpoint_dict,
                        os.path.join(args.save_dir, "best_return_agent.pth"),
                    )

                if (
                    args.bootstrap_target_success_rate is not None
                    and success_rate >= args.bootstrap_target_success_rate
                ):
                    atomic_torch_save(
                        checkpoint_dict,
                        os.path.join(args.save_dir, "bootstrap_target_checkpoint.pth"),
                    )
                    atomic_torch_save(
                        actor_export,
                        os.path.join(args.save_dir, "bootstrap_actor.pth"),
                    )
                    termination_reason = (
                        "native bootstrap target reached: "
                        f"{success_rate:.3f} >= {args.bootstrap_target_success_rate:.3f}"
                    )
                    print(termination_reason)
                    break

                if (
                    global_step >= args.minimum_steps_before_convergence
                    and consecutive_successes >= 2
                ):
                    atomic_torch_save(
                        checkpoint_dict,
                        os.path.join(args.save_dir, "converged_agent.pth"),
                    )
                    termination_reason = "target success reached in two consecutive evaluations"
                    print("Convergence reached.")
                    break

    final_checkpoint = build_training_checkpoint(
        latest_success_rate,
        latest_eval_reward,
    )
    atomic_torch_save(
        final_checkpoint,
        os.path.join(args.save_dir, "final_agent.pth"),
    )
    atomic_torch_save(
        compact_actor_checkpoint(
            actor,
            log_alpha,
            args,
            state_dim,
            action_dim,
            max_action,
            global_step,
            latest_success_rate,
            latest_eval_reward,
        ),
        os.path.join(args.save_dir, "final_actor_export.pth"),
    )
    env.close()
    print(f"Training finished: {termination_reason}.")


if __name__ == "__main__":
    main()
