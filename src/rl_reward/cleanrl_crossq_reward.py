import argparse
import logging
import os
import random
import re
import sys
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
from src.rl.cleanrl_crossq import (
    PolicyNetwork,
    QNetwork,
    SimpleReplayBuffer,
    evaluate_policy,
    get_state_dict,
    set_seed,
)


IMAGE_PLACEHOLDER = "[IMG]"
REWARD_PROMPT_TEMPLATE = """You are an expert roboticist tasked to predict normalized dense environment rewards for frames of a robot performing the task: {task_description}

The normalized dense reward is between 0% and 100%, where 100% is the maximum task reward. The frames may be presented in arbitrary order, so judge each frame independently.

For each frame, format your response exactly as follows:
Frame X:
Normalized Dense Reward: <score>XX.XX%</score>

Please provide your predictions for the following frames:
{frames_list}
"""


class StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf: str) -> None:
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self) -> None:
        pass


def build_frames_list(num_frames: int) -> str:
    return "\n".join(f"Frame {idx}: {IMAGE_PLACEHOLDER}" for idx in range(1, num_frames + 1))


def build_user_content(prompt: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
    parts = prompt.split(IMAGE_PLACEHOLDER)
    if len(parts) - 1 != len(images):
        raise ValueError(f"Prompt has {len(parts) - 1} image placeholders but got {len(images)} images")

    content = []
    for text, image in zip(parts[:-1], images):
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "image", "image": image})
    if parts[-1]:
        content.append({"type": "text", "text": parts[-1]})
    return content


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


def get_step_state_dict(env: gym.Env, obs: Any, task: str, moving_camera: bool, need_image: bool) -> Dict[str, Any]:
    if need_image:
        return get_state_dict(env, obs, task, moving_camera)
    return {"state": obs, "image": None}


def annotate_reward_episodes(
    episodes_data: List[List[Dict[str, Any]]],
    model: torch.nn.Module,
    processor: Any,
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
                task_description=task_description,
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
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")

    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--target-success-rate", type=float, default=0.90)
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

    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--bootstrap-at-done", type=str, default="always", choices=["always", "never"])
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--policy-delay", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir = os.path.join(args.save_dir, args.task)
    os.makedirs(args.save_dir, exist_ok=True)

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
    obs, _ = env.reset()
    state_dim = obs.shape[-1]
    action_dim = env.action_space.shape[-1]
    max_action = float(env.action_space.high.flatten()[0])

    if need_images:
        model, processor = load_vlm(args)
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/configs.yaml"))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        task_description = config["tasks"][args.task]["description"]
    else:
        model = None
        processor = None
        task_description = ""
        print("Bypassing VLM and using native normalized dense rewards.")

    warmup_steps = int(args.max_steps * 0.04)
    actor = PolicyNetwork(state_dim, action_dim, max_action, hidden_dim=args.actor_hidden_dim).to(device)
    critic = QNetwork(state_dim, action_dim, hidden_dim=args.critic_hidden_dim, warmup_steps=warmup_steps).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, 0.999))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, 0.999))

    if args.alpha is None:
        target_entropy = args.target_entropy if args.target_entropy is not None else -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.learning_rate, betas=(args.adam_beta1, 0.999))
    else:
        alpha = torch.tensor(args.alpha, device=device)
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
    consecutive_successes = 0
    last_eval_step = 0

    latest_checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if args.resume and os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        total_training_updates = checkpoint["total_training_updates"]
        total_episodes_completed = checkpoint["total_episodes_completed"]
        best_success_rate = checkpoint["best_success_rate"]
        consecutive_successes = checkpoint["consecutive_successes"]
        last_eval_step = checkpoint["last_eval_step"]
        buffer.buffer.extend(checkpoint["buffer"])
        if args.alpha is None:
            log_alpha.data = checkpoint["log_alpha"].data
            alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        print(f"Resumed from {latest_checkpoint_path} at step {global_step}")

    random_action_momentum = env.action_space.sample()
    target_action = env.action_space.sample()

    while global_step < args.max_steps:
        if global_step < args.learning_starts:
            if global_step % 10 == 0:
                target_action = env.action_space.sample()
            random_action_momentum = 0.90 * random_action_momentum + 0.10 * target_action
            action = np.clip(random_action_momentum, env.action_space.low, env.action_space.high)
        else:
            flat_state_ts = torch.FloatTensor(state_to_numpy(state_dict["state"])).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = actor.sample(flat_state_ts)
            action = action.cpu().numpy().flatten()

        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        if hasattr(env_reward, "cpu"):
            env_reward = env_reward.cpu().item()
        terminated_bool = bool(terminated.cpu().item()) if hasattr(terminated, "cpu") else bool(terminated)
        truncated_bool = bool(truncated.cpu().item()) if hasattr(truncated, "cpu") else bool(truncated)
        done = terminated_bool or truncated_bool

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

            if global_step < args.learning_starts:
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
                else:
                    print(f"Annotating {len(unannotated_episodes)} episode(s) at step {global_step}...")
                    annotated_transitions = annotate_reward_episodes(
                        episodes_data=unannotated_episodes,
                        model=model,
                        processor=processor,
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
                if len(buffer) > args.learning_starts:
                    for _ in range(updates_to_do):
                        b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(args.batch_size)
                        state_batch = torch.FloatTensor(b_states).to(device)
                        action_batch = torch.FloatTensor(b_actions).to(device)
                        reward_batch = torch.FloatTensor(b_rewards).view(-1, 1).to(device)
                        next_state_batch = torch.FloatTensor(b_next_states).to(device)
                        done_batch = torch.FloatTensor(b_dones).view(-1, 1).to(device)

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

                        if total_training_updates % args.policy_delay == 0:
                            for param in critic.parameters():
                                param.requires_grad = False

                            pi_action, pi_log_prob = actor.sample(state_batch)
                            critic.eval()
                            pi_q1, pi_q2 = critic(state_batch, pi_action)
                            critic.train()
                            actor_loss = (current_alpha.detach() * pi_log_prob - torch.min(pi_q1, pi_q2)).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()

                            for param in critic.parameters():
                                param.requires_grad = True

                            if args.alpha is None:
                                alpha_loss = -(log_alpha.exp() * (pi_log_prob + target_entropy).detach()).mean()
                                alpha_optimizer.zero_grad()
                                alpha_loss.backward()
                                alpha_optimizer.step()

                        if total_training_updates % args.learning_starts == 0:
                            alpha_value = current_alpha.item()
                            print(
                                f"Update {total_training_updates} | Buffer {len(buffer)} | "
                                f"Alpha {alpha_value:.3f} | Avg batch reward {np.mean(b_rewards):.3f}"
                            )

                unannotated_episodes = []

            if global_step - last_eval_step >= args.eval_freq and global_step > args.learning_starts:
                print(f"\n--- Evaluation at {global_step} steps ---")
                success_rate, avg_reward = evaluate_policy(actor, args.task, device, args.eval_episodes)
                print(f"Evaluation Success Rate: {success_rate * 100:.1f}% | Avg Total Reward: {avg_reward:.2f}")

                checkpoint_dict = {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "actor_optimizer_state_dict": actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": critic_optimizer.state_dict(),
                    "global_step": global_step,
                    "total_training_updates": total_training_updates,
                    "total_episodes_completed": total_episodes_completed,
                    "avg_reward": avg_reward,
                    "success_rate": success_rate,
                    "best_success_rate": best_success_rate,
                    "consecutive_successes": consecutive_successes,
                    "last_eval_step": global_step,
                    "buffer": buffer.buffer,
                }
                if args.alpha is None:
                    checkpoint_dict["log_alpha"] = log_alpha
                    checkpoint_dict["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()

                torch.save(checkpoint_dict, os.path.join(args.save_dir, "latest_checkpoint.pth"))
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    checkpoint_dict["best_success_rate"] = best_success_rate
                    torch.save(checkpoint_dict, os.path.join(args.save_dir, "best_agent.pth"))

                if success_rate >= args.target_success_rate:
                    consecutive_successes += 1
                    if consecutive_successes >= 2:
                        torch.save(checkpoint_dict, os.path.join(args.save_dir, "converged_agent.pth"))
                        print("Convergence reached.")
                        break
                else:
                    consecutive_successes = 0

                last_eval_step = global_step

    final_checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "global_step": global_step,
    }
    if args.alpha is None:
        final_checkpoint["log_alpha"] = log_alpha
        final_checkpoint["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    torch.save(final_checkpoint, os.path.join(args.save_dir, "final_agent.pth"))
    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()

