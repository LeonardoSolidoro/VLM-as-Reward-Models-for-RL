import os
import sys
import time
import argparse
import random
import re
from collections import deque
import yaml
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import mani_skill.envs  # Registers ManiSkill3 environments

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# Ensure src is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.qwen_progress_dataset import build_frames_list, build_user_content

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate_policy(actor: nn.Module, task: str, device: torch.device, num_episodes: int = 10) -> Tuple[float, float]:
    eval_env = gym.make(
        task,
        obs_mode="state", 
        control_mode="pd_ee_delta_pos",
        render_mode=None, 
        sim_backend="physx_cpu",
        render_backend="sapien_cpu", 
    )
    successes = 0
    total_rewards = 0.0
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
    
        while not done:
            # We extract the flat state vector
            if isinstance(obs, dict):
                flat_state = obs["state"]
            else:
                flat_state = obs
            if hasattr(flat_state, "cpu"): flat_state = flat_state.cpu().numpy()
            flat_state_ts = torch.FloatTensor(np.array(flat_state).flatten()).unsqueeze(0).to(device)
        
            with torch.no_grad():
                # For evaluation, we use the deterministic mean
                mean, _ = actor(flat_state_ts)
                action = torch.tanh(mean).cpu().data.numpy().flatten() * actor.max_action
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if hasattr(reward, "cpu"): episode_reward += reward.cpu().item()
            else: episode_reward += float(reward)
        
            done = terminated or truncated
            step_count += 1
        
            is_success = info["success"]
            if hasattr(is_success, "any"): is_success = bool(is_success.any())
            
            if is_success:
                successes += 1
                break
            
        total_rewards += episode_reward
    eval_env.close()
    return successes / num_episodes, total_rewards / num_episodes

# ==============================================================================
# ALGORITHM MODULES (CrossQ components)
# ==============================================================================

class BatchRenorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.01, warmup_steps: int = 10000):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Features)
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
        
            # Snap to first batch stats to prevent huge eval() lag at step 0
            if self.num_batches_tracked.item() == 0:
                self.running_mean.copy_(batch_mean.detach())
                self.running_var.copy_(batch_var.detach())
        
            # CrossQ BRN Warmup
            step = self.num_batches_tracked.item()
            r_max = 1.0 if step < self.warmup_steps else 3.0
            d_max = 0.0 if step < self.warmup_steps else 5.0
        
            r = torch.clamp(torch.sqrt(batch_var.detach() + self.eps) / torch.sqrt(self.running_var + self.eps), 1.0 / r_max, r_max)
            d = torch.clamp((batch_mean.detach() - self.running_mean) / torch.sqrt(self.running_var + self.eps), -d_max, d_max)
        
            x_norm = ((x - batch_mean) / torch.sqrt(batch_var + self.eps)) * r + d
        
            self.running_mean.add_(self.momentum * (batch_mean.detach() - self.running_mean))
            self.running_var.add_(self.momentum * (batch_var.detach() - self.running_var))
            self.num_batches_tracked.add_(1)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return self.weight * x_norm + self.bias

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state shape: (Batch, state_dim)
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.tanh(log_std)
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state shape: (Batch, state_dim)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
    
        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 2048, warmup_steps: int = 10000):
        super().__init__()
        # CrossQ: 2048 width, BatchRenorm in the critic
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            BatchRenorm1d(hidden_dim, warmup_steps=warmup_steps),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            BatchRenorm1d(hidden_dim, warmup_steps=warmup_steps),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            BatchRenorm1d(hidden_dim, warmup_steps=warmup_steps),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            BatchRenorm1d(hidden_dim, warmup_steps=warmup_steps),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state shape: (Batch, state_dim)
        # action shape: (Batch, action_dim)
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

# ==============================================================================
# REPLAY BUFFER & VLM ANNOTATION LOGIC
# ==============================================================================

class SimpleReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self) -> int:
        return len(self.buffer)

def parse_percentages(text: str) -> List[float]:
    values = []
    for match in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        value = float(match) / 100.0
        values.append(value)
    return values

def check_vlm_diagnostics(percentages: List[float], shuffled_indices: List[int], ep_idx: Union[int, str] = ""):
    # Align percentages with their actual episode steps and sort chronologically
    known_pairs = sorted(zip(shuffled_indices, percentages), key=lambda x: x[0])
    chrono_steps = [x[0] for x in known_pairs]
    chrono_pcts = [x[1] * 100 for x in known_pairs]

    if len(chrono_pcts) < 2:
        return
    
    prefix = f"[VLM Diagnostic - Ep {ep_idx}]"
    
    for i in range(1, len(chrono_pcts)):
        prev = chrono_pcts[i-1]
        curr = chrono_pcts[i]
        prev_step = chrono_steps[i-1]
        curr_step = chrono_steps[i]
    
        if prev > 20 and curr < 5:
            print(f"{prefix} WARNING: Reward plummeted from {prev:.1f}% (step {prev_step}) to {curr:.1f}% (step {curr_step}). Possible occlusion!")
            continue 
        
        if abs(curr - prev) > 20:
            print(f"{prefix} WARNING: Massive reward jump ({abs(curr-prev):.1f}%) from {prev:.1f}% (step {prev_step}) to {curr:.1f}% (step {curr_step}). VLM is possibly hallucinating!")
        
    consecutive_identical = 1
    for i in range(1, len(chrono_pcts)):
        if chrono_pcts[i] == chrono_pcts[i-1] and chrono_pcts[i] > 0:
            consecutive_identical += 1
            if consecutive_identical == 3:
                print(f"{prefix} WARNING: VLM returned the EXACT same score ({chrono_pcts[i]:.1f}%) for 3+ consecutive frames (ending at step {chrono_steps[i]}). It might be failing to detect small robotic arm movements.")
                break 
        else:
            consecutive_identical = 1

def annotate_batch(episodes_data: List[List[Dict[str, Any]]], model: nn.Module, processor: Any, task_description: str, prompt_template: str, device: torch.device, context_len: int = 20, base_episode_idx: int = 0, use_difference_rewards: bool = False) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
    """
    Synchronously annotates a batch of episodes using the VLM, computes PBRS rewards, 
    and returns a flattened list of (state, action, reward, next_state, done) transitions.
    """
    batch_messages = []
    batch_metadata = []

    for episode_data in episodes_data:
        n_states = len(episode_data) + 1
        s0_image = episode_data[0]["image"]
    
        if n_states <= context_len:
            remaining_indices = list(range(1, n_states))
        else:
            remaining_indices = np.round(np.linspace(1, n_states - 1, context_len - 1)).astype(int).tolist()
        
        remaining_images = []
        for idx in remaining_indices:
            if idx < len(episode_data):
                remaining_images.append(episode_data[idx]["image"])
            else:
                remaining_images.append(episode_data[-1]["image"])

        shuffled_indices = remaining_indices
        shuffled_images = remaining_images

        pil_s0 = Image.fromarray(s0_image)
        pil_query_images = [Image.fromarray(img) for img in shuffled_images]
        all_pil_images = [pil_s0] + pil_query_images
    
        frames_list = build_frames_list(len(pil_query_images))
        user_prompt = prompt_template.format(task_description=task_description, frames_list=frames_list)
        content = build_user_content(user_prompt, all_pil_images)
        messages = [{"role": "user", "content": content}]
        batch_messages.append(messages)
        batch_metadata.append({
            "episode_data": episode_data,
            "shuffled_indices": shuffled_indices,
            "n_states": n_states
        })

    inputs = processor.apply_chat_template(
        batch_messages, 
        tokenize=True, 
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={"padding": True, "return_tensors": "pt"}
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, prompt_len:]
    generated_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    all_annotated_transitions = []

    for batch_idx, (text, meta) in enumerate(zip(generated_texts, batch_metadata)):
        percentages = parse_percentages(text)
        shuffled_indices = meta["shuffled_indices"]
        episode_data = meta["episode_data"]
        n_states = meta["n_states"]
    
        if len(percentages) != len(shuffled_indices):
            print(f"Warning: Expected {len(shuffled_indices)} percentages, got {len(percentages)}.")
            while len(percentages) < len(shuffled_indices):
                percentages.append(0.0)
            percentages = percentages[:len(shuffled_indices)]

        current_ep_idx = base_episode_idx + batch_idx
        check_vlm_diagnostics(
            percentages, 
            shuffled_indices=shuffled_indices, 
            ep_idx=current_ep_idx
        )

        progress = [None] * n_states
        for idx, prog in zip(shuffled_indices, percentages):
            progress[idx] = prog
        
        known_indices = [i for i, p in enumerate(progress) if p is not None]
        if 0 not in known_indices:
            progress[0] = 0.0
            known_indices.insert(0, 0)
        if (n_states - 1) not in known_indices:
            progress[-1] = progress[known_indices[-1]]
            known_indices.append(n_states - 1)
        
        for i in range(len(known_indices) - 1):
            start_idx = known_indices[i]
            end_idx = known_indices[i+1]
            start_val = progress[start_idx]
            end_val = progress[end_idx]
            for j in range(start_idx + 1, end_idx):
                weight = (j - start_idx) / (end_idx - start_idx)
                progress[j] = start_val + weight * (end_val - start_val)

        for t_idx in range(len(episode_data)):
            item = episode_data[t_idx]
            
            # Difference Reward
            if use_difference_rewards:
                r_t = (progress[t_idx + 1] - progress[t_idx])
            else:
                r_t = progress[t_idx]
            
            all_annotated_transitions.append((
                item["state"], item["action"], r_t, item["next_state"], item["done"]
            ))

    return all_annotated_transitions


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def get_task_camera_target(env, task):
    import numpy as np
    if task == "PickCube-v1":
        obj = env.cube.pose.p[0].cpu().numpy()
        goal = env.goal_site.pose.p[0].cpu().numpy()
        return 0.6 * obj + 0.4 * goal
    elif task == "PushCube-v1":
        obj = env.obj.pose.p[0].cpu().numpy()
        goal = env.goal_region.pose.p[0].cpu().numpy()
        return 0.5 * obj + 0.5 * goal
    elif task == "PegInsertionSide-v1":
        obj = env.peg.pose.p[0].cpu().numpy()
        goal = env.goal_pose.p[0].cpu().numpy()
        return 0.5 * obj + 0.5 * goal
    return np.array([0.0, 0.0, 0.1])

def update_wrist_follow_camera(env, task):
    import numpy as np
    from mani_skill.utils import sapien_utils
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

def get_state_dict(env, obs, task, use_moving_mounted_camera=False):
    if isinstance(obs, dict) and "image" in obs:
        return obs
    if use_moving_mounted_camera:
        update_wrist_follow_camera(env.unwrapped, task)
    if hasattr(env.unwrapped, "render_rgb_array"):
        image = env.unwrapped.render_rgb_array(camera_name="render_camera")
    else:
        image = env.render()
    if hasattr(image, "cpu"): image = image.cpu().numpy()
    if image.ndim == 4: image = image[0]
    image = image.astype(np.uint8)
    return {"state": obs, "image": image}

import logging

class StreamToLogger(object):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PushCube-v1")
    parser.add_argument("--max-steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=1024) # Can be 1024 for more stable gradients
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=5000) # Can be 10000
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Enable deterministic operations in PyTorch (enabled by default)")
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic", help="Disable deterministic operations")

    # Eval
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--target-success-rate", type=float, default=0.90)
    parser.add_argument("--save-dir", type=str, default="finetuning_output/cleanrl_crossq/weights")
    parser.add_argument("--reward-stage1", type=float, default=5.0, help="First reward milestone for checkpointing")
    parser.add_argument("--reward-stage2", type=float, default=10.0, help="Second reward milestone for checkpointing")

    # VLM
    parser.add_argument("--use-env-rewards", action="store_true", help="Bypass VLM and use native env dense rewards for debugging")
    parser.add_argument("--moving-mounted-camera", action="store_true", help="Enable the wrist-follow camera (moving mounted) instead of the default static camera")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="outputs/qwen3vl-progress-lora-tiny")
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--vlm-context-len", type=int, default=20)
    parser.add_argument("--vlm-batch-size", type=int, default=1) # Max 4, less is best for stability!
    parser.add_argument("--difference-rewards", action="store_true", help="Use difference in progress instead of absolute progress as reward")
    parser.add_argument("--4bit-quant", dest="quant_4bit", action="store_true", help="Enable 4-bit quantization for the VLM")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision instead of float16")

    # CrossQ & SAC Hyperparameters
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256) # Can be 2048 for better performance, with lr = 1e-3
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.8) # Can be 0.99 for better long-term performance, but 0.8 is more stable for VLM-based rewards and short-horizon tasks.
    parser.add_argument("--alpha", type=float, default=None, help="Fixed entropy regularization. If not provided, alpha is auto-tuned.")
    parser.add_argument("--target-entropy", type=float, default=None, help="Target entropy for auto-tuning. If not provided, defaults to -action_dim.")
    parser.add_argument("--bootstrap-at-done", type=str, default="always", choices=["always", "never"], help="Whether to bootstrap at terminal states.")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--policy-delay", type=int, default=3)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.task)
    os.makedirs(args.save_dir, exist_ok=True)
    
    log_file = os.path.join(args.save_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    set_seed(args.seed, args.deterministic)

    device = torch.device(args.device)

    # 1. Init Environment
    env_kwargs = {
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pos",
        "render_mode": "rgb_array",
        "sim_backend": "physx_cpu",
        "render_backend": "sapien_cpu",
        "reward_mode": "sparse",
    }
    if args.use_env_rewards:
        env_kwargs["reward_mode"] = "normalized_dense"
        env_kwargs["render_mode"] = None

    env = gym.make(args.task, **env_kwargs)
    obs, _ = env.reset()
    state_dim = obs.shape[-1]
    action_dim = env.action_space.shape[-1]
    max_action = float(env.action_space.high.flatten()[0])

    # 2. Init VLM Model
    if not args.use_env_rewards:
        print("Loading VLM...")
        processor = AutoProcessor.from_pretrained(args.model_id)
        
        model_kwargs = {"device_map": args.device}
        dtype = torch.bfloat16 if args.bf16 else torch.float16
        
        if args.quant_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype
            )
        else:
            model_kwargs["torch_dtype"] = dtype
            
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, **model_kwargs
        )
        if args.adapter_dir:
            print(f"Loading LoRA adapter: {args.adapter_dir}")
            model = PeftModel.from_pretrained(model, args.adapter_dir)
            if not args.quant_4bit:
                print("Merging LoRA weights into base model...")
                model = model.merge_and_unload()
        model.eval()
        processor.tokenizer.padding_side = 'left'
    
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/configs.yaml"))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        prompt_template = config["finetuning_prompt_template"]
        task_description = config["tasks"][args.task]["description"]
    else:
        print("Bypassing VLM (--use-env-rewards is enabled).")

    # 3. Init CrossQ Networks
    warmup_steps = int(args.max_steps * 0.04)
    actor = PolicyNetwork(state_dim, action_dim, max_action, hidden_dim=args.actor_hidden_dim).to(device)
    critic = QNetwork(state_dim, action_dim, hidden_dim=args.critic_hidden_dim, warmup_steps=warmup_steps).to(device)

    # CrossQ: Adam beta1 = 0.5 (or args.adam_beta1)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, 0.999))
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, 0.999))

    if args.alpha is None:
        target_entropy = args.target_entropy if args.target_entropy is not None else -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate, betas=(args.adam_beta1, 0.999))
    else:
        alpha = args.alpha

    buffer = SimpleReplayBuffer(capacity=args.buffer_size)

    # 4. Training Loop
    obs, _ = env.reset()
    state_dict = get_state_dict(env, obs, args.task, args.moving_mounted_camera)
    episode_data = []
    unannotated_episodes = []
    
    global_step = 0
    total_training_updates = 0
    total_episodes_completed = 0
    
    best_success_rate = 0.0
    consecutive_successes = 0
    last_eval_step = 0
    
    saved_ckpt_10_reward = False
    saved_ckpt_20_reward = False
    saved_ckpt_50_success = False

    latest_checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if args.resume and os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            print("Checkpoint loaded successfully!")
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
            saved_ckpt_10_reward = checkpoint["saved_ckpt_10_reward"]
            saved_ckpt_20_reward = checkpoint["saved_ckpt_20_reward"]
            saved_ckpt_50_success = checkpoint["saved_ckpt_50_success"]
            buffer.buffer.extend(checkpoint["buffer"])
            unannotated_episodes = checkpoint["unannotated_episodes"]
            
            if args.alpha is None:
                log_alpha.data = checkpoint["log_alpha"].data
                alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
                
            print(f"Resumed from global_step {global_step} (Completed episodes: {total_episodes_completed}, Updates: {total_training_updates})")
        except Exception as e:
            print(f"Failed to load checkpoint from {latest_checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    print("Starting Synchronous CleanRL CrossQ Training...")
    random_action_momentum = env.action_space.sample()
    target_action = env.action_space.sample()
    
    while global_step < args.max_steps:
        # --- ROLLOUT ---
        # Random exploration for initial phase
        if global_step < args.learning_starts:
            if global_step % 10 == 0:
                target_action = env.action_space.sample()
            random_action_momentum = 0.90 * random_action_momentum + 0.10 * target_action
            action = np.clip(random_action_momentum, env.action_space.low, env.action_space.high)
        else:
            flat_state = state_dict["state"]
            if hasattr(flat_state, "cpu"): flat_state = flat_state.cpu().numpy()
            flat_state_ts = torch.FloatTensor(np.array(flat_state).flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = actor.sample(flat_state_ts)
            action = action.cpu().data.numpy().flatten()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if hasattr(reward, "cpu"): reward = reward.cpu().item()
        if hasattr(terminated, "cpu"): terminated = terminated.cpu().item()
        
        next_state_dict = get_state_dict(env, next_obs, args.task, args.moving_mounted_camera)
        
        flat_state = np.array(state_dict["state"] if isinstance(state_dict["state"], np.ndarray) else state_dict["state"].cpu().numpy()).flatten()
        flat_next_state = np.array(next_state_dict["state"] if isinstance(next_state_dict["state"], np.ndarray) else next_state_dict["state"].cpu().numpy()).flatten()
        
        episode_data.append({
            "state": flat_state,
            "image": state_dict["image"],
            "action": action,
            "next_state": flat_next_state,
            "done": terminated,
            "task_reward": reward
        })

        state_dict = next_state_dict
        global_step += 1

        # --- EPISODE END: ANNOTATE AND TRAIN ---
        if done:
            total_episodes_completed += 1
            unannotated_episodes.append(episode_data)
            episode_data = []
            obs, _ = env.reset()
            state_dict = get_state_dict(env, obs, args.task, args.moving_mounted_camera)
            
            # Reset random momentum for the new episode
            if global_step < args.learning_starts:
                random_action_momentum = env.action_space.sample()
                target_action = env.action_space.sample()

            if len(unannotated_episodes) >= args.vlm_batch_size:
                if args.use_env_rewards:
                    annotated_transitions = []
                    for ep in unannotated_episodes:
                        prev_phi = 0.0
                        for item in ep:
                            phi_next = item["task_reward"]
                            if args.difference_rewards:
                                r_t = phi_next - prev_phi
                            else:
                                r_t = phi_next
                            annotated_transitions.append((item["state"], item["action"], r_t, item["next_state"], item["done"]))
                            prev_phi = phi_next
                else:
                    print(f"Batch full! Annotating {len(unannotated_episodes)} episodes at step {global_step}...")
                    # Annotate with VLM
                    annotated_transitions = annotate_batch(
                        unannotated_episodes, model, processor, task_description, prompt_template, args.device, 
                        context_len=args.vlm_context_len,
                        base_episode_idx=(total_episodes_completed - len(unannotated_episodes)),
                        use_difference_rewards=args.difference_rewards
                    )
                
                # Push to replay buffer
                for t in annotated_transitions:
                    buffer.add(*t)
                    
                # Train Network (CrossQ Update)
                updates_to_do = len(annotated_transitions) * args.utd_ratio
                if len(buffer) > args.learning_starts:
                    for _ in range(updates_to_do):
                        # Sample batch
                        b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(args.batch_size)
                    
                        state_batch = torch.FloatTensor(b_states).to(device)
                        action_batch = torch.FloatTensor(b_actions).to(device)
                        reward_batch = torch.FloatTensor(b_rewards).view(-1, 1).to(device)
                        next_state_batch = torch.FloatTensor(b_next_states).to(device)
                        done_batch = torch.FloatTensor(b_dones).view(-1, 1).to(device)
                    
                        if args.alpha is None:
                            alpha = log_alpha.exp()
                        else:
                            alpha = torch.tensor(args.alpha).to(device)
                    
                        # === CROSSQ CRITIC UPDATE ===
                        with torch.no_grad():
                            next_action, next_log_prob = actor.sample(next_state_batch)
                        
                        # Concatenated forward pass! No Target Network!
                        cat_states = torch.cat([state_batch, next_state_batch], dim=0)
                        cat_actions = torch.cat([action_batch, next_action], dim=0)
                    
                        all_Q1, all_Q2 = critic(cat_states, cat_actions)
                    
                        current_Q1, next_Q1 = torch.split(all_Q1, args.batch_size)
                        current_Q2, next_Q2 = torch.split(all_Q2, args.batch_size)
                    
                        next_Q_min = torch.min(next_Q1, next_Q2)
                        target_q = (next_Q_min - alpha.detach() * next_log_prob).detach()
                        
                        if args.bootstrap_at_done == "always":
                            target_q = reward_batch + args.gamma * target_q
                        else:
                            target_q = reward_batch + (1 - done_batch) * args.gamma * target_q
                    
                        critic_loss = F.mse_loss(current_Q1, target_q) + F.mse_loss(current_Q2, target_q)
                    
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()
                    
                        total_training_updates += 1
                    
                        # === CROSSQ ACTOR UPDATE ===
                        if total_training_updates % args.policy_delay == 0:
                            for param in critic.parameters():
                                param.requires_grad = False
                            
                            pi_action, pi_log_prob = actor.sample(state_batch)
                        
                            # CRITICAL CROSSQ FIX: The Critic MUST be in eval mode during the actor update. 
                            # Otherwise, the BatchRenorm running stats will be corrupted by this unbalanced batch
                            # that only contains (s, pi(s)) and not the (s', pi(s')) concatenated mixture.
                            critic.eval()
                            pi_Q1, pi_Q2 = critic(state_batch, pi_action)
                            critic.train()
                        
                            pi_Q = torch.min(pi_Q1, pi_Q2)
                        
                            actor_loss = (alpha.detach() * pi_log_prob - pi_Q).mean()
                        
                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()
                        
                            for param in critic.parameters():
                                param.requires_grad = True
                            
                            # Alpha update
                            if args.alpha is None:
                                alpha_loss = -(log_alpha.exp() * (pi_log_prob + target_entropy).detach()).mean()
                                alpha_optimizer.zero_grad()
                                alpha_loss.backward()
                                alpha_optimizer.step()
                        
                        if total_training_updates % args.learning_starts == 0:
                            print(f"Buffer Stats -> Annotated Transitions: {len(buffer)} | Total Generated: {global_step}")
                            alpha_val = alpha.item() if args.alpha is None else args.alpha
                            print(f"[Learner Debug] Update {total_training_updates} | Alpha: {alpha_val:.3f} | Avg Single-Step Reward in Batch: {np.mean(b_rewards):.3f}")
                        
                unannotated_episodes = []

            # --- EVALUATION AND SAVING ---
            if global_step - last_eval_step >= args.eval_freq and global_step > args.learning_starts:
                print(f"\n--- Running Evaluation at {global_step} total steps ---")
                success_rate, avg_reward = evaluate_policy(actor, args.task, device, args.eval_episodes)
                print(f"Evaluation Success Rate: {success_rate * 100:.1f}% | Avg Total Reward: {avg_reward:.2f}")
                
                os.makedirs(args.save_dir, exist_ok=True)
                
                checkpoint_dict = {
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                    'global_step': global_step,
                    'avg_reward': avg_reward,
                    'success_rate': success_rate
                }
                
                if args.alpha is None:
                    checkpoint_dict['log_alpha'] = log_alpha
                    checkpoint_dict['alpha_optimizer_state_dict'] = alpha_optimizer.state_dict()
                
                if avg_reward > args.reward_stage1 and not saved_ckpt_10_reward:
                    saved_ckpt_10_reward = True
                    print(f"Agent reached > {args.reward_stage1} reward! Saving checkpoint...")
                    torch.save(checkpoint_dict, f"{args.save_dir}/agent_reward_stage1.pth")

                if avg_reward > args.reward_stage2 and not saved_ckpt_20_reward:
                    saved_ckpt_20_reward = True
                    print(f"Agent reached > {args.reward_stage2} reward! Saving checkpoint...")
                    torch.save(checkpoint_dict, f"{args.save_dir}/agent_reward_stage2.pth")

                if success_rate >= 0.50 and not saved_ckpt_50_success:
                    saved_ckpt_50_success = True
                    print("Agent reached 50% success! Saving checkpoint...")
                    torch.save(checkpoint_dict, f"{args.save_dir}/agent_success_50.pth")
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    print("New best success rate! Saving agent...")
                    torch.save(checkpoint_dict, f"{args.save_dir}/best_agent.pth")
                    
                if success_rate >= args.target_success_rate:
                    consecutive_successes += 1
                    if consecutive_successes >= 2:
                        print(f"\nConvergence Reached! Success rate >= {args.target_success_rate*100}% for 2 consecutive evaluations.")
                        torch.save(checkpoint_dict, f"{args.save_dir}/converged_agent.pth")
                        break
                else:
                    consecutive_successes = 0
                    
                checkpoint_dict["total_training_updates"] = total_training_updates
                checkpoint_dict["total_episodes_completed"] = total_episodes_completed
                checkpoint_dict["best_success_rate"] = best_success_rate
                checkpoint_dict["consecutive_successes"] = consecutive_successes
                checkpoint_dict["last_eval_step"] = global_step
                checkpoint_dict["saved_ckpt_10_reward"] = saved_ckpt_10_reward
                checkpoint_dict["saved_ckpt_20_reward"] = saved_ckpt_20_reward
                checkpoint_dict["saved_ckpt_50_success"] = saved_ckpt_50_success
                checkpoint_dict["buffer"] = buffer.buffer
                checkpoint_dict["unannotated_episodes"] = unannotated_episodes
                
                torch.save(checkpoint_dict, f"{args.save_dir}/latest_checkpoint.pth")
                    
                last_eval_step = global_step

    print("Training finished! Saving final checkpoint...")
    final_checkpoint_dict = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'global_step': global_step,
        'avg_reward': avg_reward if 'avg_reward' in locals() else 0.0,
        'success_rate': success_rate if 'success_rate' in locals() else 0.0
    }
    
    if args.alpha is None:
        final_checkpoint_dict['log_alpha'] = log_alpha
        final_checkpoint_dict['alpha_optimizer_state_dict'] = alpha_optimizer.state_dict()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(final_checkpoint_dict, f"{args.save_dir}/final_agent.pth")

if __name__ == "__main__":
    main()
