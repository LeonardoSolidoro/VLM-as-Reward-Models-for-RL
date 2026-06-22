import os
import sys
import time
import argparse
import random
import re
from collections import deque
import yaml

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

def evaluate_policy(actor, task, device, num_episodes=10):
    eval_env = gym.make(
        task,
        obs_mode="state", 
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
        
            is_success = info.get("success", False)
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
    def __init__(self, num_features, eps=1e-5, momentum=0.01):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
        
            # Snap to first batch stats to prevent huge eval() lag at step 0
            if self.num_batches_tracked.item() == 0:
                self.running_mean.copy_(batch_mean.detach())
                self.running_var.copy_(batch_var.detach())
        
            # CrossQ BRN Warmup
            step = self.num_batches_tracked.item()
            r_max = 1.0 if step < 100000 else 3.0
            d_max = 0.0 if step < 100000 else 5.0
        
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
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
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

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.tanh(log_std)
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample(self, state):
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
    def __init__(self, state_dim, action_dim, hidden_dim=2048):
        super().__init__()
        # CrossQ: 2048 width, BatchRenorm in the critic
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            BatchRenorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            BatchRenorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            BatchRenorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            BatchRenorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

# ==============================================================================
# REPLAY BUFFER & VLM ANNOTATION LOGIC
# ==============================================================================

class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)

def parse_percentages(text):
    values = []
    for match in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        value = float(match) / 100.0
        values.append(value)
    return values

def check_vlm_diagnostics(percentages, episode_idx=""):
    # Convert to 0-100 scale for intuitive printing and logic
    pcts = [p * 100 for p in percentages]
    if len(pcts) < 2:
        return
    
    for i in range(1, len(pcts)):
        prev = pcts[i-1]
        curr = pcts[i]
    
        if prev > 20 and curr < 5:
            print(f"[VLM Diagnostic] WARNING: Reward plummeted from {prev:.1f}% to {curr:.1f}% at step {episode_idx}. Possible occlusion!")
            continue 
        
        if abs(curr - prev) > 20:
            print(f"[VLM Diagnostic] WARNING: Massive reward jump ({abs(curr-prev):.1f}%) from {prev:.1f}% to {curr:.1f}% at step {episode_idx}. VLM is possibly hallucinating!")
        
    consecutive_identical = 1
    for i in range(1, len(pcts)):
        if pcts[i] == pcts[i-1] and pcts[i] > 0:
            consecutive_identical += 1
            if consecutive_identical == 3:
                print(f"[VLM Diagnostic] WARNING: VLM returned the EXACT same score ({pcts[i]:.1f}%) for 3+ consecutive frames at step {episode_idx}. It might be failing to detect small robotic arm movements.")
                break 
        else:
            consecutive_identical = 1

def annotate_batch(episodes_data, model, processor, task_description, prompt_template, device, context_len=20, gamma=0.99, global_step=""):
    """
    Synchronously annotates a batch of episodes using the VLM, computes PBRS rewards, 
    and returns a flattened list of (state, action, reward, next_state, done) transitions.
    """
    features = []
    batch_metadata = []

    for episode_data in episodes_data:
        n_states = len(episode_data) + 1
        s0_image = episode_data[0]["image"]
    
        if n_states <= context_len:
            remaining_indices = list(range(1, n_states))
        else:
            remaining_indices = random.sample(range(1, n_states), context_len - 1)
            remaining_indices.sort()
        
        remaining_images = []
        for idx in remaining_indices:
            if idx < len(episode_data):
                remaining_images.append(episode_data[idx]["image"])
            else:
                remaining_images.append(episode_data[-1]["image"])

        shuffle_order = list(range(len(remaining_indices)))
        random.shuffle(shuffle_order)
        shuffled_indices = [remaining_indices[i] for i in shuffle_order]
        shuffled_images = [remaining_images[i] for i in shuffle_order]

        pil_s0 = Image.fromarray(s0_image)
        pil_query_images = [Image.fromarray(img) for img in shuffled_images]
        all_pil_images = [pil_s0] + pil_query_images
    
        frames_list = build_frames_list(len(pil_query_images))
        user_prompt = prompt_template.format(task_description=task_description, frames_list=frames_list)
        content = build_user_content(user_prompt, all_pil_images)
        messages = [{"role": "user", "content": content}]
    
        prompt_inputs = processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
    
        # Squeeze batch dimension (1, L) -> (L,)
        for k in ["input_ids", "attention_mask", "mm_token_type_ids"]:
            if k in prompt_inputs and prompt_inputs[k].ndim == 2:
                prompt_inputs[k] = prompt_inputs[k].squeeze(0)
            
        features.append(prompt_inputs)
        batch_metadata.append({
            "episode_data": episode_data,
            "shuffled_indices": shuffled_indices,
            "n_states": n_states
        })

    # Collate batch exactly like QwenProgressCollator
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    
    def left_pad(tensors, pad_value):
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_value, dtype=t.dtype, device=t.device)
                padded.append(torch.cat([padding, t]))
            else:
                padded.append(t)
        return torch.stack(padded)

    inputs = {
        "input_ids": left_pad([f["input_ids"] for f in features], pad_token_id),
        "attention_mask": left_pad([f["attention_mask"] for f in features], 0),
        "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
        "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
    }
    if "mm_token_type_ids" in features[0]:
        inputs["mm_token_type_ids"] = left_pad([f["mm_token_type_ids"] for f in features], 0)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, prompt_len:]
    generated_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    all_annotated_transitions = []

    for text, meta in zip(generated_texts, batch_metadata):
        percentages = parse_percentages(text)
        shuffled_indices = meta["shuffled_indices"]
        episode_data = meta["episode_data"]
        n_states = meta["n_states"]
    
        if len(percentages) != len(shuffled_indices):
            print(f"Warning: Expected {len(shuffled_indices)} percentages, got {len(percentages)}.")
            while len(percentages) < len(shuffled_indices):
                percentages.append(0.0)
            percentages = percentages[:len(shuffled_indices)]

        check_vlm_diagnostics(percentages, episode_idx=global_step)

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
            task_reward = item["task_reward"]
        
            if item["done"]:
                r_t = task_reward - progress[t_idx]
            else:
                r_t = task_reward + gamma * progress[t_idx + 1] - progress[t_idx]
            
            all_annotated_transitions.append((
                item["state"], item["action"], r_t, item["next_state"], item["done"]
            ))

    return all_annotated_transitions


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def get_state_dict(env, obs):
    if isinstance(obs, dict) and "image" in obs:
        return obs
    try:
        if hasattr(env.unwrapped, "render_rgb_array"):
            image = env.unwrapped.render_rgb_array(camera_name="render_camera")
        else:
            image = env.render()
        if hasattr(image, "cpu"): image = image.cpu().numpy()
        if image.ndim == 4: image = image[0]
        image = image.astype(np.uint8)
        return {"state": obs, "image": image}
    except:
        return {"state": obs, "image": np.zeros((128, 128, 3), dtype=np.uint8)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PickCube-v1")
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=256) # Can be 1024 for more stable gradients
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=5000) # Can be 10000

    # Eval
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--target-success-rate", type=float, default=0.90)
    parser.add_argument("--save-dir", type=str, default="finetuning_output/cleanrl_crossq/weights")

    # VLM
    parser.add_argument("--use-env-rewards", action="store_true", help="Bypass VLM and use native env dense rewards for debugging")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="outputs/qwen3vl-progress-lora-tiny")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vlm-context-len", type=int, default=20)
    parser.add_argument("--vlm-batch-size", type=int, default=1) # Max 4, less is best for stability!

    # CrossQ & SAC Hyperparameters
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256) # Can be 2048 for better performance, with lr = 1e-3
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.8) # Can be 0.99 for better long-term performance, but 0.8 is more stable for VLM-based rewards and short-horizon tasks.
    parser.add_argument("--bootstrap-at-done", type=str, default="always", choices=["always", "never"], help="Whether to bootstrap at terminal states.")
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--policy-delay", type=int, default=3)

    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Init Environment
    env_kwargs = {
        "obs_mode": "state",
        "render_mode": "rgb_array",
        "sim_backend": "physx_cpu",
        "render_backend": "sapien_cpu",
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
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, torch_dtype=torch.float16, device_map=args.device
        )
        if args.adapter_dir:
            model = PeftModel.from_pretrained(model, args.adapter_dir).merge_and_unload()
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
    actor = PolicyNetwork(state_dim, action_dim, max_action, hidden_dim=args.actor_hidden_dim).to(device)
    critic = QNetwork(state_dim, action_dim, hidden_dim=args.critic_hidden_dim).to(device)

    # CrossQ: Adam beta1 = 0.5 (or args.adam_beta1)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, 0.999))
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, 0.999))

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate, betas=(args.adam_beta1, 0.999))
    target_entropy = -action_dim

    buffer = SimpleReplayBuffer(capacity=args.buffer_size)

    # 4. Training Loop
    obs, _ = env.reset()
    state_dict = get_state_dict(env, obs)
    episode_data = []
    unannotated_episodes = []
    
    global_step = 0
    total_training_updates = 0
    
    best_success_rate = 0.0
    consecutive_successes = 0
    last_eval_step = 0
    
    print("Starting Synchronous CleanRL CrossQ Training...")
    
    while global_step < args.max_steps:
        # --- ROLLOUT ---
        # Random exploration for initial phase
        if global_step < args.learning_starts:
            action = env.action_space.sample()
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
        
        next_state_dict = get_state_dict(env, next_obs)
        
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
            unannotated_episodes.append(episode_data)
            episode_data = []
            obs, _ = env.reset()
            state_dict = get_state_dict(env, obs)

            if len(unannotated_episodes) >= args.vlm_batch_size or (args.use_env_rewards and len(unannotated_episodes) > 0):
                if args.use_env_rewards:
                    annotated_transitions = []
                    for ep in unannotated_episodes:
                        annotated_transitions.extend([(item["state"], item["action"], item["task_reward"], item["next_state"], item["done"]) for item in ep])
                else:
                    print(f"Batch full! Annotating {len(unannotated_episodes)} episodes at step {global_step}...")
                    # Annotate with VLM
                    annotated_transitions = annotate_batch(
                        unannotated_episodes, model, processor, task_description, prompt_template, args.device, 
                        context_len=args.vlm_context_len, gamma=args.gamma, global_step=global_step
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
                    
                        alpha = log_alpha.exp()
                    
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
                            alpha_loss = -(log_alpha.exp() * (pi_log_prob + target_entropy).detach()).mean()
                            alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            alpha_optimizer.step()
                        
                        if total_training_updates % 5000 == 0:
                            print(f"Buffer Stats -> Annotated Transitions: {len(buffer)} | Total Generated: {global_step}")
                            print(f"[Learner Debug] Update {total_training_updates} | Alpha: {alpha.item():.3f} | Avg Single-Step Reward in Batch: {np.mean(b_rewards):.3f}")
                        
                unannotated_episodes = []

            # --- EVALUATION AND SAVING ---
            if global_step - last_eval_step >= args.eval_freq and global_step > args.learning_starts:
                print(f"\n--- Running Evaluation at {global_step} total steps ---")
                success_rate, avg_reward = evaluate_policy(actor, args.task, device, args.eval_episodes)
                print(f"Evaluation Success Rate: {success_rate * 100:.1f}% | Avg Total Reward: {avg_reward:.2f}")
                
                os.makedirs(args.save_dir, exist_ok=True)
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    print("New best success rate! Saving agent...")
                    torch.save({
                        'actor_state_dict': actor.state_dict(),
                        'critic_state_dict': critic.state_dict(),
                        'log_alpha': log_alpha
                    }, f"{args.save_dir}/best_agent.pth")
                    
                if success_rate >= args.target_success_rate:
                    consecutive_successes += 1
                    if consecutive_successes >= 2:
                        print(f"\nConvergence Reached! Success rate >= {args.target_success_rate*100}% for 2 consecutive evaluations.")
                        torch.save({
                            'actor_state_dict': actor.state_dict(),
                            'critic_state_dict': critic.state_dict(),
                            'log_alpha': log_alpha
                        }, f"{args.save_dir}/converged_agent.pth")
                        break
                else:
                    consecutive_successes = 0
                    
                last_eval_step = global_step

if __name__ == "__main__":
    main()
