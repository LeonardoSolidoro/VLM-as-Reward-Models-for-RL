import os
import sys
import re
import numpy as np
from typing import Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.finetune_reward_contrastive.qwen_reward_contrastive_dataset import (
    REWARD_PROMPT_TEMPLATE,
    TASK_REWARD_GUIDANCE,
    build_frames_list,
    build_user_content,
)

def parse_reward_percentages(text: str) -> List[float]:
    score_matches = re.findall(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</score>", text, flags=re.IGNORECASE)
    if score_matches:
        return [float(value) / 100.0 for value in score_matches]

    percent_matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
    return [float(value) / 100.0 for value in percent_matches]

def clip_reward(value: float, episode_idx: int, transition_idx: int) -> float:
    if value < 0.0 or value > 1.0:
        print(f"[VLM Reward] Clipping predicted reward for episode {episode_idx}, transition {transition_idx}: {value:.4f}")
    return float(np.clip(value, 0.0, 1.0))

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

def annotate_batch(episodes_data: List[List[Dict[str, Any]]], model: nn.Module, processor: Any, task_name: str, task_description: str, device: torch.device, context_len: int = 20, base_episode_idx: int = 0, use_difference_rewards: bool = False, vlm_reward_scale: float = 1.0, global_step: int = 0, enable_filters: bool = False, phase1_steps: int = 20000, phase1_cap: float = 0.6, warmup_frames: int = 5, warmup_cap: float = 0.5, tv_threshold: float = 1.5) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
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
            remaining_indices = list(range(0, n_states))
        else:
            remaining_indices = np.round(np.linspace(0, n_states - 1, context_len)).astype(int).tolist()
        
        remaining_images = []
        for idx in remaining_indices:
            if idx < len(episode_data):
                remaining_images.append(episode_data[idx]["image"])
            else:
                remaining_images.append(episode_data[-1]["image"])

        import random
        combined = list(zip(remaining_indices, remaining_images))
        random.shuffle(combined)
        shuffled_indices = [x[0] for x in combined]
        shuffled_images = [x[1] for x in combined]

        pil_query_images = [Image.fromarray(img) for img in shuffled_images]
    
        frames_list = build_frames_list(len(pil_query_images))
        user_prompt = REWARD_PROMPT_TEMPLATE.format(
            task=task_name,
            task_description=task_description,
            reward_guidance=TASK_REWARD_GUIDANCE[task_name],
            frames_list=frames_list
        )
        content = build_user_content(user_prompt, pil_query_images)
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

    abs_errors = []

    for batch_idx, (text, meta) in enumerate(zip(generated_texts, batch_metadata)):
        percentages = parse_reward_percentages(text)
        shuffled_indices = meta["shuffled_indices"]
        episode_data = meta["episode_data"]
        n_states = meta["n_states"]
    
        if len(percentages) != len(shuffled_indices):
            print(f"Warning: Expected {len(shuffled_indices)} percentages, got {len(percentages)}.")
            while len(percentages) < len(shuffled_indices):
                percentages.append(0.0)
            percentages = percentages[:len(shuffled_indices)]

        # Restore chronological order before filtering and diagnostics
        chrono_pairs = sorted(zip(shuffled_indices, percentages), key=lambda x: x[0])
        shuffled_indices = [x[0] for x in chrono_pairs]
        percentages = [x[1] for x in chrono_pairs]

        current_ep_idx = base_episode_idx + batch_idx
        check_vlm_diagnostics(
            percentages, 
            shuffled_indices=shuffled_indices, 
            ep_idx=current_ep_idx
        )

        # 1. Median Filter (Window = 3) on the queried predictions
        if enable_filters:
            smoothed_percentages = list(percentages)
            for i in range(len(percentages)):
                start_i = max(0, i - 1)
                end_i = min(len(percentages), i + 2)
                smoothed_percentages[i] = float(np.median(percentages[start_i:end_i]))
            percentages = smoothed_percentages

        progress = [None] * n_states
        for idx, prog in zip(shuffled_indices, percentages):
            progress[idx] = clip_reward(prog, current_ep_idx, idx)
        
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

        if enable_filters:
            # 2. Volatility Rejection
            # Calculated BEFORE artificial caps flatten the signal, so we catch true VLM instability
            tv = sum(abs(progress[i] - progress[i-1]) for i in range(1, len(progress)))
            if tv > tv_threshold:
                print(f"[VLM Filter] Rejecting episode {current_ep_idx} due to high volatility (TV = {tv:.2f} > {tv_threshold})")
                continue

            # 3. Phase 1 Strict Cap
            phase1_triggered = False
            if global_step < phase1_steps:
                for i in range(len(progress)):
                    if progress[i] > phase1_cap:
                        progress[i] = phase1_cap
                        phase1_triggered = True
                if phase1_triggered:
                    print(f"[VLM Filter] Phase 1 cap ({phase1_cap}) triggered for episode {current_ep_idx}")
            else:
                # 4. Physical Warmup Constraint (Applies only when Phase 1 is inactive)
                warmup_triggered = False
                for i in range(min(warmup_frames, len(progress))):
                    if progress[i] > warmup_cap:
                        progress[i] = warmup_cap
                        warmup_triggered = True
                if warmup_triggered:
                    print(f"[VLM Filter] Physical Warmup cap ({warmup_cap}) triggered for early frames in episode {current_ep_idx}")

        for t_idx in range(len(episode_data)):
            item = episode_data[t_idx]
            
            # Difference Reward
            if use_difference_rewards:
                r_t = (progress[t_idx + 1] - progress[t_idx]) * vlm_reward_scale
            else:
                r_t = progress[t_idx + 1] * vlm_reward_scale
            
            abs_errors.append(abs(r_t - item["task_reward"]))
            
            all_annotated_transitions.append((
                item["state"], item["action"], r_t, item["next_state"], item["done"]
            ))

    if abs_errors:
        print(f"[VLM Reward] Batch env-reward MAE diagnostic: {np.mean(abs_errors):.4f}")

    return all_annotated_transitions
