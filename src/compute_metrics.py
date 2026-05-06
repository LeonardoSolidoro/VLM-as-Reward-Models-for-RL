import os
import json
import yaml
import numpy as np
from scipy.stats import pearsonr

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def compute_metrics():
    data_root = config.get("data_root")
    output_root = config.get("output_root")
    camera_name = config.get("camera_name")
    rewards_path = os.path.join(output_root, camera_name)
    data_path = os.path.join(data_root, camera_name)
    
    if not os.path.exists(rewards_path):
        print(f"Error: Rewards path {rewards_path} does not exist.")
        return

    # Find all reward files
    reward_files = [f for f in os.listdir(rewards_path) if f.endswith("_rewards.json")]
    
    overall_metrics = {}

    for reward_file in reward_files:
        with open(os.path.join(rewards_path, reward_file), "r") as f:
            vlm_data = json.load(f)
        
        task = vlm_data["task"]
        step_type = vlm_data["step"]
        results = vlm_data["results"]
        
        task_pearson = []
        task_preference = []

        for level, rollouts in results.items():
            for rollout_id, frames_data in rollouts.items():
                
                # 1. Load Ground Truth Rewards
                gt_metadata_path = os.path.join(data_path, task, level, rollout_id, "metadata.json")
                
                with open(gt_metadata_path, "r") as f:
                    gt_metadata = json.load(f)
                
                gt_rewards = gt_metadata["rewards"]
                
                # 2. Align VLM rewards with GT rewards
                vlm_scores = []
                gt_aligned_rewards = []
                
                for entry in frames_data:
                    frame_idx = int(entry["frame"].split("_")[1].split(".")[0])
                    vlm_scores.append(float(entry["score"]))
                    # MetaWorld rewards are per step, frames are rendered at each step
                    gt_aligned_rewards.append(gt_rewards[frame_idx])

                if len(vlm_scores) < 2:
                    continue

                # 3. Compute Pearson Correlation
                corr, _ = pearsonr(vlm_scores, gt_aligned_rewards)
                if not np.isnan(corr):
                    task_pearson.append(corr)

                # 4. Compute Preference Alignment
                # Sample random pairs to check if VLM and GT agree on which is better
                correct_preferences = 0
                total_pairs = 0
                n = len(vlm_scores)
                
                # Use all non-identical pairs for smaller datasets, or a subset for larger
                for i in range(n):
                    for j in range(i + 1, n):
                        # Ground Truth Preference
                        gt_diff = gt_aligned_rewards[i] - gt_aligned_rewards[j]
                        # VLM Preference
                        vlm_diff = vlm_scores[i] - vlm_scores[j]
                        
                        # Only count if GT has a clear preference (difference > 0)
                        if abs(gt_diff) > 1e-5:
                            total_pairs += 1
                            if (gt_diff > 0 and vlm_diff > 0) or (gt_diff < 0 and vlm_diff < 0):
                                correct_preferences += 1
                
                if total_pairs > 0:
                    task_preference.append(correct_preferences / total_pairs)

        # Average metrics for this task/step combination
        if task_pearson:
            metric_key = f"{task}_{step_type}"
            overall_metrics[metric_key] = {
                "avg_pearson": float(np.mean(task_pearson)),
                "avg_preference_alignment": float(np.mean(task_preference)) if task_preference else 0.0,
                "num_rollouts": len(task_pearson)
            }

    # Print and Save results
    print("\nReward Model Metrics Alignment:")
    print("-" * 60)
    print(f"{'Task & Step':<40} | {'Pearson':<8} | {'Pref Acc':<8}")
    print("-" * 60)
    for key, m in overall_metrics.items():
        print(f"{key:<40} | {m['avg_pearson']:<8.3f} | {m['avg_preference_alignment']:<8.3f}")

    output_file = os.path.join(rewards_path, "metrics.json")
    with open(output_file, "w") as f:
        json.dump(overall_metrics, f, indent=4)
    print(f"\nSaved detailed metrics to {output_file}")

if __name__ == "__main__":
    compute_metrics()
