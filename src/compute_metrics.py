import os
import json
import yaml
import numpy as np
from scipy.stats import spearmanr
from utilities import set_all_seeds

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

seed = config.get("seed")
set_all_seeds(seed)

def compute_metrics():
    data_root = config.get("data_root")
    output_root = config.get("output_root")
    experiment_name = config.get("experiment_name", "exp_default")
    rewards_path = os.path.join(output_root, experiment_name)
    data_path = data_root
    
    if not os.path.exists(rewards_path):
        print(f"Error: Rewards path {rewards_path} does not exist.")
        return

    # Find all reward files
    reward_files = [f for f in os.listdir(rewards_path) if f.endswith("_rewards.json")]
    
    overall_metrics = {}
    
    global_pearson = []
    global_preference = []

    for reward_file in reward_files:
        with open(os.path.join(rewards_path, reward_file), "r") as f:
            vlm_data = json.load(f)
        
        task = vlm_data["task"]
        step_type = vlm_data["step"]
        results = vlm_data["results"]
        
        task_pearson = []
        task_preference = []
        
        level = vlm_data["level"]

        for rollout_id, frames_data in results.items():
            
            # 1. Load Ground Truth Rewards
            gt_metadata_path = os.path.join(data_path, task, level, rollout_id, "rewards.json")
                
            if not os.path.exists(gt_metadata_path):
                continue
                
            with open(gt_metadata_path, "r") as f:
                gt_rewards = json.load(f)
            
            # 2. Align VLM rewards with GT rewards (Sort chronologically)
            frames_data_sorted = sorted(frames_data, key=lambda x: int(x["frame"].split("_")[2].split(".")[0]))
            
            vlm_scores = []
            gt_aligned_rewards = []
            
            total_frames = len(gt_rewards)
            
            for entry in frames_data_sorted:
                frame_idx = int(entry["frame"].split("_")[2].split(".")[0])
                vlm_scores.append(float(entry["score"]))
                
                # Ground truth is temporal progress (t / T-1) mapped to 0-100%
                gt_percentage = (frame_idx / (total_frames - 1)) * 100.0 if total_frames > 1 else 0.0
                gt_aligned_rewards.append(gt_percentage)

            if len(vlm_scores) < 2:
                continue

            # 3. Compute Spearman Correlation with GT
            corr, _ = spearmanr(vlm_scores, gt_aligned_rewards)
            if not np.isnan(corr):
                task_pearson.append(corr)
                global_pearson.append(corr)

            # 4. Compute VOC
            # rank-correlation(argsort(v_tilde), arange(T))
            voc, _ = spearmanr(np.argsort(vlm_scores), np.arange(len(vlm_scores)))
            if not np.isnan(voc):
                task_preference.append(voc)
                global_preference.append(voc)
            
            # Save raw scores for this rollout
            if "raw_scores" not in overall_metrics.setdefault(f"{task}_{step_type}", {}):
                overall_metrics[f"{task}_{step_type}"]["raw_scores"] = []
                
            overall_metrics[f"{task}_{step_type}"]["raw_scores"].append({
                "rollout": rollout_id,
                "spearman": float(corr) if not np.isnan(corr) else None,
                "voc": float(voc) if not np.isnan(voc) else None
            })

        # Average metrics for this task/step combination
        if task_pearson:
            metric_key = f"{task}_{step_type}"
            overall_metrics[metric_key].update({
                "avg_spearman": float(np.mean(task_pearson)),
                "avg_voc": float(np.mean(task_preference)) if task_preference else 0.0,
                "median_spearman": float(np.median(task_pearson)),
                "median_voc": float(np.median(task_preference)) if task_preference else 0.0,
                "num_rollouts": len(task_pearson)
            })

    # Add global metrics to overall_metrics
    if global_pearson:
        overall_metrics["global_metrics"] = {
            "avg_spearman": float(np.mean(global_pearson)),
            "avg_voc": float(np.mean(global_preference)) if global_preference else 0.0,
            "median_spearman": float(np.median(global_pearson)),
            "median_voc": float(np.median(global_preference)) if global_preference else 0.0,
            "num_rollouts": len(global_pearson)
        }

    # Print and Save results
    print("\nReward Model Metrics Alignment:")
    print("-" * 85)
    print(f"{'Task & Step':<30} | {'Avg Spear.':<10} | {'Med Spear.':<10} | {'Avg VOC':<10} | {'Med VOC':<10}")
    print("-" * 85)
    for key, m in overall_metrics.items():
        if key != "global_metrics" and "avg_spearman" in m:
            print(f"{key:<30} | {m['avg_spearman']:<10.3f} | {m['median_spearman']:<10.3f} | {m['avg_voc']:<10.3f} | {m['median_voc']:<10.3f}")
            
    if "global_metrics" in overall_metrics:
        print("-" * 85)
        m = overall_metrics["global_metrics"]
        print(f"{'GLOBAL (ALL TASKS)':<30} | {m['avg_spearman']:<10.3f} | {m['median_spearman']:<10.3f} | {m['avg_voc']:<10.3f} | {m['median_voc']:<10.3f}")

    output_file = os.path.join(rewards_path, "metrics.json")
    with open(output_file, "w") as f:
        json.dump(overall_metrics, f, indent=4)
    print(f"\nSaved detailed metrics to {output_file}")

if __name__ == "__main__":
    compute_metrics()
