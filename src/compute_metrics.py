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
    output_root = config.get("output_root")
    experiment_name = config.get("experiment_name", "exp_default")
    rewards_path = os.path.join(output_root, experiment_name)
    
    if not os.path.exists(rewards_path):
        print(f"Error: Rewards path {rewards_path} does not exist.")
        return

    # Find all reward files
    reward_files = [f for f in os.listdir(rewards_path) if f.endswith("_rewards.json")]
    
    overall_metrics = {}
    
    global_preference = []

    for reward_file in reward_files:
        with open(os.path.join(rewards_path, reward_file), "r") as f:
            vlm_data = json.load(f)
        
        task = vlm_data["task"]
        step_type = vlm_data["step"]
        results = vlm_data["results"]
        
        task_preference = []

        for rollout_id, frames_data in results.items():
            
            # Align VLM rewards with chronological order
            frames_data_sorted = sorted(frames_data, key=lambda x: int(x["frame"].split("_")[2].split(".")[0]))
            
            vlm_scores = []
            
            for entry in frames_data_sorted:
                vlm_scores.append(float(entry["score"]))

            # Exclude the first frame from VOC computation since its score is hardcoded to 0.0 
            # and was provided in the prompt, not predicted by the VLM.
            vlm_scores = vlm_scores[1:]

            if len(vlm_scores) < 2:
                continue

            # Compute VOC
            # spearmanr internally converts values to ranks and handles ties correctly.
            # If the VLM predicts a perfectly flat line (e.g. all 100%), spearmanr is mathematically undefined (NaN).
            # We explicitly catch this and assign 0.0 because there is no temporal correlation.
            if len(set(vlm_scores)) == 1:
                voc = 0.0
            else:
                voc, _ = spearmanr(vlm_scores, np.arange(len(vlm_scores)))
            task_preference.append(voc)
            global_preference.append(voc)
            
            # Save raw scores for this rollout
            if "raw_scores" not in overall_metrics.setdefault(f"{task}_{step_type}", {}):
                overall_metrics[f"{task}_{step_type}"]["raw_scores"] = []
                
            overall_metrics[f"{task}_{step_type}"]["raw_scores"].append({
                "rollout": rollout_id,
                "voc": float(voc) if not np.isnan(voc) else None
            })

        # Average metrics for this task/step combination
        if task_preference:
            metric_key = f"{task}_{step_type}"
            if metric_key not in overall_metrics:
                overall_metrics[metric_key] = {}
            overall_metrics[metric_key].update({
                "avg_voc": float(np.mean(task_preference)),
                "median_voc": float(np.median(task_preference)),
                "num_rollouts": len(task_preference)
            })

    # Add global metrics to overall_metrics
    if global_preference:
        overall_metrics["global_metrics"] = {
            "avg_voc": float(np.mean(global_preference)),
            "median_voc": float(np.median(global_preference)),
            "num_rollouts": len(global_preference)
        }

    # Print and Save results
    print("\nReward Model Metrics Alignment:")
    print("-" * 55)
    print(f"{'Task & Step':<30} | {'Avg VOC':<10} | {'Med VOC':<10}")
    print("-" * 55)
    for key, m in overall_metrics.items():
        if key != "global_metrics" and "avg_voc" in m:
            print(f"{key:<30} | {m['avg_voc']:<10.3f} | {m['median_voc']:<10.3f}")
            
    if "global_metrics" in overall_metrics:
        print("-" * 55)
        m = overall_metrics["global_metrics"]
        print(f"{'GLOBAL (ALL TASKS)':<30} | {m['avg_voc']:<10.3f} | {m['median_voc']:<10.3f}")

    output_file = os.path.join(rewards_path, "metrics.json")
    with open(output_file, "w") as f:
        json.dump(overall_metrics, f, indent=4)
    print(f"\nSaved detailed metrics to {output_file}")

    # --- Plotting the Histogram ---
    import matplotlib.pyplot as plt

    task_vocs = {}
    for key, m in overall_metrics.items():
        if key == "global_metrics": continue
        raw = m.get("raw_scores", [])
        vocs = [r["voc"] for r in raw if r["voc"] is not None]
        if vocs:
            task_vocs[key] = vocs

    if task_vocs:
        bins = np.arange(-1.0, 1.1, 0.1)
        plt.figure(figsize=(10, 6))
        
        labels = list(task_vocs.keys())
        data = [task_vocs[label] for label in labels]
        
        plt.hist(data, bins=bins, stacked=True, label=labels, edgecolor='black', alpha=0.8)
        
        global_avg = overall_metrics["global_metrics"]["avg_voc"]
        global_med = overall_metrics["global_metrics"]["median_voc"]
        
        plt.axvline(global_avg, color='red', linestyle='dashed', linewidth=2, label=f'Global Mean ({global_avg:.2f})')
        plt.axvline(global_med, color='blue', linestyle='dotted', linewidth=2, label=f'Global Median ({global_med:.2f})')
        
        plt.xlim(-1.0, 1.0)
        plt.xlabel('Value-Order Correlation (VOC)')
        plt.ylabel('Frequency')
        plt.title(f'VOC Distribution Across All Tasks ({experiment_name})')
        plt.legend()
        
        plot_path = os.path.join(rewards_path, "voc_histogram.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved VOC histogram plot to {plot_path}")

if __name__ == "__main__":
    compute_metrics()
