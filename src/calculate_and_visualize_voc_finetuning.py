import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to prediction JSON, e.g. outputs/tiny_overfit_predictions.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where VOC metrics and histogram will be saved",
    )
    return parser.parse_args()


def compute_voc(pred_scores, target_scores):
    """
    VOC = Spearman correlation between predicted progress scores
    and target progress scores.

    This measures whether the predicted value ordering matches
    the ground-truth value ordering.
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)

    if len(pred_scores) != len(target_scores):
        min_len = min(len(pred_scores), len(target_scores))
        pred_scores = pred_scores[:min_len]
        target_scores = target_scores[:min_len]

    if len(pred_scores) < 2:
        return None

    # Spearman is undefined if either side is completely constant.
    if len(set(pred_scores.tolist())) == 1:
        return 0.0
    if len(set(target_scores.tolist())) == 1:
        return 0.0

    voc, _ = spearmanr(pred_scores, target_scores)

    if np.isnan(voc):
        return 0.0

    return float(voc)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r") as f:
        data = json.load(f)

    vocs = []
    all_preds = []
    all_targets = []

    for item in data:
        pred = item["generated_percentages"]
        target = item["target_percentages"]

        all_preds.extend(pred)
        all_targets.extend(target)

        voc = compute_voc(pred, target)

        if voc is not None:
            vocs.append(voc)

    if not vocs:
        raise ValueError("No valid VOC values were computed.")

    mean_voc = float(np.mean(vocs))
    median_voc = float(np.median(vocs))
    std_voc = float(np.std(vocs))

    metrics = {
        "input_json": args.input_json,
        "num_examples": len(vocs),
        "mean_voc": mean_voc,
        "median_voc": median_voc,
        "std_voc": std_voc,
        "vocs": vocs,
    }

    metrics_path = os.path.join(args.output_dir, "voc_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nVOC Results")
    print("-" * 40)
    print(f"Number of examples: {len(vocs)}")
    print(f"Mean VOC:          {mean_voc:.4f}")
    print(f"Median VOC:        {median_voc:.4f}")
    print(f"Std VOC:           {std_voc:.4f}")
    print(f"Saved metrics to:  {metrics_path}")

    # Calculate accuracy metrics
    pred_arr = np.array(all_preds)
    target_arr = np.array(all_targets)
    
    mae = float(np.mean(np.abs(pred_arr - target_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - target_arr)**2)))
    acc_5 = float(np.mean(np.abs(pred_arr - target_arr) <= 5) * 100)
    
    accuracy_metrics = {
        "num_frames": len(pred_arr),
        "mae": mae,
        "rmse": rmse,
        "accuracy_within_5_percent": acc_5,
    }
    
    accuracy_metrics_path = os.path.join(args.output_dir, "accuracy_metrics.json")
    with open(accuracy_metrics_path, "w") as f:
        json.dump(accuracy_metrics, f, indent=2)

    print("\nAccuracy Metrics")
    print("-" * 40)
    print(f"Total frames:      {len(pred_arr)}")
    print(f"MAE:               {mae:.2f}%")
    print(f"RMSE:              {rmse:.2f}%")
    print(f"Accuracy (+/- 5%): {acc_5:.2f}%")
    print(f"Saved metrics to:  {accuracy_metrics_path}")

    # VOC histogram only
    plt.figure(figsize=(8, 5))

    bins = np.linspace(-1.0, 1.0, 21)
    plt.hist(
        vocs,
        bins=bins,
        color="#5DADE2",
        edgecolor="black",
        alpha=0.7,
        label="VOC",
    )

    plt.axvline(
        mean_voc,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean VOC = {mean_voc:.3f}",
    )

    plt.axvline(
        median_voc,
        color="blue",
        linestyle=":",
        linewidth=2,
        label=f"Median VOC = {median_voc:.3f}",
    )

    plt.xlim(-1.0, 1.0)
    plt.xlabel("Value-Order Correlation (VOC)")
    plt.ylabel("Frequency")
    plt.title("VOC Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "voc_histogram.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved VOC histogram to: {plot_path}")


if __name__ == "__main__":
    main()