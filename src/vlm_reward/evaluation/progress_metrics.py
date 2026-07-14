"""
Evaluate supervised/contrastive task-progress predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Progress-prediction JSON produced by the inference script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for VOC, error metrics, and the histogram.",
    )
    return parser.parse_args()


def compute_voc(
    predicted_scores: Sequence[float],
    target_scores: Sequence[float],
) -> float:
    """
    Return Spearman value-order correlation for one trajectory.
    """
    predictions = np.asarray(predicted_scores, dtype=np.float64)
    targets = np.asarray(target_scores, dtype=np.float64)
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("VOC inputs must be one-dimensional")
    if len(predictions) != len(targets):
        raise ValueError(
            f"Received {len(predictions)} predictions for {len(targets)} targets"
        )
    if len(predictions) < 2:
        raise ValueError("VOC requires at least two predictions")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise ValueError("VOC inputs contain non-finite values")
    if np.unique(predictions).size == 1 or np.unique(targets).size == 1:
        return 0.0
    correlation, _ = spearmanr(predictions, targets)
    if np.isnan(correlation):
        raise FloatingPointError("Spearman correlation is unexpectedly NaN")
    return float(correlation)


def evaluate_predictions(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("Prediction file contains no examples")
    vocs: list[float] = []
    all_predictions: list[float] = []
    all_targets: list[float] = []
    for row in rows:
        predictions = [float(value) for value in row["generated_percentages"]]
        targets = [float(value) for value in row["target_percentages"]]
        vocs.append(compute_voc(predictions, targets))
        all_predictions.extend(predictions)
        all_targets.extend(targets)

    prediction_array = np.asarray(all_predictions, dtype=np.float64)
    target_array = np.asarray(all_targets, dtype=np.float64)
    if np.any(prediction_array < 0.0) or np.any(prediction_array > 100.0):
        raise ValueError("Generated progress percentages fall outside [0, 100]")
    if np.any(target_array < 0.0) or np.any(target_array > 100.0):
        raise ValueError("Target progress percentages fall outside [0, 100]")
    absolute_errors = np.abs(prediction_array - target_array)
    voc_metrics = {
        "num_examples": len(vocs),
        "mean_voc": float(np.mean(vocs)),
        "median_voc": float(np.median(vocs)),
        "std_voc": float(np.std(vocs)),
        "vocs": vocs,
    }
    accuracy_metrics = {
        "num_frames": len(prediction_array),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean((prediction_array - target_array) ** 2))),
        "accuracy_within_5_percent": float(np.mean(absolute_errors <= 5.0) * 100.0),
    }
    return voc_metrics, accuracy_metrics


def save_voc_histogram(voc_metrics: dict[str, Any], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        voc_metrics["vocs"],
        bins=np.linspace(-1.0, 1.0, 21),
        color="#5DADE2",
        edgecolor="black",
        alpha=0.7,
    )
    axis.axvline(
        voc_metrics["mean_voc"],
        color="red",
        linestyle="--",
        label=f"Mean VOC = {voc_metrics['mean_voc']:.3f}",
    )
    axis.axvline(
        voc_metrics["median_voc"],
        color="blue",
        linestyle=":",
        label=f"Median VOC = {voc_metrics['median_voc']:.3f}",
    )
    axis.set(
        xlim=(-1.0, 1.0),
        xlabel="Value-order correlation (VOC)",
        ylabel="Frequency",
        title="VOC distribution",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.input_json.is_file():
        raise FileNotFoundError(args.input_json)
    rows = json.loads(args.input_json.read_text())
    if not isinstance(rows, list):
        raise TypeError("Prediction JSON must contain a list")
    voc_metrics, accuracy_metrics = evaluate_predictions(rows)
    voc_metrics["input_json"] = str(args.input_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    voc_path = args.output_dir / "voc_metrics.json"
    accuracy_path = args.output_dir / "accuracy_metrics.json"
    histogram_path = args.output_dir / "voc_histogram.png"
    voc_path.write_text(json.dumps(voc_metrics, indent=2) + "\n")
    accuracy_path.write_text(json.dumps(accuracy_metrics, indent=2) + "\n")
    save_voc_histogram(voc_metrics, histogram_path)

    print(json.dumps({"voc": voc_metrics, "accuracy": accuracy_metrics}, indent=2))
    print(f"Saved metrics and histogram to {args.output_dir}")


if __name__ == "__main__":
    main()
