"""
Metrics for paper-reproduction task-progress predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr


def value_order_correlation(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("VOC requires at least two scalar scores")
    if not np.all(np.isfinite(values)):
        raise ValueError("VOC scores contain non-finite values")
    if np.unique(values).size == 1:
        return 0.0
    correlation, _ = spearmanr(values, np.arange(values.size))
    if np.isnan(correlation):
        raise ValueError("Spearman correlation is unexpectedly NaN")
    return float(correlation)


def _frame_number(frame_name: str) -> int:
    stem = Path(frame_name).stem
    marker = "_frame_"
    if marker not in stem:
        raise ValueError(f"Cannot parse frame number from {frame_name}")
    return int(stem.rsplit(marker, 1)[1])


def evaluate_result_files(result_files: Sequence[Path]) -> dict[str, Any]:
    if not result_files:
        raise ValueError("No result files were provided")

    grouped: dict[str, dict[str, Any]] = {}
    all_vocs: list[float] = []
    for result_file in result_files:
        payload = json.loads(result_file.read_text())
        task = payload["task"]
        level = payload["level"]
        results = payload["results"]
        key = f"{task}_{level}"
        rollout_metrics: list[dict[str, Any]] = []
        for rollout_id, frames in sorted(results.items()):
            ordered = sorted(frames, key=lambda item: _frame_number(item["frame"]))
            if len(ordered) < 3:
                raise ValueError(
                    f"{result_file}:{rollout_id} has only {len(ordered)} frames"
                )
            # Frame zero is supplied as the prompt anchor rather than predicted.
            scores = [float(item["score"]) for item in ordered[1:]]
            voc = value_order_correlation(scores)
            rollout_metrics.append({"rollout": rollout_id, "voc": voc})
            all_vocs.append(voc)
        vocs = [row["voc"] for row in rollout_metrics]
        if not vocs:
            raise ValueError(f"{result_file} contains no evaluable rollouts")
        grouped[key] = {
            "mean_voc": float(np.mean(vocs)),
            "median_voc": float(np.median(vocs)),
            "num_rollouts": len(vocs),
            "rollouts": rollout_metrics,
        }

    if not all_vocs:
        raise ValueError("No evaluable rollouts were found")
    global_metrics = {
        "mean_voc": float(np.mean(all_vocs)),
        "median_voc": float(np.median(all_vocs)),
        "std_voc": float(np.std(all_vocs)),
        "num_rollouts": len(all_vocs),
    }
    return {
        "schema_version": 2,
        "groups": grouped,
        "global": global_metrics,
    }


def save_plots(metrics: dict[str, Any], output_dir: Path) -> list[Path]:
    """
    Save the two compact figures used for the API-baseline comparison.
    """
    import matplotlib.pyplot as plt

    if metrics["schema_version"] != 2:
        raise ValueError(f"Unsupported metrics schema: {metrics['schema_version']}")
    groups = metrics["groups"]
    task_keys = list(groups)
    if not task_keys:
        raise ValueError("Metrics contain no per-task entries")
    all_vocs = [
        float(row["voc"])
        for key in task_keys
        for row in groups[key]["rollouts"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    histogram_path = output_dir / "voc_histogram.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        all_vocs,
        bins=np.arange(-1.0, 1.1, 0.1),
        color="#5DADE2",
        edgecolor="black",
        alpha=0.75,
    )
    axis.axvline(
        metrics["global"]["mean_voc"],
        color="red",
        linestyle="--",
        label=f"Mean: {metrics['global']['mean_voc']:.2f}",
    )
    axis.set(xlim=(-1.0, 1.0), xlabel="Value-order correlation", ylabel="Rollouts")
    axis.legend()
    figure.tight_layout()
    figure.savefig(histogram_path, dpi=200)
    plt.close(figure)

    aggregate_path = output_dir / "voc_by_task.png"
    labels = task_keys
    means = [groups[key]["mean_voc"] for key in task_keys]
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.6), 5))
    axis.bar(labels, means, color="#F47A38")
    axis.set(ylim=(-1.0, 1.0), ylabel="Mean value-order correlation")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    figure.savefig(aggregate_path, dpi=200)
    plt.close(figure)
    return [histogram_path, aggregate_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_files = sorted(args.results_dir.glob("*_rewards.json"))
    metrics = evaluate_result_files(result_files)
    output = args.output or args.results_dir / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    if not args.no_plots:
        plot_paths = save_plots(metrics, args.results_dir)
        print("Saved plots:", *(str(path) for path in plot_paths))
    print(json.dumps(metrics["global"], indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
