import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REWARD_BINS: List[Tuple[float, float]] = [
    (0.0, 0.1),
    (0.1, 0.3),
    (0.3, 0.6),
    (0.6, 0.9),
    (0.9, 1.000001),
]


def mae(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def add_error(bucket: Dict[str, List[float]], key: str, error: float) -> None:
    if key not in bucket:
        bucket[key] = []
    bucket[key].append(error)


def bin_name(target_reward: float) -> str:
    for low, high in REWARD_BINS:
        if low <= target_reward < high:
            upper = high if high <= 1.0 else 1.0
            return f"{low:.1f}-{upper:.1f}"
    raise ValueError(f"Target reward outside expected bins: {target_reward}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    with input_path.open("r") as f:
        rows = json.load(f)

    all_errors: List[float] = []
    by_task: Dict[str, List[float]] = {}
    by_level: Dict[str, List[float]] = {}
    by_bin: Dict[str, List[float]] = {}

    for row in rows:
        generated = row["generated_rewards"]
        target = row["target_rewards"]
        count = min(len(generated), len(target))
        if count != len(target):
            print(f"Warning: {row['id']} has {len(generated)} predictions for {len(target)} targets")

        for pred_reward, target_reward in zip(generated[:count], target[:count]):
            pred_value = max(0.0, min(1.0, float(pred_reward)))
            target_value = float(target_reward)
            error = abs(pred_value - target_value)
            all_errors.append(error)
            add_error(by_task, row["task"], error)
            add_error(by_level, row["level"], error)
            add_error(by_bin, bin_name(target_value), error)

    metrics = {
        "input_json": str(input_path),
        "num_frames": len(all_errors),
        "mae": mae(all_errors),
        "mae_percent": mae(all_errors) * 100,
        "mae_by_task": {key: mae(errors) for key, errors in sorted(by_task.items())},
        "mae_by_level": {key: mae(errors) for key, errors in sorted(by_level.items())},
        "mae_by_reward_bin": {key: mae(errors) for key, errors in sorted(by_bin.items())},
    }

    output_path = Path(args.output_json) if args.output_json is not None else input_path.with_name("reward_mae_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nReward MAE")
    print("-" * 40)
    print(f"Frames:      {metrics['num_frames']}")
    print(f"MAE:         {metrics['mae']:.4f}")
    print(f"MAE percent: {metrics['mae_percent']:.2f}%")
    print(f"Saved to:    {output_path}")


if __name__ == "__main__":
    main()
