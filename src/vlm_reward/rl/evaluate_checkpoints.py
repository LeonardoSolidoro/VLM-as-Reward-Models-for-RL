"""
Evaluate saved CrossQ actor checkpoints on held-out ManiSkill seeds.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from vlm_reward.rl.checkpoints import atomic_json_dump, load_actor_checkpoint
from vlm_reward.rl.core import PolicyNetwork, set_seed
from vlm_reward.rl.evaluation import evaluate_actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate new-format CrossQ actor checkpoints on fixed held-out seeds."
    )
    parser.add_argument("checkpoints", nargs="+", type=str)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=200000)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write clean JSON directly to this file (recommended on macOS/Vulkan).",
    )
    return parser.parse_args()


def load_actor(checkpoint_path: str, device: torch.device) -> tuple[PolicyNetwork, Dict[str, Any]]:
    return load_actor_checkpoint(checkpoint_path, device)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    set_seed(args.base_seed, deterministic=True)
    # Resolve every path before ManiSkill initializes.  Some native rendering
    # backends can change process-level path state during environment teardown.
    checkpoint_paths = [
        str(Path(checkpoint_path).expanduser().resolve())
        for checkpoint_path in args.checkpoints
    ]
    output_path = (
        None
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )

    results: List[Dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        actor, metadata = load_actor(checkpoint_path, device)
        evaluation = evaluate_actor(
            actor=actor,
            task=metadata["task"],
            device=device,
            num_episodes=args.episodes,
            base_seed=args.base_seed,
        )
        results.append(
            {
                "checkpoint": checkpoint_path,
                "checkpoint_type": metadata["checkpoint_type"],
                "global_step": metadata["global_step"],
                "task": metadata["task"],
                "episodes": args.episodes,
                "base_seed": args.base_seed,
                "success_rate": evaluation.success_rate,
                "average_normalized_dense_return": (
                    evaluation.average_normalized_dense_return
                ),
            }
        )
        del actor

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(results, output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
