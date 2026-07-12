import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.cleanrl_crossq import PolicyNetwork, evaluate_policy, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate new-format CrossQ actor checkpoints on fixed held-out seeds."
    )
    parser.add_argument("checkpoints", nargs="+", type=str)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=200000)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    return parser.parse_args()


def load_actor(checkpoint_path: str, device: torch.device) -> tuple[PolicyNetwork, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = [
        "task",
        "state_dim",
        "action_dim",
        "max_action",
        "actor_hidden_dim",
        "actor_state_dict",
        "checkpoint_type",
        "global_step",
    ]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing required metadata: {missing_keys}"
        )

    actor = PolicyNetwork(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        max_action=checkpoint["max_action"],
        hidden_dim=checkpoint["actor_hidden_dim"],
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    metadata = {
        "task": checkpoint["task"],
        "checkpoint_type": checkpoint["checkpoint_type"],
        "global_step": checkpoint["global_step"],
    }
    return actor, metadata


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    set_seed(args.base_seed, deterministic=True)

    results: List[Dict[str, Any]] = []
    for checkpoint_path in args.checkpoints:
        actor, metadata = load_actor(checkpoint_path, device)
        success_rate, avg_reward = evaluate_policy(
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
                "success_rate": success_rate,
                "average_normalized_dense_return": avg_reward,
            }
        )
        del actor

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
