import argparse
import json
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer-state",
        default="trainer_state.json",
        help="Path to trainer_state.json",
    )
    parser.add_argument(
        "--output",
        default="loss_curve.png",
        help="Path to save the loss curve image",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale for y-axis",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.trainer_state, "r") as f:
        state = json.load(f)

    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = []

    for log in state["log_history"]:
        if "loss" in log:
            train_epochs.append(log["epoch"])
            train_losses.append(log["loss"])

        if "eval_loss" in log:
            eval_epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])

    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_losses, label="Train loss")
    plt.plot(eval_epochs, eval_losses, marker="o", label="Eval loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if args.log_y:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved loss curve to: {args.output}")


if __name__ == "__main__":
    main()