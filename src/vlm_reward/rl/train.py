"""Online CrossQ training with native or direct visual-head rewards.

The production visual-reward path intentionally excludes autoregressive text
generation.  The original progress-as-text reward trial remains runnable at
``src/historical_progress_reward/cleanrl_crossq.py``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
import numpy as np
import torch

from ..models.reward_predictor import QwenRewardHeadPredictor
from .checkpoints import (
    atomic_json_dump,
    atomic_torch_save,
    build_actor_export,
    load_training_checkpoint,
    validate_resume_arguments,
)
from .core import PolicyNetwork, QNetwork, SimpleReplayBuffer, set_seed
from .environments import get_step_state, state_to_numpy, success_to_bool
from .evaluation import evaluate_actor
from .rewards import (
    annotate_native_reward_episodes,
    annotate_visual_reward_episodes,
)
from .training_state import TrainingProgress, build_training_checkpoint
from .updates import CrossQUpdater


class StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self.logger = logger
        self.level = level

    def write(self, buffer: str) -> None:
        for line in buffer.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self) -> None:
        return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CrossQ using either ManiSkill normalized-dense reward or the "
            "frozen adapted-Qwen visual reward head."
        )
    )
    parser.add_argument("--task", default="PickCube-v1")
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument(
        "--random-steps",
        type=int,
        default=None,
        help="Random-action steps; defaults to --learning-starts.",
    )
    parser.add_argument(
        "--deterministic-prefill-steps",
        type=int,
        default=0,
        help="Initial steps using the initialized actor's deterministic mean.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume network, optimizer, replay, and global RNG state from the last "
            "episode-boundary checkpoint. ManiSkill's environment RNG stream is not "
            "bitwise restored."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")

    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--target-success-rate", type=float, default=0.90)
    parser.add_argument("--minimum-steps-before-convergence", type=int, default=0)
    parser.add_argument(
        "--bootstrap-target-success-rate",
        type=float,
        default=None,
        help="Stop a native-reward bootstrap run at the first matching evaluation.",
    )
    parser.add_argument(
        "--save-dir", default="finetuning_output/cleanrl_crossq_reward/weights"
    )
    parser.add_argument(
        "--save-eval-snapshots",
        action="store_true",
        help="Save every compact evaluation actor; best/initial/final are always saved.",
    )

    reward_group = parser.add_mutually_exclusive_group(required=False)
    reward_group.add_argument(
        "--use-env-rewards",
        action="store_true",
        help="Use ManiSkill normalized-dense reward to create a bootstrap actor.",
    )
    reward_group.add_argument(
        "--reward-head-checkpoint",
        default=None,
        help="Portable direct visual reward-head checkpoint used by the final experiment.",
    )
    parser.add_argument("--moving-mounted-camera", action="store_true")
    default_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument(
        "--annotation-episodes",
        "--vlm-batch-size",
        dest="vlm_batch_size",
        type=int,
        default=1,
        help="Completed episodes annotated together (legacy alias: --vlm-batch-size).",
    )
    parser.add_argument(
        "--reward-scale",
        "--vlm-reward-scale",
        dest="vlm_reward_scale",
        type=float,
        default=1.0,
        help="Scale non-overridden visual or native rewards.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--reward-head-batch-size", type=int, default=128)
    parser.add_argument(
        "--env-success-override",
        action="store_true",
        help="Set reward to 1.0 on transitions where ManiSkill reports success.",
    )

    # Accepted so commands from completed direct-head runs remain runnable.  These
    # options no longer trigger autoregressive reward generation.
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct", help=argparse.SUPPRESS)
    parser.add_argument(
        "--adapter-dir",
        default="finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--vlm-context-len", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument(
        "--vlm-generation-batch-size", type=int, default=1, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--4bit-quant", dest="quant_4bit", action="store_true", help=argparse.SUPPRESS
    )

    parser.add_argument("--init-actor-checkpoint", default=None)
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--restore-alpha-from-init", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--minimum-initial-success-rate", type=float, default=0.0)
    parser.add_argument("--initial-eval-episodes", type=int, default=20)
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument(
        "--bootstrap-at-done", choices=["always", "never"], default="always"
    )
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--policy-delay", type=int, default=3)
    parser.add_argument("--critic-only-warmup-updates", type=int, default=0)
    parser.add_argument("--batch-renorm-warmup-steps", type=int, default=None)
    parser.add_argument("--actor-max-grad-norm", type=float, default=1.0)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.random_steps is None:
        args.random_steps = args.learning_starts
    if args.actor_learning_rate is None:
        args.actor_learning_rate = args.learning_rate
    if not args.use_env_rewards and args.reward_head_checkpoint is None:
        raise ValueError(
            "Choose --use-env-rewards for bootstrap training or provide "
            "--reward-head-checkpoint. Autoregressive reward generation is retained "
            "only in src/historical_progress_reward/cleanrl_crossq.py."
        )
    if args.env_success_override and args.reward_head_checkpoint is None:
        raise ValueError("--env-success-override requires --reward-head-checkpoint")
    if args.reward_head_batch_size <= 0 or args.vlm_batch_size <= 0:
        raise ValueError("Reward annotation batch sizes must be positive")
    if args.vlm_reward_scale < 0.0:
        raise ValueError("--reward-scale must be non-negative")
    if args.resume and args.init_actor_checkpoint is not None:
        raise ValueError("--resume and --init-actor-checkpoint are mutually exclusive")
    if args.reference_kl_weight < 0.0:
        raise ValueError("--reference-kl-weight must be non-negative")
    if args.reference_kl_weight > 0.0 and args.init_actor_checkpoint is None and not args.resume:
        raise ValueError(
            "Positive --reference-kl-weight requires --init-actor-checkpoint or --resume"
        )
    if args.restore_alpha_from_init and args.init_actor_checkpoint is not None and args.alpha is not None:
        raise ValueError(
            "Cannot restore alpha from an initialization checkpoint with fixed --alpha"
        )
    if args.random_steps < 0 or args.deterministic_prefill_steps < 0:
        raise ValueError("Exploration and deterministic prefill steps must be non-negative")
    if args.random_steps > args.deterministic_prefill_steps > 0:
        raise ValueError("--random-steps cannot exceed --deterministic-prefill-steps")
    if args.deterministic_prefill_steps > args.learning_starts:
        raise ValueError("--deterministic-prefill-steps cannot exceed --learning-starts")
    if args.deterministic_prefill_steps > 0 and args.init_actor_checkpoint is None and not args.resume:
        raise ValueError("Deterministic prefill requires an initialized or resumed actor")
    if args.learning_starts < args.batch_size:
        raise ValueError("--learning-starts must be at least --batch-size")
    if args.batch_renorm_warmup_steps is not None and args.batch_renorm_warmup_steps <= 0:
        raise ValueError("--batch-renorm-warmup-steps must be positive")
    if args.actor_max_grad_norm <= 0.0:
        raise ValueError("--actor-max-grad-norm must be positive")
    if args.learning_rate <= 0.0 or args.actor_learning_rate <= 0.0:
        raise ValueError("Learning rates must be positive")
    if args.critic_only_warmup_updates < 0:
        raise ValueError("--critic-only-warmup-updates must be non-negative")
    if args.max_steps <= 0 or args.batch_size <= 0 or args.utd_ratio <= 0:
        raise ValueError("Step, batch, and update-to-data counts must be positive")
    if args.eval_freq <= 0 or args.initial_eval_episodes <= 0 or args.eval_episodes <= 0:
        raise ValueError("Evaluation frequencies and episode counts must be positive")
    if args.minimum_steps_before_convergence < 0:
        raise ValueError("--minimum-steps-before-convergence must be non-negative")
    for name, value in (
        ("target success rate", args.target_success_rate),
        ("minimum initial success rate", args.minimum_initial_success_rate),
    ):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if args.bootstrap_target_success_rate is not None:
        if not args.use_env_rewards:
            raise ValueError("--bootstrap-target-success-rate requires --use-env-rewards")
        if not 0.0 <= args.bootstrap_target_success_rate <= 1.0:
            raise ValueError("--bootstrap-target-success-rate must be in [0, 1]")
    if args.minimum_initial_success_rate > 0.0 and args.init_actor_checkpoint is None and not args.resume:
        raise ValueError(
            "--minimum-initial-success-rate requires an initialized or resumed actor"
        )


def _configure_output(args: argparse.Namespace) -> tuple[Path, Path, Optional[Path]]:
    # Resolve before ManiSkill/Vulkan initialization; native backends may change
    # the process working directory during environment teardown.
    task_directory = (Path(args.save_dir).expanduser() / args.task).resolve()
    task_directory.mkdir(parents=True, exist_ok=True)
    artifacts = list(task_directory.iterdir())
    if not args.resume and artifacts:
        raise FileExistsError(
            "Refusing to mix a new run with existing artifacts. Remove or rename "
            f"the run directory first: {artifacts}"
        )
    latest_checkpoint = task_directory / "latest_checkpoint.pth"
    if args.resume and not latest_checkpoint.is_file():
        raise FileNotFoundError(
            f"Cannot resume; checkpoint does not exist: {latest_checkpoint}"
        )
    snapshot_directory: Optional[Path] = None
    if args.save_eval_snapshots:
        snapshot_directory = task_directory / "eval_actors"
        snapshot_directory.mkdir(exist_ok=True)
    args.save_dir = str(task_directory)
    return task_directory, latest_checkpoint, snapshot_directory


def _configure_logging(task_directory: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(task_directory / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)


def _load_initial_actor(
    actor: PolicyNetwork,
    args: argparse.Namespace,
    state_dim: int,
    action_dim: int,
    max_action: float,
    checkpoint_path: Optional[Path],
) -> tuple[Optional[PolicyNetwork], float]:
    if checkpoint_path is None:
        return None, 0.0
    path = checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary at {path}")
    for key in ("actor_state_dict",):
        if key not in checkpoint:
            raise KeyError(f"Initialization checkpoint is missing {key!r}")
    optional_dimension_checks = (
        ("task", args.task),
        ("actor_hidden_dim", args.actor_hidden_dim),
        ("state_dim", state_dim),
        ("action_dim", action_dim),
    )
    for key, expected in optional_dimension_checks:
        if key in checkpoint and checkpoint[key] != expected:
            raise ValueError(
                f"Initialization {key} {checkpoint[key]!r} does not match {expected!r}"
            )
    if "max_action" in checkpoint and not np.isclose(checkpoint["max_action"], max_action):
        raise ValueError(
            f"Initialization max_action {checkpoint['max_action']} does not match {max_action}"
        )
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    reference_actor = copy.deepcopy(actor)
    reference_actor.eval()
    for parameter in reference_actor.parameters():
        parameter.requires_grad = False
    initial_log_alpha = 0.0
    if args.alpha is None and args.restore_alpha_from_init:
        if "log_alpha" not in checkpoint or checkpoint["log_alpha"] is None:
            raise KeyError(
                "Initialization checkpoint has no log_alpha; use "
                "--no-restore-alpha-from-init or provide --alpha"
            )
        initial_log_alpha = float(
            torch.as_tensor(checkpoint["log_alpha"])
            .detach()
            .reshape(-1)[0]
            .item()
        )
    print(f"Initialized actor only from {path}")
    print(
        "Frozen reference policy created from the imported actor. Critic, replay "
        "buffer, and optimizers are freshly initialized."
    )
    return reference_actor, initial_log_alpha


def _resume_immutable_arguments() -> List[str]:
    return [
        "use_env_rewards",
        "reward_head_checkpoint",
        "env_success_override",
        "moving_mounted_camera",
        "vlm_reward_scale",
        "vlm_batch_size",
        "vlm_generation_batch_size",
        "reward_head_batch_size",
        "model_id",
        "adapter_dir",
        "vlm_context_len",
        "bf16",
        "quant_4bit",
        "seed",
        "deterministic",
        "actor_hidden_dim",
        "critic_hidden_dim",
        "batch_renorm_warmup_steps",
        "learning_starts",
        "random_steps",
        "deterministic_prefill_steps",
        "batch_size",
        "buffer_size",
        "utd_ratio",
        "learning_rate",
        "actor_learning_rate",
        "adam_beta1",
        "gamma",
        "alpha",
        "target_entropy",
        "bootstrap_at_done",
        "policy_delay",
        "critic_only_warmup_updates",
        "reference_kl_weight",
        "actor_max_grad_norm",
    ]


def _write_evaluations(task_directory: Path, progress: TrainingProgress) -> None:
    atomic_json_dump(progress.evaluation_history, task_directory / "evaluations.json")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    task_directory, latest_checkpoint_path, snapshot_directory = _configure_output(args)
    reward_head_path = (
        None
        if args.reward_head_checkpoint is None
        else Path(args.reward_head_checkpoint).expanduser().resolve()
    )
    initialization_actor_path = (
        None
        if args.init_actor_checkpoint is None
        else Path(args.init_actor_checkpoint).expanduser().resolve()
    )
    if not args.resume:
        atomic_json_dump(vars(args), task_directory / "run_config.json")
    _configure_logging(task_directory)
    print("=" * 60)
    print("DIRECT VISUAL-REWARD CROSSQ TRAINING")
    print("=" * 60)
    for argument, value in vars(args).items():
        print(f"{argument}: {value}")
    print("=" * 60)

    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    need_images = not args.use_env_rewards
    env = gym.make(
        args.task,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array" if need_images else None,
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
        reward_mode="normalized_dense",
    )
    observation, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    state_dim = int(observation.shape[-1])
    action_dim = int(env.action_space.shape[-1])
    max_action = float(env.action_space.high.reshape(-1)[0])

    predictor: Optional[QwenRewardHeadPredictor]
    if reward_head_path is not None:
        reward_dtype = (
            torch.float32
            if device.type == "cpu"
            else torch.bfloat16
            if args.bf16
            else torch.float16
        )
        predictor = QwenRewardHeadPredictor(
            checkpoint_path=str(reward_head_path),
            device=device,
            dtype=reward_dtype,
        )
        if predictor.task != args.task:
            raise ValueError(
                f"Reward head task {predictor.task} does not match RL task {args.task}"
            )
        print(f"Using direct visual reward head: {reward_head_path}")
        print(
            "Replay reward source: frozen adapted visual encoder + attention pooler + "
            "reward head. Native dense reward is diagnostic-only."
        )
        if args.env_success_override:
            print("ManiSkill success overrides successful transition rewards to 1.0.")
    else:
        predictor = None
        print("Using native ManiSkill normalized-dense rewards for bootstrap training.")

    warmup_steps = (
        args.batch_renorm_warmup_steps
        if args.batch_renorm_warmup_steps is not None
        else int(args.max_steps * 0.04)
    )
    actor = PolicyNetwork(
        state_dim, action_dim, max_action, hidden_dim=args.actor_hidden_dim
    ).to(device)
    critic = QNetwork(
        state_dim,
        action_dim,
        hidden_dim=args.critic_hidden_dim,
        warmup_steps=warmup_steps,
    ).to(device)
    reference_actor, initial_log_alpha = _load_initial_actor(
        actor,
        args,
        state_dim,
        action_dim,
        max_action,
        initialization_actor_path,
    )

    resume_checkpoint: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_checkpoint = load_training_checkpoint(latest_checkpoint_path, args.task)
        if resume_checkpoint["state_dim"] != state_dim or resume_checkpoint["action_dim"] != action_dim:
            raise ValueError("Resume checkpoint dimensions do not match the environment")
        if resume_checkpoint["effective_batch_renorm_warmup_steps"] != warmup_steps:
            raise ValueError(
                "Effective BatchRenorm warmup changed on resume: "
                f"saved={resume_checkpoint['effective_batch_renorm_warmup_steps']}, "
                f"requested={warmup_steps}. Pass the original explicit value."
            )
        validate_resume_arguments(
            resume_checkpoint["args"], vars(args), _resume_immutable_arguments()
        )
        # Preserve the original run_config.json and record the validated resume
        # request separately.
        atomic_json_dump(vars(args), task_directory / "resume_config.json")
        actor.load_state_dict(resume_checkpoint["actor_state_dict"], strict=True)
        critic.load_state_dict(resume_checkpoint["critic_state_dict"], strict=True)
        saved_reference = resume_checkpoint["reference_actor_state_dict"]
        if saved_reference is not None:
            reference_actor = PolicyNetwork(
                state_dim, action_dim, max_action, hidden_dim=args.actor_hidden_dim
            ).to(device)
            reference_actor.load_state_dict(saved_reference, strict=True)
            reference_actor.eval()
            for parameter in reference_actor.parameters():
                parameter.requires_grad = False
        elif args.reference_kl_weight > 0.0:
            raise ValueError("Anchored resume checkpoint has no frozen reference actor")

    target_entropy = (
        args.target_entropy if args.target_entropy is not None else -action_dim
    )
    updater = CrossQUpdater(
        actor=actor,
        critic=critic,
        reference_actor=reference_actor,
        device=device,
        batch_size=args.batch_size,
        gamma=args.gamma,
        bootstrap_at_done=args.bootstrap_at_done,
        policy_delay=args.policy_delay,
        critic_only_warmup_updates=args.critic_only_warmup_updates,
        reference_kl_weight=args.reference_kl_weight,
        actor_max_grad_norm=args.actor_max_grad_norm,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        fixed_alpha=args.alpha,
        target_entropy=None if args.alpha is not None else target_entropy,
        initial_log_alpha=initial_log_alpha,
    )
    replay_buffer = SimpleReplayBuffer(args.buffer_size)
    observation, _ = env.reset()
    step_state = get_step_state(
        env, observation, args.task, args.moving_mounted_camera, need_images
    )
    current_episode: List[Dict[str, Any]] = []
    unannotated_episodes: List[List[Dict[str, Any]]] = []
    progress = TrainingProgress()
    initialization_checkpoint_path = args.init_actor_checkpoint

    if resume_checkpoint is not None:
        updater.actor_optimizer.load_state_dict(
            resume_checkpoint["actor_optimizer_state_dict"]
        )
        updater.critic_optimizer.load_state_dict(
            resume_checkpoint["critic_optimizer_state_dict"]
        )
        updater.restore_metrics(resume_checkpoint)
        updater.restore_temperature(resume_checkpoint)
        replay_buffer.extend(resume_checkpoint["buffer"])
        unannotated_episodes = resume_checkpoint["unannotated_episodes"]
        progress = TrainingProgress.from_checkpoint(resume_checkpoint)
        initialization_checkpoint_path = resume_checkpoint[
            "initialization_checkpoint_path"
        ]
        random.setstate(resume_checkpoint["python_rng_state"])
        np.random.set_state(resume_checkpoint["numpy_rng_state"])
        torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_checkpoint["cuda_rng_state"]]
            )
        print(
            f"Resumed episode-boundary state from {latest_checkpoint_path} "
            f"at step {progress.global_step}"
        )
        print(
            "Resume note: ManiSkill environment/action-space RNG state is restarted; "
            "continuation is safe but not trajectory-identical to an uninterrupted run."
        )
        del resume_checkpoint

    def rolling_checkpoint() -> Dict[str, Any]:
        return build_training_checkpoint(
            args=args,
            actor=actor,
            critic=critic,
            reference_actor=reference_actor,
            updater=updater,
            replay_buffer=replay_buffer,
            progress=progress,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            batch_renorm_warmup_steps=warmup_steps,
            current_episode=current_episode,
            unannotated_episodes=unannotated_episodes,
            initialization_checkpoint_path=initialization_checkpoint_path,
        )

    if args.init_actor_checkpoint is not None:
        initial_result = evaluate_actor(
            actor,
            args.task,
            device,
            args.initial_eval_episodes,
            base_seed=args.eval_seed,
        )
        progress.latest_success_rate = initial_result.success_rate
        progress.latest_evaluation_return = (
            initial_result.average_normalized_dense_return
        )
        progress.best_success_rate = initial_result.success_rate
        progress.best_evaluation_return = (
            initial_result.average_normalized_dense_return
        )
        progress.append_evaluation(
            0,
            initial_result.success_rate,
            initial_result.average_normalized_dense_return,
        )
        print(
            f"Initial policy evaluation | success={initial_result.success_rate * 100:.1f}% | "
            "average_normalized_dense_return="
            f"{initial_result.average_normalized_dense_return:.2f}"
        )
        atomic_torch_save(rolling_checkpoint(), latest_checkpoint_path)
        initial_actor = build_actor_export(
            actor,
            updater.log_alpha,
            args,
            state_dim,
            action_dim,
            max_action,
            0,
            initial_result.success_rate,
            initial_result.average_normalized_dense_return,
        )
        atomic_torch_save(initial_actor, task_directory / "initial_actor_export.pth")
        atomic_torch_save(initial_actor, task_directory / "best_actor_export.pth")
        atomic_torch_save(initial_actor, task_directory / "best_return_actor_export.pth")
        _write_evaluations(task_directory, progress)
        if initial_result.success_rate < args.minimum_initial_success_rate:
            env.close()
            raise RuntimeError(
                f"Initial policy success {initial_result.success_rate:.3f} is below "
                f"required {args.minimum_initial_success_rate:.3f}"
            )

    random_action_momentum = env.action_space.sample()
    target_action = env.action_space.sample()
    termination_reason = "maximum environment steps reached"
    stop_training = False
    try:
        while progress.global_step < args.max_steps and not stop_training:
            if progress.global_step < args.random_steps:
                if progress.global_step % 10 == 0:
                    target_action = env.action_space.sample()
                random_action_momentum = (
                    0.90 * random_action_momentum + 0.10 * target_action
                )
                action = np.clip(
                    random_action_momentum,
                    env.action_space.low,
                    env.action_space.high,
                )
            else:
                state_tensor = torch.as_tensor(
                    state_to_numpy(step_state["state"]),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    if progress.global_step < args.deterministic_prefill_steps:
                        mean, _ = actor(state_tensor)
                        action_tensor = torch.tanh(mean) * actor.max_action
                    else:
                        action_tensor, _ = actor.sample(state_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()

            next_observation, environment_reward, terminated, truncated, info = env.step(action)
            if isinstance(environment_reward, torch.Tensor):
                environment_reward = float(environment_reward.detach().item())
            else:
                environment_reward = float(environment_reward)
            terminated_bool = success_to_bool(terminated)
            truncated_bool = success_to_bool(truncated)
            episode_done = terminated_bool or truncated_bool
            success = success_to_bool(info["success"])
            next_step_state = get_step_state(
                env,
                next_observation,
                args.task,
                args.moving_mounted_camera,
                need_images,
            )
            current_episode.append(
                {
                    "state": state_to_numpy(step_state["state"]),
                    "image": step_state["image"],
                    "action": action,
                    "next_state": state_to_numpy(next_step_state["state"]),
                    "next_image": next_step_state["image"],
                    # Preserve the original timeout semantics: only termination is
                    # stored in replay, while either flag ends collection.
                    "done": terminated_bool,
                    "task_reward": environment_reward,
                    "success": success,
                }
            )
            step_state = next_step_state
            progress.global_step += 1

            if not episode_done:
                continue

            progress.total_episodes_completed += 1
            unannotated_episodes.append(current_episode)
            current_episode = []
            observation, _ = env.reset()
            step_state = get_step_state(
                env,
                observation,
                args.task,
                args.moving_mounted_camera,
                need_images,
            )
            if progress.global_step < args.random_steps:
                random_action_momentum = env.action_space.sample()
                target_action = env.action_space.sample()

            if len(unannotated_episodes) >= args.vlm_batch_size:
                if args.use_env_rewards:
                    transitions = annotate_native_reward_episodes(
                        unannotated_episodes, args.vlm_reward_scale
                    )
                else:
                    if predictor is None:
                        raise RuntimeError("Visual reward predictor is not initialized")
                    print(
                        f"Annotating {len(unannotated_episodes)} episode(s) with "
                        f"visual reward head at step {progress.global_step}..."
                    )
                    transitions, _ = annotate_visual_reward_episodes(
                        episodes=unannotated_episodes,
                        predictor=predictor,
                        batch_size=args.reward_head_batch_size,
                        reward_scale=args.vlm_reward_scale,
                        env_success_override=args.env_success_override,
                    )
                for transition in transitions:
                    replay_buffer.add(*transition)
                previous_update_count = updater.metrics.total_updates
                minimum_replay_size = max(args.batch_size, args.learning_starts)
                if len(replay_buffer) >= minimum_replay_size:
                    updater.update(replay_buffer, len(transitions) * args.utd_ratio)
                    if (
                        updater.metrics.total_updates // args.learning_starts
                        > previous_update_count // args.learning_starts
                    ):
                        metrics = updater.metrics
                        if metrics.actor_updates_started:
                            print(
                                f"Update {metrics.total_updates} | Buffer {len(replay_buffer)} | "
                                f"Alpha {metrics.alpha:.5f} | "
                                f"Avg batch reward {metrics.average_batch_reward:.3f} | "
                                f"SAC actor loss {metrics.sac_actor_loss:.4f} | "
                                f"Reference KL {metrics.reference_kl:.6f} | "
                                f"Total actor loss {metrics.total_actor_loss:.4f} | "
                                "Reference action MSE "
                                f"{metrics.reference_action_mse:.6f}"
                            )
                        else:
                            print(
                                f"Update {metrics.total_updates} | Buffer {len(replay_buffer)} | "
                                f"Alpha {metrics.alpha:.5f} | "
                                f"Avg batch reward {metrics.average_batch_reward:.3f} | "
                                "Critic-only warmup active"
                            )
                unannotated_episodes = []

            if (
                progress.global_step - progress.last_evaluation_step >= args.eval_freq
                and progress.global_step >= args.learning_starts
            ):
                print(f"\n--- Evaluation at {progress.global_step} steps ---")
                result = evaluate_actor(
                    actor,
                    args.task,
                    device,
                    args.eval_episodes,
                    base_seed=args.eval_seed,
                )
                print(
                    f"Evaluation success rate: {result.success_rate * 100:.1f}% | "
                    "Average normalized dense return: "
                    f"{result.average_normalized_dense_return:.2f}"
                )
                success_improved = result.success_rate > progress.best_success_rate
                return_improved = (
                    result.average_normalized_dense_return
                    > progress.best_evaluation_return
                )
                progress.best_success_rate = max(
                    progress.best_success_rate, result.success_rate
                )
                progress.best_evaluation_return = max(
                    progress.best_evaluation_return,
                    result.average_normalized_dense_return,
                )
                progress.latest_success_rate = result.success_rate
                progress.latest_evaluation_return = (
                    result.average_normalized_dense_return
                )
                progress.last_evaluation_step = progress.global_step
                progress.append_evaluation(
                    progress.global_step,
                    result.success_rate,
                    result.average_normalized_dense_return,
                )
                if result.success_rate >= args.target_success_rate:
                    progress.consecutive_target_evaluations += 1
                else:
                    progress.consecutive_target_evaluations = 0

                atomic_torch_save(rolling_checkpoint(), latest_checkpoint_path)
                actor_export = build_actor_export(
                    actor,
                    updater.log_alpha,
                    args,
                    state_dim,
                    action_dim,
                    max_action,
                    progress.global_step,
                    result.success_rate,
                    result.average_normalized_dense_return,
                )
                if snapshot_directory is not None:
                    success_per_mille = int(round(result.success_rate * 1000.0))
                    atomic_torch_save(
                        actor_export,
                        snapshot_directory
                        / (
                            f"actor_step_{progress.global_step:07d}_"
                            f"success_{success_per_mille:04d}.pth"
                        ),
                    )
                if success_improved:
                    atomic_torch_save(
                        actor_export, task_directory / "best_actor_export.pth"
                    )
                if return_improved:
                    atomic_torch_save(
                        actor_export,
                        task_directory / "best_return_actor_export.pth",
                    )
                _write_evaluations(task_directory, progress)

                if (
                    args.bootstrap_target_success_rate is not None
                    and result.success_rate >= args.bootstrap_target_success_rate
                ):
                    atomic_torch_save(
                        actor_export, task_directory / "bootstrap_actor.pth"
                    )
                    termination_reason = (
                        "native bootstrap target reached: "
                        f"{result.success_rate:.3f} >= "
                        f"{args.bootstrap_target_success_rate:.3f}"
                    )
                    stop_training = True
                elif (
                    progress.global_step >= args.minimum_steps_before_convergence
                    and progress.consecutive_target_evaluations >= 2
                ):
                    termination_reason = (
                        "target success reached in two consecutive evaluations"
                    )
                    stop_training = True
    except Exception as error:
        logging.exception("Training failed: %s", error)
        env.close()
        raise

    final_actor = build_actor_export(
        actor,
        updater.log_alpha,
        args,
        state_dim,
        action_dim,
        max_action,
        progress.global_step,
        progress.latest_success_rate,
        progress.latest_evaluation_return,
    )
    atomic_torch_save(final_actor, task_directory / "final_actor_export.pth")
    if not current_episode:
        atomic_torch_save(rolling_checkpoint(), latest_checkpoint_path)
    else:
        if latest_checkpoint_path.exists():
            print(
                "Final step occurred mid-episode; latest_checkpoint.pth remains at "
                "the preceding safe episode-boundary evaluation."
            )
        else:
            print(
                "Final step occurred before the first safe episode boundary; no "
                "resumable full-state checkpoint was written."
            )
    _write_evaluations(task_directory, progress)
    atomic_json_dump(
        {
            "termination_reason": termination_reason,
            "global_step": progress.global_step,
            "replay_transitions": len(replay_buffer),
            "pending_completed_episodes": len(unannotated_episodes),
            "pending_completed_transitions": sum(
                len(episode) for episode in unannotated_episodes
            ),
            "partial_episode_transitions": len(current_episode),
            "total_episodes_completed": progress.total_episodes_completed,
            "total_training_updates": updater.metrics.total_updates,
            "best_success_rate": progress.best_success_rate,
            "best_average_normalized_dense_return": (
                progress.best_evaluation_return
                if np.isfinite(progress.best_evaluation_return)
                else None
            ),
            "latest_success_rate": progress.latest_success_rate,
            "latest_average_normalized_dense_return": (
                progress.latest_evaluation_return
                if np.isfinite(progress.latest_evaluation_return)
                else None
            ),
        },
        task_directory / "training_summary.json",
    )
    env.close()
    print(f"Training finished: {termination_reason}.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
