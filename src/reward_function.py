import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Read settings from .env file
load_dotenv()

# Hide Hugging Face cache/download progress bars during normal reward calls
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

MODEL_NAME = os.getenv("MODEL_NAME", "mlx-community/Qwen3-VL-8B-Instruct-4bit")

# Maximum number of text tokens the model is allowed to generate for the answer
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "80"))

# 0.0 means deterministic decoding: the model picks the most likely next token.
# This is useful for reward functions because repeated calls should be stable.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# MLX processes the prompt in chunks before generating the answer. Smaller chunks
# use less peak memory, which is helpful on laptops, but can be slower.
MLX_PREFILL_STEP_SIZE = int(os.getenv("MLX_PREFILL_STEP_SIZE", "512"))

# Some Hugging Face models require custom Python code from the model repository.
# We keep this off by default; enable it in .env only if a chosen model asks for it.
MLX_TRUST_REMOTE_CODE = os.getenv("MLX_TRUST_REMOTE_CODE", "0").lower() in {
    "1",
    "true",
    "yes",
}

# These globals cache the loaded model and processor after the first call.
# Loading a local LLM is expensive, so we do it once and reuse it for later
# reward queries in the same Python process.
_MLX_MODEL = None
_MLX_PROCESSOR = None


def _build_prompt(task_description):
    """
    Build the instruction given to Qwen3-VL.
    """
    return (
        f"Task: {task_description}\n"
        "You are a reward model for robotic reinforcement learning.\n"
        "You will receive exactly two images from the same rollout:\n"
        "Image 1 is the earlier reference frame.\n"
        "Image 2 is the later/current frame.\n"
        "Score how much progress Image 2 shows toward the task compared with Image 1.\n"
        "Use 0.0 for no progress, negative progress, or a clearly failed state.\n"
        "Use 10.0 only when the task appears completed.\n"
        "Intermediate scores should reflect partial progress toward the goal.\n"
        "Return one short sentence, then a final line in exactly this format:\n"
        "SCORE: <number from 0.0 to 10.0>"
    )


def _extract_score(text):
    """
    Convert the model's text answer into a Python float.

    The prompt asks for a line like "SCORE: 6.5". We parse that first. If the
    model ignores the exact format, the fallback takes the last number in the
    answer, which is usually still the score.
    """
    score_match = re.search(r"score\s*:\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
    if score_match:
        return _clamp_score(float(score_match.group(1)))

    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not matches:
        raise ValueError("No numeric score found in model output")
    return _clamp_score(float(matches[-1]))


def _clamp_score(score):
    # Keep malformed model outputs from producing rewards outside the rubric.
    return max(0.0, min(10.0, score))


def _load_mlx_model():
    """
    Load Qwen3-VL with mlx-vlm.

    In MLX-VLM, "model" is the neural network weights and architecture, while
    "processor" handles the non-obvious input/output details: chat templates,
    tokenization, image preprocessing, and detokenization.
    """
    global _MLX_MODEL, _MLX_PROCESSOR
    if _MLX_MODEL is None:
        try:
            from huggingface_hub.utils import disable_progress_bars
            from mlx_vlm import load
        except Exception as exc:
            raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm") from exc

        # "mlx_vlm.load" checks the Hugging Face cache before loading local
        # weights. This keeps that cache check quiet once the model is already
        # downloaded.
        disable_progress_bars()

        _MLX_MODEL, _MLX_PROCESSOR = load(
            MODEL_NAME,
            trust_remote_code = MLX_TRUST_REMOTE_CODE,
        )
    return _MLX_MODEL, _MLX_PROCESSOR


def _build_mlx_prompt(prompt, processor, model):
    """
    Wrap our plain instruction in the model's expected chat format.

    Different VLMs expect different special tokens around text and images.
    "apply_chat_template" inserts those tokens for this specific model. The
    "num_images = 2" argument is important because Qwen3-VL needs placeholders for
    both the earlier and later frame.
    """
    try:
        from mlx_vlm import apply_chat_template
    except Exception:
        return prompt

    return apply_chat_template(
        processor,
        model.config,
        prompt,
        num_images = 2,
    )


def _mlx_generate(prompt, frame_paths):
    """
    Run local Qwen3-VL inference on two image files.

    "generate" receives the formatted prompt and the local image paths. MLX-VLM
    loads the images, preprocesses them, runs the model locally on the Mac, and
    returns generated text. No API server is involved here.
    """
    from mlx_vlm import generate

    model, processor = _load_mlx_model()
    formatted_prompt = _build_mlx_prompt(prompt, processor, model)
    result = generate(
        model,
        processor,
        formatted_prompt,
        image = [str(path) for path in frame_paths],
        max_tokens = MAX_TOKENS,
        temperature = TEMPERATURE,
        prefill_step_size = MLX_PREFILL_STEP_SIZE,
        verbose = False,
    )

    return getattr(result, "text", str(result)).strip()


def get_reward_score(frame1_path, frame2_path, task_description):
    """
    Compute a pairwise progress reward from two rollout frames.

    Args:
        frame1_path: Earlier framefrom the rollout.
        frame2_path: Later/current frame from the rollout.
        task_description: Natural-language task goal, e.g. "open the drawer".

    Returns:
        A tuple `(content, score)`, where `content` is the raw explanation from
        the VLM and `score` is a float in the range 0.0 to 10.0.
    """

    prompt = _build_prompt(task_description)
    content = _mlx_generate(prompt, [frame1_path, frame2_path])

    try:
        score = _extract_score(content)
        return content, score
    except Exception as e:
        # Reward code should fail softly during data collection/training.
        # Returning 0.0 is conservative if the VLM response cannot be parsed.
        print(f"Error: {e}")
        print(f"Full response data: {content}")
        return None, 0.0


if __name__ == "__main__":
    # Minimal local smoke test. The first run may take a while because MLX-VLM
    # downloads the model from Hugging Face and loads it into memory.
    repo_root = Path(__file__).resolve().parent.parent
    frame1 = repo_root / "data" / "metaworld" / "drawer-open-v3" / "expert" / "rollout_0" / "frame_000.jpg"
    frame2 = repo_root / "data" / "metaworld" / "drawer-open-v3" / "expert" / "rollout_0" / "frame_030.jpg"
    task = "Reach the drawer handle and open the drawer fully."
    content, score = get_reward_score(str(frame1), str(frame2), task)
    print(f"Reward Score: {score}")
    print(f"Explanation: {content}")
