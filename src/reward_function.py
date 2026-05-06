import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Read settings from .env file
load_dotenv()

# Hide Hugging Face cache/download progress bars during normal reward calls
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

MODEL_NAME = os.getenv("MODEL_NAME", "mlx-community/Qwen3-VL-8B-Instruct-4bit")

# Maximum number of text tokens the model is allowed to generate for the answer
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))

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

    Expected image order:
    - Image A: earlier frame I_{t-k}
    - Image B: later frame I_t

    The model returns a pairwise visual preference. The reward score is derived
    from that preference and the model's 1-10 confidence.
    """
    return (
        "You are evaluating progress in a robotic manipulation task.\n\n"
        f"Task goal:\n{task_description}\n\n"
        "You are given two images from the same robot rollout:\n"
        "- Image A: earlier frame\n"
        "- Image B: later frame\n\n"
        "Your task is to decide which image shows more visible progress toward the task goal.\n"
        "This is a relative visual comparison, not an absolute task-completion judgment.\n"
        "Choose Image B only if it is visibly better for the task goal than Image A.\n"
        "Choose Image A if Image B is visibly worse for the task goal than Image A.\n"
        "Choose same if there is no meaningful visible task-relevant difference.\n"
        "Choose unclear if the visual evidence is too ambiguous to judge.\n\n"
        "Consider only visually observable task progress. Ignore camera changes, lighting changes, "
        "background changes, and irrelevant robot motion.\n\n"
        "Do not infer hidden physical events from the task description alone. "
        "Do not claim contact, grasping, pulling, pushing, pressing, lifting, opening, insertion, "
        "or completion unless it is directly visible in the images.\n\n"
        "Output exactly one JSON-formatted text object with three fields: preference, confidence_score, and visible_change.\n"
        'The preference must be one of: "A", "B", "same", or "unclear".\n'
        "The confidence_score must be an integer from 1 to 10, where 1 means very uncertain and 10 means very certain.\n"
        "The visible_change must be one short sentence describing only directly visible evidence.\n"
        "Example output:\n"
        "{\n"
        '  "preference": "B",\n'
        '  "confidence_score": 6,\n'
        '  "visible_change": "The robot gripper is visibly closer to the target object in Image B."\n'
        "}\n"
        "Do not include markdown, comments, or any text outside the JSON object.\n"
    )


def _extract_score(text: str) -> float:
    """
    Convert Qwen's pairwise preference output into a signed reward.

    Image B is the later/current frame, so "B" means positive progress and "A"
    means regression. "same" and "unclear" are conservative zero rewards.
    """
    data = json.loads(text.strip())
    preference = str(data["preference"]).strip().lower()
    confidence = _clamp_confidence(float(data["confidence_score"]))

    if preference == "b":
        return confidence
    if preference == "a":
        return -confidence
    if preference in {"same", "unclear"}:
        return 0.0
    raise ValueError(f"Unexpected preference value: {data['preference']}")


def _clamp_confidence(score):
    return max(1.0, min(10.0, score))

def _load_mlx_model():
    """
    Load Qwen3-VL with mlx-vlm.
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
    Wrap the plain instruction in the model's expected chat format.
    """
    from mlx_vlm import apply_chat_template

    return apply_chat_template(
        processor,
        model.config,
        prompt,
        num_images = 2,
    )


def _mlx_generate(prompt, frame_paths):
    """
    Run local Qwen3-VL inference on two image files.
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
        frame1_path: Earlier frame from the rollout.
        frame2_path: Later/current frame from the rollout.
        task_description: Natural-language task goal, e.g. "open the drawer".

    Returns:
        A tuple `(content, score)`, where `content` is the raw explanation from
        the VLM and `score` is a float in the range -10.0 to 10.0.
    """

    prompt = _build_prompt(task_description)
    content = _mlx_generate(prompt, [frame1_path, frame2_path])
    score = _extract_score(content)

    return content, score

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    frame1 = repo_root / "data" / "metaworld" / "drawer-open-v3" / "expert" / "rollout_0" / "frame_000.jpg"
    frame2 = repo_root / "data" / "metaworld" / "drawer-open-v3" / "expert" / "rollout_0" / "frame_058.jpg"
    task = (
        "Open the green drawer by pulling the white handle outward."
    )
    content, score = get_reward_score(str(frame1), str(frame2), task)
    print(f"Reward Score: {score}")
    print(f"Explanation: {content}")
