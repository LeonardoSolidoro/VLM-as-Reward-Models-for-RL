"""
OpenAI-compatible multimodal API client for progress prediction.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import aiohttp
from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)
IMAGE_PLACEHOLDER = "[IMG]"


@dataclass(frozen=True)
class VLMAPISettings:
    """
    Connection settings for an OpenAI-compatible chat-completions endpoint.
    """

    url: str
    api_key: str
    model: str
    max_tokens: int

    @classmethod
    def from_environment(cls, max_tokens: int) -> "VLMAPISettings":
        load_dotenv()
        missing = [name for name in ("VLM_API_URL", "MODEL_NAME") if name not in os.environ]
        if missing:
            raise KeyError(f"Missing required environment variables: {missing}")
        return cls(
            url=os.environ["VLM_API_URL"],
            api_key=os.environ["VLM_API_KEY"] if "VLM_API_KEY" in os.environ else "EMPTY",
            model=os.environ["MODEL_NAME"],
            max_tokens=max_tokens,
        )


def encode_image(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_multimodal_content(prompt: str, image_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """
    Interleave text and image payloads at exact ``[IMG]`` locations.
    """

    chunks = prompt.split(IMAGE_PLACEHOLDER)
    placeholder_count = len(chunks) - 1
    if placeholder_count != len(image_paths):
        raise ValueError(
            f"Prompt contains {placeholder_count} image placeholders, "
            f"but {len(image_paths)} images were provided"
        )

    content: list[dict[str, Any]] = []
    for chunk, image_path in zip(chunks[:-1], image_paths):
        if chunk:
            content.append({"type": "text", "text": chunk})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                },
            }
        )
    if chunks[-1]:
        content.append({"type": "text", "text": chunks[-1]})
    return content


def parse_progress_scores(response_text: str, expected_frames: int) -> dict[int, float]:
    """
    Parse exactly one percentage for every one-indexed query frame.
    """

    without_thoughts = re.sub(
        r"<think>.*?</think>",
        "",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = re.findall(
        r"Frame\s+(\d+)\s*:.*?<score>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</score>",
        without_thoughts,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        raise ValueError(f"Response contains no frame-scoped <score> values:\n{response_text}")

    scores: dict[int, float] = {}
    for frame_text, score_text in matches:
        frame = int(frame_text)
        if frame in scores:
            raise ValueError(f"Response contains duplicate score for Frame {frame}")
        score = float(score_text)
        if score < 0.0 or score > 100.0:
            raise ValueError(f"Frame {frame} score is outside [0, 100]: {score}")
        scores[frame] = score

    expected = set(range(1, expected_frames + 1))
    actual = set(scores)
    if actual != expected:
        raise ValueError(
            f"Expected scores for frames {sorted(expected)}, got {sorted(actual)}"
        )
    return scores


class VLMAPIClient:
    """
    Small, strict client that returns raw text and parsed progress scores.
    """

    def __init__(self, settings: VLMAPISettings) -> None:
        self.settings = settings

    async def predict_progress(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        image_paths: Sequence[str | Path],
        expected_frames: int,
    ) -> tuple[str, dict[int, float]]:
        payload = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_multimodal_content(prompt, image_paths),
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.settings.max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }
        async with session.post(
            self.settings.url,
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            response_data = await response.json()

        try:
            response_text = response_data["choices"][0]["message"]["content"].strip()
        except (AttributeError, KeyError, IndexError, TypeError) as error:
            LOGGER.exception("Malformed chat-completions response: %r", response_data)
            raise ValueError("Malformed chat-completions response") from error
        return response_text, parse_progress_scores(response_text, expected_frames)
