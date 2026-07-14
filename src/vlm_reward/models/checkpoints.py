"""
Portable paths for model bundles and backward-compatible artifact loading.
"""
import os
from pathlib import Path
from typing import List


def relative_artifact_path(artifact_path: Path, checkpoint_path: Path) -> str:
    """
    Encode an artifact path relative to the checkpoint that references it.
    """
    artifact = artifact_path.expanduser().resolve()
    checkpoint_parent = checkpoint_path.expanduser().resolve().parent
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    return Path(os.path.relpath(artifact, start=checkpoint_parent)).as_posix()


def resolve_checkpoint_artifact(checkpoint_path: Path, stored_path: str) -> Path:
    """Resolve new relative paths and absolute paths embedded by historical runs.

    Historical VastAI checkpoints contain absolute ``/workspace/...`` paths. If
    that path no longer exists, matching path anchors (for example
    ``finetuning_output`` or the run directory name) are mapped onto the current
    checkpoint location.
    """

    checkpoint = checkpoint_path.expanduser().resolve()
    encoded = Path(stored_path).expanduser()
    candidates: List[Path] = []

    if not encoded.is_absolute():
        candidates.append(checkpoint.parent / encoded)
    else:
        candidates.append(encoded)
        encoded_parts = encoded.parts
        for ancestor in (checkpoint.parent, *checkpoint.parents):
            for part_index, part in enumerate(encoded_parts):
                if ancestor.name == part:
                    candidates.append(ancestor.joinpath(*encoded_parts[part_index + 1 :]))
        candidates.append(checkpoint.parent / encoded.name)

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(normalized)
        if normalized.exists():
            return normalized

    attempted = "\n  ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(
        f"Could not resolve checkpoint artifact {stored_path!r} referenced by "
        f"{checkpoint}. Tried:\n  {attempted}"
    )
