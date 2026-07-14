"""
Content fingerprints for expensive frozen-embedding caches.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


EMBEDDING_CACHE_FORMAT_VERSION = 3
EMBEDDING_PREPROCESSING_VERSION = "qwen-image-processor-and-attention-pool-v1"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Hash one file without loading it entirely into memory.
    """
    file_path = path.expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path, ignored_names: Iterable[str] = (".DS_Store",)) -> str:
    """
    Hash relative names and contents of every file in a model directory.
    """
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ignored = set(ignored_names)
    files = sorted(
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file() and candidate.name not in ignored
    )
    if not files:
        raise ValueError(f"Cannot fingerprint empty directory: {directory}")

    digest = hashlib.sha256()
    for file_path in files:
        relative_name = file_path.relative_to(directory).as_posix()
        digest.update(relative_name.encode("utf-8"))
        digest.update(file_sha256(file_path).encode("ascii"))
    return digest.hexdigest()


def referenced_files_sha256(paths: Iterable[Path]) -> str:
    """
    Hash referenced files in dataset order so image changes invalidate caches.
    """
    digest = hashlib.sha256()
    file_count = 0
    for file_count, file_path in enumerate(paths, start=1):
        resolved = file_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        digest.update(str(file_count).encode("ascii"))
        digest.update(file_sha256(resolved).encode("ascii"))
    if file_count == 0:
        raise ValueError("Cannot fingerprint an empty referenced-file sequence")
    return digest.hexdigest()


def adapter_sha256(path: Path) -> str:
    """Hash only files that affect frozen adapter inference.

    Hugging Face output directories can also contain multi-gigabyte optimizer
    checkpoints. Those do not affect embeddings and therefore must not enter the
    cache identity.
    """

    directory = path.expanduser().resolve()
    required = [directory / "adapter_config.json", directory / "contrastive_extras.pt"]
    model_candidates = [
        directory / "adapter_model.safetensors",
        directory / "adapter_model.bin",
    ]
    for required_path in required:
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    model_files = [candidate for candidate in model_candidates if candidate.is_file()]
    if len(model_files) != 1:
        raise ValueError(
            f"Expected exactly one adapter model file in {directory}, found {model_files}"
        )

    digest = hashlib.sha256()
    for file_path in sorted(required + model_files):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(file_sha256(file_path).encode("ascii"))
    return digest.hexdigest()


def embedding_cache_fingerprint(
    *,
    dataset_jsonl: Path,
    model_id: str,
    adapter_dir: Path,
    referenced_files_fingerprint: str,
    dataset_options: Mapping[str, Any],
    adapter_fingerprint: Optional[str] = None,
) -> str:
    """
    Fingerprint all inputs that determine frozen visual embeddings.
    """
    payload = {
        "format_version": EMBEDDING_CACHE_FORMAT_VERSION,
        "preprocessing_version": EMBEDDING_PREPROCESSING_VERSION,
        "dataset_sha256": file_sha256(dataset_jsonl),
        "model_id": model_id,
        "referenced_files_sha256": referenced_files_fingerprint,
        "adapter_sha256": (
            directory_sha256(adapter_dir)
            if adapter_fingerprint is None
            else adapter_fingerprint
        ),
        "dataset_options": dict(dataset_options),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_embedding_cache(
    cache: Dict[str, Any],
    *,
    expected_fingerprint: str,
    expected_examples: int,
) -> None:
    """
    Fail fast when a cache predates or does not match the requested run.
    """
    for key in ("embeddings", "rewards", "views", "fingerprint"):
        if key not in cache:
            raise ValueError(
                f"Embedding cache is missing {key!r}; rebuild it with --rebuild-cache"
            )
    cached_examples = int(cache["embeddings"].shape[0])
    if cached_examples != expected_examples:
        raise ValueError(
            f"Cached example count {cached_examples} does not match dataset length "
            f"{expected_examples}; rebuild it with --rebuild-cache"
        )
    if int(cache["rewards"].shape[0]) != expected_examples:
        raise ValueError("Cached reward count does not match cached embeddings")
    if len(cache["views"]) != expected_examples:
        raise ValueError("Cached view count does not match cached embeddings")
    if cache["fingerprint"] != expected_fingerprint:
        raise ValueError(
            "Embedding cache fingerprint does not match the dataset/model/adapter; "
            "rebuild it with --rebuild-cache"
        )
