"""
Fail-fast experiment output-directory handling.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Collection


def prepare_run_directory(
    output_dir: Path,
    *,
    overwrite: bool,
    allowed_existing_names: Collection[str] = (),
) -> None:
    """
    Create a clean run directory or retain only explicitly allowed caches.
    """
    path = output_dir.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.is_dir():
        contents = list(path.iterdir())
        unexpected = [item for item in contents if item.name not in allowed_existing_names]
        if unexpected and not overwrite:
            raise FileExistsError(
                f"Output directory contains prior run artifacts: {unexpected}. "
                "Choose a new directory or pass --overwrite."
            )
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
