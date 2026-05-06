"""Shared filesystem and CSV helpers for numbered pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.workspace import repository_root


def stage_output_dir(stage_name: str, *, root: Path | None = None) -> Path:
    """Return and create ``<repo>/outputs/<stage_name>``."""
    repo = root if root is not None else repository_root()
    output_dir = repo / "outputs" / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_csv_artifact(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV artifact with ``low_memory=False`` by default."""
    read_kwargs: dict[str, Any] = {"low_memory": False}
    read_kwargs.update(kwargs)
    return pd.read_csv(Path(path), **read_kwargs)


def write_csv_artifact(
    df: pd.DataFrame,
    path: Path | str,
    *,
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> None:
    """Write a CSV artifact with parent directory creation."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=index, encoding=encoding, **kwargs)
