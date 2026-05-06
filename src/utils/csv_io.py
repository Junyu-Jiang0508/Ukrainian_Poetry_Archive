"""Shared ``pandas.read_csv`` defaults for large, occasionally malformed exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

STANDARD_ANNOTATION_READ_KW: dict[str, Any] = {
    "low_memory": False,
    "on_bad_lines": "skip",
}


def read_annotation_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Merge ``STANDARD_ANNOTATION_READ_KW`` with caller overrides (e.g. ``dtype=``)."""
    merged: dict[str, Any] = {**STANDARD_ANNOTATION_READ_KW, **kwargs}
    return pd.read_csv(path, **merged)
