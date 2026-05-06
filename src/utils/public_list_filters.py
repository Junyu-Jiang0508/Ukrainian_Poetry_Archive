"""Shared helpers for public-list author and translation filters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PUBLIC_INCLUDE_COL = "Include in public list"
DEFAULT_ALIAS_COLUMNS = [
    "Author",
    "Facebook name",
    "Alias (English transliteration)",
    "Alias (Russian)",
    "Alias (Ukrainian)",
]


def normalize_author_name(value: object) -> str:
    """Normalize whitespace for author names used in matching."""
    return " ".join(str(value).strip().split())


def is_public_yes(value: object) -> bool:
    """Return True when a spreadsheet value marks public inclusion."""
    if pd.isna(value):
        return False
    normalized_value = str(value).strip().lower()
    return normalized_value in ("yes", "y", "1", "true")


def has_translation_marker(value: object) -> bool:
    """Return True when a translation-related cell is filled with text."""
    if pd.isna(value):
        return False
    normalized_value = str(value).strip()
    return bool(normalized_value and normalized_value.lower() != "nan")


def load_allowed_author_names(
    path: Path,
    *,
    alias_columns: list[str] | None = None,
) -> set[str]:
    """Build author allow-list from ``author.xlsx`` where include-column is yes."""
    author_df = pd.read_excel(path)
    if PUBLIC_INCLUDE_COL not in author_df.columns:
        raise KeyError(
            f"{path}: missing column {PUBLIC_INCLUDE_COL!r} "
            f"(have {list(author_df.columns)})"
        )
    alias_columns = alias_columns or DEFAULT_ALIAS_COLUMNS
    public_rows = author_df[author_df[PUBLIC_INCLUDE_COL].map(is_public_yes)]
    allowed_names: set[str] = set()
    for column_name in alias_columns:
        if column_name not in public_rows.columns:
            continue
        for raw_name in public_rows[column_name].dropna().astype(str):
            normalized_name = normalize_author_name(raw_name)
            if normalized_name and normalized_name.lower() != "nan":
                allowed_names.add(normalized_name)
    return allowed_names
