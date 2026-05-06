"""Normalize coarse person/number labels for validation and agreement checks."""

from __future__ import annotations

import pandas as pd


def normalize_person_number_label(value) -> str:
    """Map shorthand labels to full UD-style strings used in evaluation."""
    s = str(value).strip() if pd.notna(value) else ""
    if s in ("Sing", "Sg"):
        return "Singular"
    if s in ("Plur", "Pl"):
        return "Plural"
    return s
