"""Shared person × number encodings for UD-style GPT annotation rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_annotation_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def pronoun_class_sixway_column(df: pd.DataFrame) -> pd.Series:
    """Vector map of ``person`` × ``number`` to ``{1sg,1pl,2sg,2pl,3sg,3pl}``."""
    if "person" not in df.columns or "number" not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    person = df["person"].fillna("").astype(str).str.strip()
    number = df["number"].fillna("").astype(str).str.strip()
    has_sing = number.str.contains("Sing", na=False, regex=False)
    has_plur = number.str.contains("Plur", na=False, regex=False)
    p1 = person.eq("1st")
    p2 = person.eq("2nd")
    p3 = person.eq("3rd")
    conds = [p1 & has_sing, p1 & has_plur, p2 & has_sing, p2 & has_plur, p3 & has_sing, p3 & has_plur]
    labels = ("1sg", "1pl", "2sg", "2pl", "3sg", "3pl")
    return pd.Series(np.select(conds, labels, default=""), index=df.index, dtype=object)


def pronoun_class_sixway(row: pd.Series) -> str | None:
    """Map person (1st/2nd/3rd) + number (Singular/Plural) to {1sg,1pl,2sg,2pl,3sg,3pl}."""
    person = normalize_annotation_str(row.get("person"))
    number = normalize_annotation_str(row.get("number"))
    if person == "1st" and "Sing" in number:
        return "1sg"
    if person == "1st" and "Plur" in number:
        return "1pl"
    if person == "2nd" and "Sing" in number:
        return "2sg"
    if person == "2nd" and "Plur" in number:
        return "2pl"
    if person == "3rd" and "Sing" in number:
        return "3sg"
    if person == "3rd" and "Plur" in number:
        return "3pl"
    return None


def ordered_index_for_crosstab(
    index: pd.Index,
    preferred: list[str] | None = None,
) -> list[str]:
    values = [str(x) for x in index]
    if preferred is None:
        return values
    head = [x for x in preferred if x in values]
    tail = [x for x in values if x not in head]
    return head + tail
