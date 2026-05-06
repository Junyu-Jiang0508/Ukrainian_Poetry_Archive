"""Poem-level language strata: Ukrainian-only, Russian-only, pooled (Ukrainian ∪ Russian)."""

from __future__ import annotations

import pandas as pd

# Exact `language` field matches from annotation CSV (see MAJOR_LANGUAGES elsewhere).
POOL_CODES = frozenset({"Ukrainian", "Russian"})

# Ordered reporting: pooled first, then single-language strata.
LANGUAGE_STRATA = ("pooled_Ukrainian_Russian", "Ukrainian", "Russian")


def filter_poems_by_language_stratum(df: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Keep poem rows whose `language_clean` falls in the stratum (exact labels)."""
    if stratum == "pooled_Ukrainian_Russian":
        return df.loc[df["language_clean"].isin(POOL_CODES)].copy()
    if stratum == "Ukrainian":
        return df.loc[df["language_clean"].eq("Ukrainian")].copy()
    if stratum == "Russian":
        return df.loc[df["language_clean"].eq("Russian")].copy()
    raise ValueError(f"Unknown language stratum: {stratum!r}; use one of {LANGUAGE_STRATA}")
