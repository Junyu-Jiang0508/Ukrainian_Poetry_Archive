"""Load GPT annotation CSVs with derived columns and standard cohort filters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.annotation_derived_columns import (
    CORE_TEMPORAL_PERIODS,
    add_derived_annotation_columns,
)
from utils.csv_io import read_annotation_csv


def load_gpt_annotation_derived(path: Path) -> pd.DataFrame:
    """Read detailed annotation CSV and attach derived QA / temporal columns."""
    df = read_annotation_csv(path)
    return add_derived_annotation_columns(df)


def load_core_temporal_cohort(
    path: Path,
    *,
    qa_clean: bool,
    with_period_column: bool = True,
    author_keys: bool = False,
) -> pd.DataFrame:
    """
    Restrict to core temporal periods (2014_2021, post_2022), optional QA-clean rows,
    optional _period mirror column, optional author normalization columns for heterogeneity.
    """
    df = load_gpt_annotation_derived(path)
    df = df[df["temporal_period_reconciled"].isin(CORE_TEMPORAL_PERIODS)].copy()
    if qa_clean:
        df = df[df["qa_clean"]].copy()
    if with_period_column:
        df = df.copy()
        df["_period"] = df["temporal_period_reconciled"]
    if author_keys:
        df = df.copy()
        if "author" not in df.columns:
            df["author"] = ""

        def _norm_author(v: object) -> str:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            return str(v).strip()

        df["_author"] = df["author"].map(_norm_author)
        df["_author_key"] = df["_author"].str.lower()
    return df
