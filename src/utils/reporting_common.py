"""Shared cohort prep and plotting helpers for report scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.annotation_cohort import load_core_temporal_cohort
from utils.annotation_derived_columns import CORE_TEMPORAL_PERIOD_ORDER
from utils.pronoun_encoding import ordered_index_for_crosstab


def load_core_period_cohort(
    csv_path: Path,
    *,
    qa_clean: bool,
    include_author_keys: bool = False,
) -> pd.DataFrame:
    """Load derived annotation rows restricted to core temporal periods."""
    return load_core_temporal_cohort(
        csv_path,
        qa_clean=qa_clean,
        with_period_column=True,
        author_keys=include_author_keys,
    )


def crosstab_counts_and_period_percentages(
    df: pd.DataFrame,
    *,
    row_col: str,
    period_col: str = "_period",
    period_order: list[str] | None = None,
    row_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build count and within-period percentage tables for a row variable."""
    period_order = period_order or CORE_TEMPORAL_PERIOD_ORDER
    subset = df[df[row_col].notna() & df[row_col].astype(str).str.strip().ne("")].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = pd.crosstab(subset[row_col], subset[period_col])
    for period in period_order:
        if period not in counts.columns:
            counts[period] = 0
    counts = counts[[period for period in period_order if period in counts.columns]]
    if row_order is not None:
        counts = counts.reindex(ordered_index_for_crosstab(counts.index, row_order), fill_value=0)
    percentages = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1) * 100
    return counts, percentages


def save_grouped_bar_percentages(
    percentages: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    ylabel: str = "Percentage within period (%)",
    figsize_scale: float = 0.55,
    x_rotation: int = 35,
    dpi: int = 150,
) -> None:
    """Save grouped bar chart where columns are periods and rows are categories."""
    if percentages.empty:
        return
    categories = list(percentages.index)
    periods = list(percentages.columns)
    x_axis = np.arange(len(categories))
    bar_width = 0.8 / max(len(periods), 1)

    fig, ax = plt.subplots(figsize=(max(7, len(categories) * figsize_scale), 5))
    for idx, period in enumerate(periods):
        offset = (idx - (len(periods) - 1) / 2) * bar_width
        values = percentages[period].fillna(0).values
        ax.bar(x_axis + offset, values, width=bar_width, label=str(period))

    ax.set_xticks(x_axis)
    ax.set_xticklabels(categories, rotation=x_rotation, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    y_max = float(np.nanmax(percentages.values) * 1.2) if percentages.values.size else 100.0
    ax.set_ylim(0, max(100, y_max))
    ax.legend(title="Period")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
