"""Shared statistical/data helpers for analysis scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD_PRE_2022 = "pre_2022"
PERIOD_POST_2022 = "post_2022"


def normalize_bool_flag(series: pd.Series) -> pd.Series:
    if getattr(series.dtype, "name", "") == "bool":
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))


def period_pre_post_2022(y) -> str:
    if pd.isna(y):
        return "unknown"
    return PERIOD_POST_2022 if int(y) >= 2022 else PERIOD_PRE_2022


def period_three_way(y) -> str:
    if pd.isna(y):
        return "unknown"
    yy = int(y)
    if 2014 <= yy <= 2018:
        return "P1_2014_18"
    if 2019 <= yy <= 2021:
        return "P2_2019_21"
    if yy >= 2022:
        return "P3_2022plus"
    return "unknown"


def mode_with_tie_order(series: pd.Series, preference: list[str]) -> str:
    vc = series.dropna().astype(str).str.strip()
    vc = vc[vc.ne("")]
    if vc.empty:
        return ""
    counts = vc.value_counts()
    top = int(counts.iloc[0])
    tied = counts[counts.eq(top)].index.tolist()
    if len(tied) == 1:
        return str(tied[0])
    for pref in preference:
        if pref in tied:
            return pref
    return str(tied[0])


def bh_adjust(pvals: pd.Series) -> pd.Series:
    vals = pvals.to_numpy(dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return pd.Series(out, index=pvals.index)
    pv = vals[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    q = ranked * m / (np.arange(1, len(ranked) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[np.where(mask)[0]] = restored
    return pd.Series(out, index=pvals.index)
