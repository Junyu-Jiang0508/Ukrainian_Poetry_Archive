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
    if 2014 <= yy <= 2021:
        return "P1_2014_2021"
    if yy >= 2022:
        return "P2_2022_plus"
    return "unknown"


PERIOD_P1_LABEL = "P1_2014_2021"
PERIOD_P2_LABEL = "P2_2022_plus"
PERIODS_P1_P2 = (PERIOD_P1_LABEL, PERIOD_P2_LABEL)


def period_p1_p2_exclude_pre_2014(y) -> str:
    """Binary P1/P2 with years before 2014 marked unknown (triple-window robustness)."""
    if pd.isna(y):
        return "unknown"
    yy = int(y)
    if yy < 2014:
        return "unknown"
    return period_three_way(y)


def period_p1_p2_invasion_precise(poem_dates: pd.Series) -> pd.Series:
    """Post-2022-02-24 → P2; 2014-01-01 through 2022-02-23 → P1; else unknown.

    ``poem_dates`` must be pandas datetime64 UTC-naïve dates (DATE of publication).
    """
    invasion = pd.Timestamp("2022-02-24")
    hi = poem_dates.dt.normalize()
    # Pandas comparison with NaT yields False comparisons; keep unknown
    p2 = hi.ge(invasion) & poem_dates.notna()
    p1_period = hi.ge(pd.Timestamp("2014-01-01")) & hi.lt(invasion)
    out = pd.Series("unknown", index=poem_dates.index, dtype=object)
    out.loc[p1_period] = PERIOD_P1_LABEL
    out.loc[p2] = PERIOD_P2_LABEL
    return out


def assign_author_calendar_period_with_onset_filter(
    year_int: pd.Series,
    author: pd.Series,
    *,
    max_onset_year: int = 2014,
) -> tuple[pd.Series, pd.Series]:
    """Calendar P1/P2 via ``period_three_way`` but mark authors whose first poem year is after ``max_onset_year``.

    Rows for those authors → ``period3`` unchanged from calendar code but companion mask excludes them
    unless caller filters. Implemented as: onset map + mask indicating ``author_passes_onset_filter``.
    """
    df = pd.DataFrame({"y": year_int, "a": author.astype(str)})
    onset = df.groupby("a")["y"].transform("min")
    passes = onset.le(max_onset_year) & onset.notna()
    per = year_int.map(period_three_way)
    return per, passes


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
