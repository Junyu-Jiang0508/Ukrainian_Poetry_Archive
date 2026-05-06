"""Derived columns for GPT pronoun annotation tables (QA, temporal core, explicit mapping)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Matches validator message in 07_gpt_annotation._validate_pronoun_row
QA_INCONSISTENT_PRO_DROP_IMPLIED = (
    "INCONSISTENT: is_pro_drop=True but source_mapping missing '(IMPLIED)'"
)

CORE_TEMPORAL_PERIODS = frozenset({"2014_2021", "post_2022"})
# Column / legend order for crosstabs and plots (do not sort alphabetically).
CORE_TEMPORAL_PERIOD_ORDER: tuple[str, ...] = ("2014_2021", "post_2022")


def year_int(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        y = int(float(s))
    except (ValueError, TypeError):
        return None
    return y if 1000 <= y <= 2100 else None


def temporal_period_from_year(y: Optional[int]) -> Optional[str]:
    if y is None:
        return None
    if y < 2014:
        return "pre_2014"
    if y <= 2021:
        return "2014_2021"
    return "post_2022"


def reconcile_temporal_period(year_val, existing_period) -> str:
    """Prefer year when parseable; otherwise keep CSV temporal_period; else unknown."""
    y = year_int(year_val)
    if y is not None:
        p = temporal_period_from_year(y)
        return p if p is not None else "unknown"
    if existing_period is None or (isinstance(existing_period, float) and pd.isna(existing_period)):
        return "unknown"
    s = str(existing_period).strip()
    return s if s else "unknown"


def _reconcile_temporal_period_series(df: pd.DataFrame) -> pd.Series:
    """Vector-friendly reconciliation (two ``Series.map`` passes, no ``axis=1``)."""
    n = len(df.index)
    if "year" in df.columns:
        y_parsed = df["year"].map(year_int)
    else:
        y_parsed = pd.Series([None] * n, index=df.index, dtype=object)

    from_year = y_parsed.map(temporal_period_from_year)

    if "temporal_period" in df.columns:
        ext = df["temporal_period"]

        def _strip_existing(v: object) -> str:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            s = str(v).strip()
            if not s or s.lower() == "nan":
                return ""
            return s

        fb = ext.map(_strip_existing)
        fb = fb.mask(fb.eq(""), "unknown")
    else:
        fb = pd.Series("unknown", index=df.index, dtype=object)

    has_y = y_parsed.notna()
    out = from_year.where(has_y, fb)
    return out.fillna("unknown").astype(str)


def add_derived_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      qa_clean — qa_flag == OK (strict; use for sensitivity vs full sample).
      qa_inconsistency_pro_drop_metadata — largest benign-looking QA class.
      explicit_source_mapping — non-empty source_mapping without (IMPLIED) pro-drop tag.
      temporal_period_reconciled — from year when possible, else original temporal_period.
      in_core_temporal_analysis — temporal_period_reconciled in {2014_2021, post_2022}.
    """
    out = df.copy()
    if "qa_flag" not in out.columns:
        out["qa_flag"] = ""
    out["qa_flag"] = out["qa_flag"].fillna("").astype(str).str.strip()
    out["qa_clean"] = out["qa_flag"].eq("OK")
    out["qa_inconsistency_pro_drop_metadata"] = out["qa_flag"].eq(
        QA_INCONSISTENT_PRO_DROP_IMPLIED
    )

    if "source_mapping" in out.columns:
        sm = out["source_mapping"].fillna("").astype(str)
        has_text = sm.str.contains(r"\S", regex=True)
        implied = sm.str.contains("(IMPLIED)", regex=False)
        out["explicit_source_mapping"] = has_text & ~implied
    else:
        out["explicit_source_mapping"] = False

    if "temporal_period_reconciled" not in out.columns:
        out["temporal_period_reconciled"] = _reconcile_temporal_period_series(out)

    out["in_core_temporal_analysis"] = out["temporal_period_reconciled"].isin(
        CORE_TEMPORAL_PERIODS
    )
    return out


def qa_flag_category(flag: str) -> str:
    """Coarse bucket for methodology appendix tables."""
    s = (flag or "").strip()
    if not s or s == "OK":
        return "OK" if s == "OK" else "missing"
    if s == QA_INCONSISTENT_PRO_DROP_IMPLIED:
        return "INCONSISTENT_pro_drop_IMPLIED_metadata"
    if s.startswith("INCONSISTENT:") and "person=" in s and "pronoun=" in s:
        return "INCONSISTENT_person_pronoun_mismatch"
    if "ви-form" in s and "thou" in s:
        return "INCONSISTENT_vy_vs_thou"
    if "is_pro_drop=False" in s and "(IMPLIED)" in s:
        return "INCONSISTENT_pro_drop_false_but_IMPLIED"
    if s.startswith("INCONSISTENT:"):
        return "INCONSISTENT_other"
    return "other"
