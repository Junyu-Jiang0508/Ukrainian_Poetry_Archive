"""Poem-level language strata: Ukrainian-only, Russian-only, pooled (Ukrainian ∪ Russian).

Also: explicit exclusions for inference (mixed-language, small langs, Crimean-Tatar/Qirimli family),
with audit logging and dropped-poem manifests for Methods.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Exact `language` field matches from annotation CSV (see MAJOR_LANGUAGES elsewhere).
POOL_CODES = frozenset({"Ukrainian", "Russian"})

# Ordered reporting: pooled first, then single-language strata.
LANGUAGE_STRATA = ("pooled_Ukrainian_Russian", "Ukrainian", "Russian")

STRATUM_PRIMARY_FOR_BH: dict[str, bool] = {
    "Ukrainian": True,
    "Russian": True,
    "pooled_Ukrainian_Russian": False,
}

CRIMEAN_TATAR_FAMILY_CODES = frozenset({"Qirimli", "Russian, Qirimli", "Ukrainian, Qirimli"})

# Single-language codes excluded from main inference strata (verbatim labels in corpus).
EXCLUDED_FROM_INFERENCE_EXACT = frozenset(
    {
        "English",
        "Hebrew",
        "Polish",
        "Spanish",
        "French",
        "German",
        "Italian",
    }
)


def classify_language_inference_eligibility(language_clean: object) -> tuple[bool, str]:
    """Return (keep_for_inference_pool, reason_if_dropped).

    Allowed through: Ukrainian, Russian only (possibly further split by strata).
    Dropped: empty, comma-separated codes (mixed-language), Crimean-Tatar/Qirimli family, excluded small langs.
    """
    if language_clean is None or (isinstance(language_clean, float) and pd.isna(language_clean)):
        return False, "empty_language"
    s = str(language_clean).strip()
    if not s or s.lower() == "nan":
        return False, "empty_language"
    if "," in s:
        return False, "mixed_language_comma"
    if s in CRIMEAN_TATAR_FAMILY_CODES:
        return False, "crimean_tatar_or_qirimli_family"
    if s in EXCLUDED_FROM_INFERENCE_EXACT:
        return False, "excluded_small_language_exact"
    if s not in POOL_CODES:
        return False, f"outside_pool_Ukrainian_Russian:{s}"
    return True, ""


def filter_annotation_for_inference_language(
    df: pd.DataFrame,
    *,
    audit_dir: Path | None = None,
    audit_filename: str = "dropped_poems_language_constraints.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop annotation rows whose poem fails language inference eligibility; log each dropped poem."""
    miss = {"poem_id", "language_clean"} - set(df.columns)
    if miss:
        raise ValueError(f"filter_annotation_for_inference_language: missing columns {sorted(miss)}")
    d = df.copy()
    d["poem_id"] = d["poem_id"].astype(str).str.strip()
    poem_lang = d.groupby("poem_id", sort=False)["language_clean"].first().rename("language_clean").reset_index()

    dropped_rows: list[dict[str, object]] = []
    keep_ids: list[str] = []
    for _, r in poem_lang.iterrows():
        pid = str(r["poem_id"])
        ok, reason = classify_language_inference_eligibility(r["language_clean"])
        if ok:
            keep_ids.append(pid)
        else:
            log.warning(
                "[stratify] dropped poem_id=%s language=%s reason=%s",
                pid,
                r["language_clean"],
                reason,
            )
            dropped_rows.append({"poem_id": pid, "language": r["language_clean"], "reason": reason})

    keep_set = set(keep_ids)
    out = d[d["poem_id"].isin(keep_set)].copy()
    dropped_df = pd.DataFrame(dropped_rows)
    if audit_dir is not None and not dropped_df.empty:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        p = audit_dir / audit_filename
        dropped_df.to_csv(p, index=False)
        log.info("[stratify] wrote %s (%s poems)", p, len(dropped_df))

    return out, dropped_df


def filter_poems_by_language_stratum(df: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Keep poem rows whose ``language_clean`` falls in the stratum (exact labels)."""
    if stratum == "pooled_Ukrainian_Russian":
        return df.loc[df["language_clean"].isin(POOL_CODES)].copy()
    if stratum == "Ukrainian":
        return df.loc[df["language_clean"].eq("Ukrainian")].copy()
    if stratum == "Russian":
        return df.loc[df["language_clean"].eq("Russian")].copy()
    raise ValueError(f"Unknown language stratum: {stratum!r}; use one of {LANGUAGE_STRATA}")


def primary_stratum_for_bh(stratum: str) -> bool:
    """Whether this stratum is in the BH primary family (exclude pooled unless explicitly overridden)."""
    if stratum not in STRATUM_PRIMARY_FOR_BH:
        raise KeyError(f"Unknown stratum {stratum!r}")
    return STRATUM_PRIMARY_FOR_BH[stratum]
