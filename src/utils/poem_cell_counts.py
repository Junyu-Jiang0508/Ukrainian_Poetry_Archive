"""Poem × pronoun-cell counts and stanza/token exposure from sentence-level annotation rows."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils.finite_verb_exposure import count_finite_verbs_in_text
from utils.pronoun_encoding import (
    CELL_2PL_LEGACY,
    CELL_2PL_VY_POLITE_SINGULAR,
    CELL_2PL_VY_TRUE_PLURAL,
    COUNT_INPUT_CELLS,
    N_TOTAL_CELLS,
    POEM_COUNT_CELL_COLUMNS,
    poem_person_cell_column,
)
from utils.stats_common import period_three_way

log = logging.getLogger(__name__)


def _progress_iter(items: list[tuple[str, pd.DataFrame]], *, desc: str):
    """Progress iterator with graceful fallback when tqdm is unavailable."""
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(items, desc=desc, unit="poem")
    except Exception:
        return items


def _stanza_token_count(text: object) -> int:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    s = str(text).strip()
    if not s:
        return 0
    return len(s.split())


def build_poem_cell_table_with_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``poem_id``: split 2pl counts, legacy ``2pl`` sum, exposure, QC flags."""
    if "poem_id" not in df.columns:
        raise ValueError("build_poem_cell_table_with_exposure: missing poem_id")

    df = df.copy()
    df["poem_person_cell"] = poem_person_cell_column(df)

    count_subset = df[df["poem_person_cell"].isin(COUNT_INPUT_CELLS)]
    counts = (
        count_subset.groupby(["poem_id", "poem_person_cell"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    all_poems = pd.DataFrame({"poem_id": df["poem_id"].astype(str).unique()})
    counts = all_poems.merge(counts, on="poem_id", how="left")

    for cell in COUNT_INPUT_CELLS:
        if cell not in counts.columns:
            counts[cell] = 0
        else:
            counts[cell] = counts[cell].fillna(0)
        counts[cell] = counts[cell].astype(np.int64)

    counts[CELL_2PL_LEGACY] = (counts[CELL_2PL_VY_POLITE_SINGULAR] + counts[CELL_2PL_VY_TRUE_PLURAL]).astype(
        np.int64
    )
    assert counts[CELL_2PL_LEGACY].equals(counts[CELL_2PL_VY_POLITE_SINGULAR] + counts[CELL_2PL_VY_TRUE_PLURAL])

    stanza_chunks: list[pd.DataFrame] = []
    df_work = df.assign(_pid=df["poem_id"].astype(str))
    has_stanza = "stanza_index" in df.columns

    if not has_stanza:
        log.warning("[cell_counts] stanza_index missing; exposure_n_stanzas set to 0 for all poems.")
        for pid in df_work["_pid"].unique():
            stanza_chunks.append(
                pd.DataFrame(
                    [
                        {
                            "poem_id": pid,
                            "exposure_n_stanzas": 0,
                            "exposure_n_stanza_index_max": np.nan,
                            "exposure_n_tokens": 0,
                            "exposure_n_finite_verbs": 0,
                            "min_stanzas": 0,
                            "min_tokens": 0,
                        }
                    ]
                )
            )
    else:
        poem_groups = list(df_work.groupby("_pid", sort=False))
        for pid, g in _progress_iter(poem_groups, desc="Building poem exposure"):
            stanza_series = pd.to_numeric(g["stanza_index"], errors="coerce")
            g2 = g.assign(_stanza_num=stanza_series)
            g2_valid = g2.loc[g2["_stanza_num"].notna()]
            if g2_valid.empty:
                stanza_chunks.append(
                    pd.DataFrame(
                        [
                            {
                                "poem_id": pid,
                                "exposure_n_stanzas": 0,
                                "exposure_n_stanza_index_max": np.nan,
                                "exposure_n_tokens": 0,
                                "exposure_n_finite_verbs": 0,
                                "min_stanzas": 0,
                                "min_tokens": 0,
                            }
                        ]
                    )
                )
                continue

            txt_col = g2_valid["stanza_ukr"] if "stanza_ukr" in g2_valid.columns else pd.Series("", index=g2_valid.index)
            g2_valid = g2_valid.assign(_stanza_txt=txt_col)
            agg = (
                g2_valid.groupby("_stanza_num", dropna=False)
                .agg(
                    stanza_txt=("_stanza_txt", "first"),
                )
                .reset_index()
            )
            agg["_tok"] = agg["stanza_txt"].map(_stanza_token_count)
            agg["_fv"] = agg["stanza_txt"].map(count_finite_verbs_in_text)
            exposure_n_stanzas = int(agg.shape[0])
            stanza_max = float(agg["_stanza_num"].max())
            mx_int = int(round(stanza_max)) if np.isfinite(stanza_max) else 0
            if exposure_n_stanzas != mx_int:
                log.warning(
                    "[cell_counts] poem_id=%s: nunique_stanza_index=%s vs max(stanza_index)=%s",
                    pid,
                    exposure_n_stanzas,
                    mx_int,
                )

            exposure_tokens = int(agg["_tok"].sum())
            exposure_finite_verbs = int(agg["_fv"].sum())
            min_tok_stanza = int(agg["_tok"].min()) if len(agg) else 0

            stanza_chunks.append(
                pd.DataFrame(
                    [
                        {
                            "poem_id": pid,
                            "exposure_n_stanzas": exposure_n_stanzas,
                            "exposure_n_stanza_index_max": stanza_max,
                            "exposure_n_tokens": exposure_tokens,
                            "exposure_n_finite_verbs": exposure_finite_verbs,
                            "min_stanzas": exposure_n_stanzas,
                            "min_tokens": min_tok_stanza,
                        }
                    ]
                )
            )

    exposure = pd.concat(stanza_chunks, ignore_index=True) if stanza_chunks else pd.DataFrame()
    out = counts.merge(exposure, on="poem_id", how="left")
    out["exposure_n_stanzas"] = out["exposure_n_stanzas"].fillna(0).astype(np.int64)
    out["exposure_n_tokens"] = out["exposure_n_tokens"].fillna(0).astype(np.int64)
    out["exposure_n_finite_verbs"] = out["exposure_n_finite_verbs"].fillna(0).astype(np.int64)
    out["min_stanzas"] = out["min_stanzas"].fillna(0).astype(np.int64)
    out["min_tokens"] = out["min_tokens"].fillna(0).astype(np.int64)

    # n_total is a stable 5-cell sum (N_TOTAL_CELLS == PRIMARY_GLM_CELLS_BAYESIAN).
    # This is intentionally decoupled from the frequentist inference loop so that
    # gating thresholds like `--min-total-per-poem` keep the same semantics even
    # if the inferential cell-set changes in the future.
    out["n_total"] = out[list(N_TOTAL_CELLS)].sum(axis=1).astype(np.int64)

    agg_kw: dict[str, tuple[str, str]] = {
        "author": ("author", "first"),
        "language_clean": ("language_clean", "first"),
        "year_int": ("year_int", "first"),
    }
    if "period3" in df.columns:
        agg_kw["period3"] = ("period3", "first")
    meta = df.groupby("poem_id", as_index=False).agg(**agg_kw).copy()
    meta["poem_id"] = meta["poem_id"].astype(str)
    if "period3" not in meta.columns:
        meta["period3"] = meta["year_int"].map(period_three_way)
    else:
        miss = meta["period3"].isna() & meta["year_int"].notna()
        meta.loc[miss, "period3"] = meta.loc[miss, "year_int"].map(period_three_way)
    out = out.merge(meta, on="poem_id", how="left")

    out["include_in_offset_models"] = out["exposure_n_stanzas"].gt(0)
    out["include_in_fv_offset_models"] = out["exposure_n_finite_verbs"].gt(0)
    for _, row in out.loc[~out["include_in_offset_models"]].iterrows():
        log.warning(
            "[cell_counts] zero_stanza_exposure poem_id=%s exposure_n_stanzas=%s exposure_n_tokens=%s",
            row["poem_id"],
            int(row["exposure_n_stanzas"]),
            int(row["exposure_n_tokens"]),
        )

    kept = out.loc[out["include_in_offset_models"]]
    if len(kept):
        log.info(
            "[cell_counts] summary: n_poems=%s min_exposure_n_stanzas=%s min_exposure_n_tokens=%s min_exposure_n_finite_verbs=%s n_zero_stanza=%s n_zero_finite_verbs=%s",
            len(out),
            int(kept["exposure_n_stanzas"].min()),
            int(kept["exposure_n_tokens"].min()),
            int(out["exposure_n_finite_verbs"].min()),
            int((~out["include_in_offset_models"]).sum()),
            int((~out["include_in_fv_offset_models"]).sum()),
        )
    else:
        log.warning("[cell_counts] no poems with positive stanza exposure")

    col_order = (
        ["poem_id"]
        + list(POEM_COUNT_CELL_COLUMNS)
        + ["n_total"]
        + [
            "exposure_n_stanzas",
            "exposure_n_stanza_index_max",
            "exposure_n_tokens",
            "exposure_n_finite_verbs",
            "min_stanzas",
            "min_tokens",
            "include_in_offset_models",
            "include_in_fv_offset_models",
            "author",
            "language_clean",
            "year_int",
            "period3",
        ]
    )
    col_order = [c for c in col_order if c in out.columns]
    rest = [c for c in out.columns if c not in col_order]
    return out[col_order + rest]
