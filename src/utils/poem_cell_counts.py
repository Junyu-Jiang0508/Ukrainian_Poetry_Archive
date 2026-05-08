"""Poem × pronoun-cell counts and stanza/token exposure from sentence-level annotation rows."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_poem_cell_table_with_exposure(
    df: pd.DataFrame,
    *,
    finite_verb_df: pd.DataFrame | None = None,
    fv_exposure_strict: bool = False,
    discontinuity_manifest_path: Path | None = None,
) -> pd.DataFrame:
    """One row per ``poem_id``: split 2pl counts, legacy ``2pl`` sum, exposure, QC flags.

    ``finite_verb_df`` must be stanza-level counts from ``stanza_finite_verb_counts.csv``
    (columns ``poem_id``, ``stanza_index``, ``n_finite_verbs``, ``n_finite_verbs_excl_imperative``).
    When omitted, finite-verb columns are NaN and FV-offset inclusion flags are False.
    """
    if fv_exposure_strict and finite_verb_df is None:
        raise ValueError("fv_exposure_strict requires finite_verb_df")

    if finite_verb_df is None:
        log.warning(
            "[cell_counts] finite_verb_df not provided; exposure_n_finite_verbs set to NaN "
            "(run 00e_compute_finite_verb_exposure.py and pass load_finite_verb_counts)."
        )

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

    fv_cols = ("n_finite_verbs", "n_finite_verbs_excl_imperative")
    fv_for_merge: pd.DataFrame | None = None
    if finite_verb_df is not None:
        fv_for_merge = finite_verb_df.copy()
        fv_for_merge["poem_id"] = fv_for_merge["poem_id"].astype(str).str.strip()
        fv_for_merge["stanza_index"] = pd.to_numeric(fv_for_merge["stanza_index"], errors="coerce")
        fv_for_merge = fv_for_merge.dropna(subset=["stanza_index"])
        fv_for_merge["stanza_index"] = fv_for_merge["stanza_index"].astype(np.int64)
        keep = ["poem_id", "stanza_index"] + [c for c in fv_cols if c in fv_for_merge.columns]
        fv_for_merge = fv_for_merge[keep].drop_duplicates(subset=["poem_id", "stanza_index"], keep="first")

    stanza_chunks: list[pd.DataFrame] = []
    discontinuity_rows: list[dict[str, object]] = []
    df_work = df.assign(_pid=df["poem_id"].astype(str))
    has_stanza = "stanza_index" in df.columns

    if not has_stanza:
        log.warning("[cell_counts] stanza_index missing; exposure_n_stanzas set to 0 for all poems.")
        nan_fv = finite_verb_df is None
        for pid in df_work["_pid"].unique():
            fv_def = np.nan if nan_fv else 0
            fv_ex = np.nan if nan_fv else 0
            stanza_chunks.append(
                pd.DataFrame(
                    [
                        {
                            "poem_id": pid,
                            "exposure_n_stanzas": 0,
                            "exposure_n_stanza_index_max": np.nan,
                            "exposure_n_tokens": 0,
                            "exposure_n_finite_verbs": fv_def,
                            "exposure_n_finite_verbs_excl_imperative": fv_ex,
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
                nan_fv = finite_verb_df is None
                fv_def = np.nan if nan_fv else 0
                fv_ex = np.nan if nan_fv else 0
                stanza_chunks.append(
                    pd.DataFrame(
                        [
                            {
                                "poem_id": pid,
                                "exposure_n_stanzas": 0,
                                "exposure_n_stanza_index_max": np.nan,
                                "exposure_n_tokens": 0,
                                "exposure_n_finite_verbs": fv_def,
                                "exposure_n_finite_verbs_excl_imperative": fv_ex,
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
            agg["_stanza_num"] = agg["_stanza_num"].astype(np.int64)
            agg["poem_id"] = pid
            agg = agg.rename(columns={"_stanza_num": "stanza_index"})

            exposure_n_stanzas = int(agg.shape[0])
            stanza_max = float(agg["stanza_index"].max())
            mx_int = int(round(stanza_max)) if np.isfinite(stanza_max) else 0
            if exposure_n_stanzas != mx_int:
                log.warning(
                    "[cell_counts] poem_id=%s: nunique_stanza_index=%s vs max(stanza_index)=%s",
                    pid,
                    exposure_n_stanzas,
                    mx_int,
                )
                discontinuity_rows.append(
                    {
                        "poem_id": pid,
                        "nunique_stanza_index": exposure_n_stanzas,
                        "max_stanza_index": mx_int,
                    }
                )

            exposure_tokens = int(agg["_tok"].sum())
            if fv_for_merge is not None:
                m = agg.merge(fv_for_merge, on=["poem_id", "stanza_index"], how="left")
                m["n_finite_verbs"] = m["n_finite_verbs"].fillna(0).astype(np.int64)
                m["n_finite_verbs_excl_imperative"] = m["n_finite_verbs_excl_imperative"].fillna(0).astype(np.int64)
                exposure_finite_verbs = int(m["n_finite_verbs"].sum())
                exposure_finite_verbs_ex = int(m["n_finite_verbs_excl_imperative"].sum())
            else:
                exposure_finite_verbs = np.nan
                exposure_finite_verbs_ex = np.nan

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
                            "exposure_n_finite_verbs_excl_imperative": exposure_finite_verbs_ex,
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

    if finite_verb_df is not None:
        out["exposure_n_finite_verbs"] = pd.to_numeric(out["exposure_n_finite_verbs"], errors="coerce").fillna(0).astype(
            np.int64
        )
        out["exposure_n_finite_verbs_excl_imperative"] = pd.to_numeric(
            out["exposure_n_finite_verbs_excl_imperative"], errors="coerce"
        ).fillna(0).astype(np.int64)
    else:
        out["exposure_n_finite_verbs"] = pd.to_numeric(out["exposure_n_finite_verbs"], errors="coerce")
        out["exposure_n_finite_verbs_excl_imperative"] = pd.to_numeric(
            out["exposure_n_finite_verbs_excl_imperative"], errors="coerce"
        )

    out["min_stanzas"] = out["min_stanzas"].fillna(0).astype(np.int64)
    out["min_tokens"] = out["min_tokens"].fillna(0).astype(np.int64)

    # n_total is a stable 5-cell sum (N_TOTAL_CELLS == PRIMARY_GLM_CELLS_BAYESIAN).
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
    if finite_verb_df is not None:
        out["include_in_fv_offset_models"] = out["exposure_n_finite_verbs"].gt(0)
        out["include_in_fv_excl_imp_offset_models"] = out["exposure_n_finite_verbs_excl_imperative"].gt(0)
    else:
        out["include_in_fv_offset_models"] = False
        out["include_in_fv_excl_imp_offset_models"] = False

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
            "[cell_counts] summary: n_poems=%s min_exposure_n_stanzas=%s min_exposure_n_tokens=%s "
            "min_exposure_n_finite_verbs=%s n_zero_stanza=%s n_zero_finite_verbs=%s",
            len(out),
            int(kept["exposure_n_stanzas"].min()),
            int(kept["exposure_n_tokens"].min()),
            int(out["exposure_n_finite_verbs"].fillna(0).min()) if finite_verb_df is not None else -1,
            int((~out["include_in_offset_models"]).sum()),
            int((~out["include_in_fv_offset_models"]).sum()) if finite_verb_df is not None else -1,
        )
    else:
        log.warning("[cell_counts] no poems with positive stanza exposure")

    if discontinuity_manifest_path is not None:
        disc = (
            pd.DataFrame(discontinuity_rows)
            if discontinuity_rows
            else pd.DataFrame(columns=["poem_id", "nunique_stanza_index", "max_stanza_index"])
        )
        discontinuity_manifest_path = Path(discontinuity_manifest_path)
        discontinuity_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        disc.to_csv(discontinuity_manifest_path, index=False)
        log.info(
            "[cell_counts] wrote stanza_index discontinuity manifest: %s (%s poems)",
            discontinuity_manifest_path,
            len(disc),
        )

    col_order = (
        ["poem_id"]
        + list(POEM_COUNT_CELL_COLUMNS)
        + ["n_total"]
        + [
            "exposure_n_stanzas",
            "exposure_n_stanza_index_max",
            "exposure_n_tokens",
            "exposure_n_finite_verbs",
            "exposure_n_finite_verbs_excl_imperative",
            "min_stanzas",
            "min_tokens",
            "include_in_offset_models",
            "include_in_fv_offset_models",
            "include_in_fv_excl_imp_offset_models",
            "author",
            "language_clean",
            "year_int",
            "period3",
        ]
    )
    col_order = [c for c in col_order if c in out.columns]
    rest = [c for c in out.columns if c not in col_order]
    return out[col_order + rest]
