"""Export qualitative poetry corpus for RQ2 / RQ3 advisor review.

For each key pronoun cell (1PL, 1SG, 2SG), exports:
  1. All poems containing the pronoun (one row per poem), with full Ukrainian
     text, poem-level metadata, and RQ3 dependency-parse collocations merged in.
  2. Token-level syntactic contexts for the same poems (one row per pronoun token).
  3. Author-level shift summaries (quantitative context; separate file).
  4. A "spotlight" subset of contrasting authors' poems (P1 / P2 examples).

Requires ``02_modeling_pronoun_collocations.py`` (``pronoun_contexts_long.csv``).

Outputs  →  data/qualitative_corpus/
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "qualitative_corpus"
OUT.mkdir(parents=True, exist_ok=True)

# ── Inputs ──────────────────────────────────────────────────────────────────
ANNOTATION  = ROOT / "data" / "Annotated_Source" / "pronoun_annotation_v2.csv"
LAYER0      = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
AUTHOR_COV  = ROOT / "data" / "author_covariates.csv"
ROSTER      = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
Q2_SLOPES   = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_author_random_slope_summaries.csv"
CONTEXTS    = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "pronoun_contexts_long.csv"

DEPREL_READABLE = {
    "nsubj": "subject-of",
    "nsubj:pass": "subject-of (passive)",
    "obj": "object-of",
    "iobj": "indirect-object-of",
    "obl": "oblique-with",
    "nmod": "modifier-with",
    "nmod:poss": "possessor-of",
    "root": "root",
    "ccomp": "clausal-complement-of",
    "xcomp": "open-clausal-complement-of",
    "advcl": "adverbial-clause-of",
    "conj": "conjunct-of",
    "other": "other",
}

PERIOD_LABELS = {
    "2014_2021":   "P1 (2014–2021, pre-invasion)",
    "post_2022":   "P2 (2022+, post-invasion)",
    "P1_2014_2021":"P1 (2014–2021, pre-invasion)",
    "P2_2022_plus":"P2 (2022+, post-invasion)",
}

CELL_LABELS = {
    "1sg": "1SG (я / я)",
    "1pl": "1PL (ми / мы)",
    "2sg": "2SG (ти / ты)",
    "2pl": "2PL (ви / вы)",
}

LOCATION_LABELS = {
    "kyiv":            "Kyiv",
    "east_ukraine":    "East Ukraine",
    "west_ukraine":    "West Ukraine",
    "central_ukraine": "Central Ukraine",
    "south_ukraine":   "South Ukraine",
    "diaspora":        "Diaspora (abroad)",
    "crimea":          "Crimea",
    "born_abroad":     "Born abroad",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_annotation() -> pd.DataFrame:
    log.info("Loading pronoun annotation …")
    df = pd.read_csv(ANNOTATION, low_memory=False, on_bad_lines="skip")
    df.columns = df.columns.str.lstrip("\ufeff")
    df["person"]  = df["person"].fillna("").astype(str).str.strip().str.lower()
    df["number"]  = df["number"].fillna("").astype(str).str.strip().str.lower()
    df["person"]  = df["person"].replace({"1st": "1", "2nd": "2", "3rd": "3"})
    df["number"]  = df["number"].replace({"singular": "sg", "plural": "pl"})
    df["cell"]    = df["person"] + df["number"]
    df["period"]  = df["temporal_period"].astype(str).str.strip()
    df["period_label"] = df["period"].map(PERIOD_LABELS).fillna(df["period"])
    # exclude repeats and translations
    for flag in ("is_repeat", "is_translation"):
        if flag in df.columns:
            df = df[~df[flag].astype(str).str.lower().isin(["true", "1", "yes"])]
    return df


def load_roster_set() -> set[str]:
    if not ROSTER.is_file():
        return set()
    r = pd.read_csv(ROSTER, low_memory=False)
    return set(
        r.loc[r["included"].astype(str).str.lower().isin(["true", "1", "yes"]), "author"]
        .astype(str)
    )


def load_author_meta() -> pd.DataFrame:
    cov = pd.read_csv(AUTHOR_COV, low_memory=False)
    cov["location_jan2022_readable"] = cov["region_jan2022"].map(LOCATION_LABELS).fillna(
        cov["region_jan2022"].fillna("unknown")
    )
    cov["location_freeze_readable"] = cov["region_at_archive_freeze"].map(LOCATION_LABELS).fillna(
        cov["region_at_archive_freeze"].fillna("unknown")
    )
    cov["displaced"] = (
        cov["region_jan2022"].fillna("") != cov["region_at_archive_freeze"].fillna("")
    ) & cov["region_jan2022"].notna() & cov["region_at_archive_freeze"].notna()
    cov["location_stayed_vs_moved"] = cov.apply(
        lambda r: "diaspora" if r["region_at_archive_freeze"] == "diaspora"
        else ("displaced within Ukraine" if r["displaced"] else
              "stayed in place" if pd.notna(r["region_jan2022"]) else "unknown"),
        axis=1,
    )
    return cov[[
        "author", "gender", "birth_year", "generation_cohort",
        "region_jan2022", "region_at_archive_freeze",
        "language_xlsx_primary_at_freeze",
        "location_jan2022_readable", "location_freeze_readable",
        "location_stayed_vs_moved",
    ]]


def load_shift_summary(cell: str) -> pd.DataFrame:
    """Return deduplicated author × cell shift estimates (one row per author)."""
    if not Q2_SLOPES.is_file():
        return pd.DataFrame()
    df = pd.read_csv(Q2_SLOPES, low_memory=False)
    df = df[df["cell"] == cell].copy()
    # Keep the row with the most common language_stratum (pooled preferred)
    preferred = "pooled_Ukrainian_Russian"
    pooled = df[df["language_stratum"] == preferred]
    others  = df[~df["author"].isin(pooled["author"])]
    df = pd.concat([pooled, others], ignore_index=True).drop_duplicates(subset=["author"])
    return df[[
        "author", "cell",
        "author_total_period_shift_mean_log_mu",
        "author_total_period_shift_rate_ratio_mean",
        "author_total_period_shift_p_direction_gt0",
        "author_total_period_shift_hdi95_low",
        "author_total_period_shift_hdi95_high",
    ]].rename(columns={
        "author_total_period_shift_mean_log_mu":      "shift_log_mu",
        "author_total_period_shift_rate_ratio_mean":  "shift_rate_ratio",
        "author_total_period_shift_p_direction_gt0":  "p_increase",
        "author_total_period_shift_hdi95_low":        "shift_hdi95_low",
        "author_total_period_shift_hdi95_high":       "shift_hdi95_high",
    })


def load_pronoun_contexts() -> pd.DataFrame:
    """Token-level (poem, stanza, pronoun → head) rows from the RQ3 parse pipeline."""
    if not CONTEXTS.is_file():
        log.warning(
            "pronoun_contexts_long.csv not found — run 02_modeling_pronoun_collocations.py; "
            "qualitative exports will omit collocation columns."
        )
        return pd.DataFrame()
    df = pd.read_csv(CONTEXTS, low_memory=False)
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df["deprel_readable"] = df["deprel"].map(DEPREL_READABLE).fillna(df["deprel"].astype(str))
    return df


def _format_context_token(row: pd.Series) -> str:
    """Human-readable: st.3: ти [subject-of] → хотіти"""
    return (
        f"st.{int(row['stanza_index'])}: {row['pronoun_form']} "
        f"[{row['deprel_readable']}] → {row['head_lemma']}"
    )


def aggregate_poem_collocations(contexts: pd.DataFrame, cell: str) -> pd.DataFrame:
    """One row per poem_id with summarized dependency contexts for the target cell."""
    if contexts.empty:
        return pd.DataFrame(columns=["poem_id"])

    sub = contexts.loc[contexts["cell"].eq(cell)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["poem_id"])

    sub["context_token"] = sub.apply(_format_context_token, axis=1)

    def _join_unique(series: pd.Series) -> str:
        return "; ".join(sorted(set(series.dropna().astype(str))))

    rows: list[dict[str, object]] = []
    for poem_id, grp in sub.groupby("poem_id", sort=False):
        nsubj = grp.loc[grp["deprel"].isin(("nsubj", "nsubj:pass")), "head_lemma"]
        obj = grp.loc[grp["deprel"].eq("obj"), "head_lemma"]
        obl_nmod = grp.loc[grp["deprel"].isin(("obl", "nmod", "nmod:poss")), "head_lemma"]
        rows.append(
            {
                "poem_id": poem_id,
                "deparse_token_count": len(grp),
                "has_collocation_parse": True,
                "collocation_all": " | ".join(grp["context_token"].tolist()),
                "collocation_nsubj_verbs": _join_unique(nsubj),
                "collocation_obj_heads": _join_unique(obj),
                "collocation_obl_nmod_heads": _join_unique(obl_nmod),
            }
        )
    return pd.DataFrame(rows)


def attach_collocations(poem_df: pd.DataFrame, contexts: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Left-merge poem-level collocation summaries onto a qualitative poem table."""
    if contexts.empty or poem_df.empty:
        out = poem_df.copy()
        out["has_collocation_parse"] = False
        for col in (
            "deparse_token_count",
            "collocation_all",
            "collocation_nsubj_verbs",
            "collocation_obj_heads",
            "collocation_obl_nmod_heads",
        ):
            if col not in out.columns:
                out[col] = "" if col != "deparse_token_count" else 0
        return out

    coll = aggregate_poem_collocations(contexts, cell)
    out = poem_df.merge(coll, on="poem_id", how="left")
    out["has_collocation_parse"] = out["has_collocation_parse"].fillna(False)
    out["deparse_token_count"] = out["deparse_token_count"].fillna(0).astype(int)
    for col in (
        "collocation_all",
        "collocation_nsubj_verbs",
        "collocation_obj_heads",
        "collocation_obl_nmod_heads",
    ):
        out[col] = out[col].fillna("")
    return out


def export_token_contexts(
    poem_df: pd.DataFrame, contexts: pd.DataFrame, cell: str, suffix: str
) -> None:
    """One row per dependency-parsed pronoun token for poems in *poem_df*."""
    if contexts.empty or poem_df.empty:
        return

    poem_ids = set(poem_df["poem_id"].astype(str))
    sub = contexts.loc[
        contexts["cell"].eq(cell) & contexts["poem_id"].isin(poem_ids)
    ].copy()
    if sub.empty:
        log.warning("No syntactic contexts for %s / %s", cell, suffix)
        return

    meta_cols = ["year", "period", "period_label", "spotlight_group"]
    meta = poem_df[["poem_id", "author"] + [c for c in meta_cols if c in poem_df.columns]].drop_duplicates(
        "poem_id"
    )
    # Prefer poem-table author/period labels; keep parse author only when missing.
    for col in meta.columns:
        if col == "poem_id":
            continue
        if col in sub.columns:
            sub = sub.drop(columns=[col])
    sub = sub.merge(meta, on="poem_id", how="left")
    sort_cols = [c for c in ("author", "poem_id", "stanza_index", "deprel", "head_lemma") if c in sub.columns]
    sub = sub.sort_values(sort_cols)

    keep = [
        "author",
        "poem_id",
        "year",
        "period",
        "period_label",
        "spotlight_group",
        "stanza_index",
        "language",
        "pronoun_form",
        "pronoun_lemma",
        "cell",
        "deprel",
        "deprel_readable",
        "head_lemma",
        "head_upos",
    ]
    keep = [c for c in keep if c in sub.columns]
    out_path = OUT / f"{suffix}_{cell}_syntactic_contexts.csv"
    sub[keep].to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Saved %s  (%d token-context rows)", out_path.name, len(sub))


def load_full_poem_texts() -> pd.DataFrame:
    log.info("Loading full poem texts from layer0 …")
    df = pd.read_csv(
        LAYER0, low_memory=False,
        usecols=["poem_id", "Poem full text (copy and paste)", "Date posted", "url of facebook post"],
    )
    df = df.rename(columns={
        "Poem full text (copy and paste)": "full_poem_text",
        "Date posted":                     "date_posted",
        "url of facebook post":            "facebook_url",
    })
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    return df


# ── Core export: poem-level qualitative corpus ───────────────────────────────

def export_poem_corpus(
    ann: pd.DataFrame,
    cell: str,
    roster: set[str],
    full_poems: pd.DataFrame,
    contexts: pd.DataFrame,
) -> pd.DataFrame:
    """One row per poem containing the target pronoun (roster authors only)."""
    sub = ann[ann["cell"] == cell].copy()
    sub = sub[sub["author"].isin(roster)]
    sub["poem_id"] = sub["poem_id"].astype(str).str.strip()

    poem_meta = (
        sub.groupby("poem_id", as_index=False)
        .agg(
            author=("author", "first"),
            year=("year", "first"),
            period=("period", "first"),
            period_label=("period_label", "first"),
            pronoun_token_count=("pronoun_word", "count"),
            pronoun_words=("pronoun_word", lambda s: "; ".join(sorted(set(s.dropna().astype(str))))),
        )
    )

    if not full_poems.empty:
        poem_meta = poem_meta.merge(
            full_poems[["poem_id", "full_poem_text", "date_posted"]],
            on="poem_id", how="left",
        )

    poem_meta["cell"] = cell
    poem_meta["cell_label"] = CELL_LABELS.get(cell, cell)
    poem_meta = attach_collocations(poem_meta, contexts, cell)

    keep = [
        "author", "poem_id", "year", "date_posted", "period", "period_label",
        "pronoun_token_count", "deparse_token_count", "pronoun_words",
        "has_collocation_parse",
        "collocation_nsubj_verbs", "collocation_obj_heads", "collocation_obl_nmod_heads",
        "collocation_all",
        "full_poem_text",
        "cell", "cell_label",
    ]
    keep = [c for c in keep if c in poem_meta.columns]
    poem_meta = poem_meta[keep].sort_values(["author", "period", "year", "poem_id"])

    out_path = OUT / f"qualitative_{cell}_poems_all_authors.csv"
    poem_meta.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_parsed = int(poem_meta["has_collocation_parse"].sum()) if "has_collocation_parse" in poem_meta else 0
    log.info(
        "Saved %s  (%d poems, %d authors, %d with parse contexts)",
        out_path.name,
        len(poem_meta),
        poem_meta["author"].nunique(),
        n_parsed,
    )
    export_token_contexts(poem_meta, contexts, cell, suffix="qualitative")
    return poem_meta


# ── Spotlight: most contrasting authors ──────────────────────────────────────

def _spotlight_authors(shifts: pd.DataFrame, meta: pd.DataFrame,
                       top_n: int = 6) -> dict[str, list[str]]:
    """
    Return {"increasing": [...], "decreasing": [...]} of spotlight author names.
    Prefer roster authors with both P1 and P2 data, sorted by shift magnitude.
    """
    if shifts.empty:
        return {"increasing": [], "decreasing": []}

    df = shifts.merge(meta[["author", "location_stayed_vs_moved",
                             "language_xlsx_primary_at_freeze"]], on="author", how="left")

    increasing = (
        df[df["shift_log_mu"] > 0.3]
        .sort_values("shift_log_mu", ascending=False)
        .head(top_n)["author"].tolist()
    )
    decreasing = (
        df[df["shift_log_mu"] < -0.1]
        .sort_values("shift_log_mu", ascending=True)
        .head(top_n)["author"].tolist()
    )
    return {"increasing": increasing, "decreasing": decreasing}


def export_spotlight(
    poem_df: pd.DataFrame,
    shifts: pd.DataFrame,
    meta: pd.DataFrame,
    cell: str,
    contexts: pd.DataFrame,
) -> None:
    """Export contrasting authors' poems (up to 4 per author × period)."""
    spotlight = _spotlight_authors(shifts, meta)
    all_spotlight = spotlight["increasing"] + spotlight["decreasing"]
    if not all_spotlight:
        return

    sub = poem_df[poem_df["author"].isin(all_spotlight)].copy()
    # Collocations already merged in poem_df; re-attach only if missing (defensive).
    if "collocation_all" not in sub.columns:
        sub = attach_collocations(sub, contexts, cell)
    sub["spotlight_group"] = sub["author"].apply(
        lambda a: "TOP INCREASE" if a in spotlight["increasing"] else "TOP DECREASE"
    )

    rows = []
    for _, grp in sub.groupby("author"):
        for period_key in ["2014_2021", "post_2022"]:
            rows.append(grp[grp["period"] == period_key].head(4))
    if not rows:
        return

    spotlight_df = pd.concat(rows, ignore_index=True)
    spotlight_df = spotlight_df.sort_values(
        ["spotlight_group", "author", "period", "year", "poem_id"]
    )

    out_path = OUT / f"spotlight_{cell}_contrasting_authors.csv"
    spotlight_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_parsed = int(spotlight_df["has_collocation_parse"].sum()) if "has_collocation_parse" in spotlight_df else 0
    log.info(
        "Saved %s  (%d poems, %d with parse contexts; authors: %s)",
        out_path.name,
        len(spotlight_df),
        n_parsed,
        ", ".join(spotlight_df["author"].unique()),
    )
    export_token_contexts(spotlight_df, contexts, cell, suffix="spotlight")


# ── Cross-author summary ──────────────────────────────────────────────────────

def export_author_summary(shifts: pd.DataFrame, meta: pd.DataFrame,
                          ann: pd.DataFrame, roster: set[str], cell: str) -> None:
    """One-row-per-author summary for the cell, with poem/stanza counts."""
    if shifts.empty:
        return

    sub = ann[(ann["cell"] == cell) & (ann["author"].isin(roster))]

    counts = (
        sub.groupby(["author", "period"])
        .agg(
            pronoun_tokens=("pronoun_word", "count"),
            unique_poems=("poem_id", "nunique"),
        )
        .unstack("period")
        .fillna(0)
        .astype(int)
    )
    counts.columns = [f"{metric}_{period}" for metric, period in counts.columns]
    counts = counts.reset_index()

    summary = shifts.merge(counts, on="author", how="left")
    summary = summary.merge(meta[[
        "author", "gender", "birth_year", "generation_cohort",
        "language_xlsx_primary_at_freeze",
        "location_jan2022_readable", "location_freeze_readable",
        "location_stayed_vs_moved",
    ]], on="author", how="left")

    summary["shift_direction"] = summary["shift_log_mu"].apply(
        lambda x: "increased" if pd.notna(x) and x > 0.1
        else ("decreased" if pd.notna(x) and x < -0.1 else "stable")
    )
    summary = summary.sort_values("shift_log_mu", ascending=False)

    out_path = OUT / f"author_summary_{cell}.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Saved %s  (%d authors)", out_path.name, len(summary))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ann         = load_annotation()
    roster      = load_roster_set()
    meta        = load_author_meta()
    full_poems  = load_full_poem_texts()
    contexts    = load_pronoun_contexts()

    for cell in ("1pl", "1sg", "2sg"):
        log.info("=== Cell: %s ===", cell.upper())
        shifts    = load_shift_summary(cell)
        poem_df = export_poem_corpus(ann, cell, roster, full_poems, contexts)
        export_author_summary(shifts, meta, ann, roster, cell)
        export_spotlight(poem_df, shifts, meta, cell, contexts)

    log.info("Done. Files saved to: %s", OUT)


if __name__ == "__main__":
    main()
