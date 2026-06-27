"""Export detailed corpus information for RQ2 and RQ3 to data/rq_corpus_details/.

RQ2: Author-level pronoun usage pre/post invasion (heterogeneity analysis)
RQ3: Syntactic collocation contexts for each pronoun cell × period

Outputs
-------
data/rq_corpus_details/
  rq2_author_pronoun_counts_by_period.csv
      One row per author × cell × period. Poem counts, pronoun token counts,
      stanza exposure, + author metadata.
  rq2_author_period_shifts.csv
      Bayesian random-slope posterior summaries per author × cell (from Q2 model).
  rq2_population_level_shifts.csv
      Population-level posterior shifts per cell (from Q2 model).
  rq3_pronoun_syntactic_contexts.csv
      Every pronoun occurrence with its dependency-parse head (verb/noun) and period.
  rq3_collocation_scores_by_cell_period.csv
      All collocate × cell × period log-Dice / PMI / G² scores.
  rq3_differential_collocations.csv
      Collocates with significant PMI shift across the invasion cutpoint.
  rq3_top_collocates_summary.csv
      Top-20 collocates per cell × period (readable overview for advisor).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "rq_corpus_details"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Input paths
# --------------------------------------------------------------------------- #
ANNOTATION = ROOT / "data" / "Annotated_Source" / "pronoun_annotation_v2.csv"
AUTHOR_COV = ROOT / "data" / "author_covariates.csv"
ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"

Q2_POEM_COUNTS = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_poem_cell_counts.csv"
Q2_AUTHOR_SLOPES = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_author_random_slope_summaries.csv"
Q2_POPULATION = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_population_shifts_by_cell.csv"

COLLOCATIONS = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "collocations_by_cell_period.csv"
CONTEXTS = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "pronoun_contexts_long.csv"
DIFFERENTIAL = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "differential_collocations.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_roster_set(roster_path: Path) -> set[str] | None:
    if not roster_path.is_file():
        return None
    r = pd.read_csv(roster_path, low_memory=False)
    if {"included", "author"}.issubset(r.columns):
        return set(r.loc[r["included"].astype(str).str.lower().isin(["true", "1", "yes"]), "author"].astype(str))
    return None


def period_label(period_val) -> str:
    """Map raw period codes to readable labels."""
    mapping = {
        "P1_2014_2021": "P1 (2014–2021, pre-invasion)",
        "P2_2022_plus": "P2 (2022+, post-invasion)",
        "2014_2021":    "P1 (2014–2021, pre-invasion)",
        "post_2022":    "P2 (2022+, post-invasion)",
    }
    return mapping.get(str(period_val), str(period_val))


# --------------------------------------------------------------------------- #
# RQ2 exports
# --------------------------------------------------------------------------- #

def export_rq2_author_counts() -> None:
    """Build author × cell × period pronoun token counts with metadata."""
    log.info("Loading pronoun annotation for RQ2 author counts …")
    ann = pd.read_csv(ANNOTATION, low_memory=False, on_bad_lines="skip")
    ann.columns = ann.columns.str.lstrip("\ufeff")  # strip BOM if present

    # Normalise person/number into cell
    ann["person"] = ann["person"].fillna("").astype(str).str.strip().str.lower()
    ann["number"] = ann["number"].fillna("").astype(str).str.strip().str.lower()
    # Map e.g. "1st"→"1", "2nd"→"2", "3rd"→"3", "singular"→"sg", "plural"→"pl"
    ann["person"] = ann["person"].replace({"1st": "1", "2nd": "2", "3rd": "3"})
    ann["number"] = ann["number"].replace({"singular": "sg", "plural": "pl"})
    ann["cell"] = ann["person"] + ann["number"]

    # Keep primary inference cells
    primary_cells = {"1sg", "1pl", "2sg", "2pl"}
    ann = ann[ann["cell"].isin(primary_cells)].copy()

    # Standardise period
    if "temporal_period" in ann.columns:
        ann["period"] = ann["temporal_period"].map({
            "2014_2021": "P1_2014_2021",
            "post_2022": "P2_2022_plus",
        }).fillna(ann["temporal_period"])
    elif "year" in ann.columns:
        ann["year_int"] = pd.to_numeric(ann["year"], errors="coerce")
        ann["period"] = ann["year_int"].apply(
            lambda y: "P1_2014_2021" if pd.notna(y) and y < 2022 else "P2_2022_plus"
        )
    else:
        log.warning("No period column found; skipping RQ2 author counts.")
        return

    # Exclude repeats and translations
    for flag in ("is_repeat", "is_translation"):
        if flag in ann.columns:
            ann = ann[~ann[flag].astype(str).str.lower().isin(["true", "1", "yes"])]

    # Count pronoun tokens per author × period × cell
    token_counts = (
        ann.groupby(["author", "language", "period", "cell"], dropna=False)
        .size()
        .reset_index(name="pronoun_token_count")
    )

    # Unique poems per author × period
    poem_counts = (
        ann.drop_duplicates(subset=["poem_id", "author", "period"])
        .groupby(["author", "period"], dropna=False)
        .size()
        .reset_index(name="poem_count")
    )

    result = token_counts.merge(poem_counts, on=["author", "period"], how="left")

    # Add stanza exposure from Q2 poem-cell counts if available
    if Q2_POEM_COUNTS.is_file():
        pc = pd.read_csv(Q2_POEM_COUNTS, low_memory=False)
        # aggregate exposure per author × period
        if "exposure_n_stanzas" in pc.columns:
            exp = (
                pc.drop_duplicates(subset=["poem_id", "author", "period3"])
                .groupby(["author", "period3"], dropna=False)["exposure_n_stanzas"]
                .sum()
                .reset_index()
                .rename(columns={"period3": "period", "exposure_n_stanzas": "total_stanza_exposure"})
            )
            result = result.merge(exp, on=["author", "period"], how="left")

    # Merge author metadata
    if AUTHOR_COV.is_file():
        cov = pd.read_csv(AUTHOR_COV, low_memory=False)
        keep_cols = ["author", "gender", "birth_year", "generation_cohort",
                     "region_of_birth", "region_jan2022", "language_xlsx_primary_at_freeze",
                     "bilingual_switcher_corpus"]
        keep_cols = [c for c in keep_cols if c in cov.columns]
        result = result.merge(cov[keep_cols], on="author", how="left")

    # Flag roster authors
    roster = load_roster_set(ROSTER)
    if roster is not None:
        result["in_inference_roster"] = result["author"].isin(roster)

    # Readable labels
    result["period_label"] = result["period"].map(period_label)
    result["cell_label"] = result["cell"].map({
        "1sg": "1SG (я / я)",
        "1pl": "1PL (ми / мы)",
        "2sg": "2SG (ти / ты)",
        "2pl": "2PL (ви / вы)",
    })

    # Sort
    result = result.sort_values(["author", "cell", "period"]).reset_index(drop=True)

    out = OUT_DIR / "rq2_author_pronoun_counts_by_period.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("Saved %s  (%d rows)", out.name, len(result))


def export_rq2_model_outputs() -> None:
    """Copy Q2 model posterior summaries with readable column labels."""

    # Author-level random slope summaries
    if Q2_AUTHOR_SLOPES.is_file():
        df = pd.read_csv(Q2_AUTHOR_SLOPES, low_memory=False)
        # Add readable labels
        df["period_shift_direction"] = df.get(
            "author_total_period_shift_p_direction_gt0",
            pd.Series(dtype=float)
        ).apply(lambda p: "increase" if float(p) > 0.5 else "decrease" if pd.notna(p) else "")
        df["cell_label"] = df["cell"].map({
            "1sg": "1SG (я / я)",
            "1pl": "1PL (ми / мы)",
            "2sg": "2SG (ти / ты)",
            "2pl": "2PL (ви / вы)",
        }) if "cell" in df.columns else ""
        out = OUT_DIR / "rq2_author_period_shifts.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("Saved %s  (%d rows)", out.name, len(df))

    # Population-level shifts
    if Q2_POPULATION.is_file():
        df = pd.read_csv(Q2_POPULATION, low_memory=False)
        df["cell_label"] = df["cell"].map({
            "1sg": "1SG (я / я)",
            "1pl": "1PL (ми / мы)",
            "2sg": "2SG (ти / ты)",
            "2pl": "2PL (ви / вы)",
        }) if "cell" in df.columns else ""
        out = OUT_DIR / "rq2_population_level_shifts.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("Saved %s  (%d rows)", out.name, len(df))


# --------------------------------------------------------------------------- #
# RQ3 exports
# --------------------------------------------------------------------------- #

def export_rq3_contexts() -> None:
    """Export full syntactic dependency contexts for RQ3."""
    if not CONTEXTS.is_file():
        log.warning("pronoun_contexts_long.csv not found; skipping.")
        return

    log.info("Loading pronoun syntactic contexts …")
    df = pd.read_csv(CONTEXTS, low_memory=False)

    # Readable labels
    df["period_label"] = df["period"].map(period_label) if "period" in df.columns else ""
    df["cell_label"] = df["cell"].map({
        "1sg": "1SG (я / я)",
        "1pl": "1PL (ми / мы)",
        "2sg": "2SG (ти / ты)",
        "2pl": "2PL (ви / вы)",
        "3sg": "3SG",
        "3pl": "3PL",
    }) if "cell" in df.columns else ""
    df["relation_readable"] = df["deprel"].map({
        "nsubj": "subject of verb",
        "obj":   "object of verb",
        "obl":   "oblique argument",
        "nmod":  "nominal modifier",
        "conj":  "conjunct",
    }).fillna(df.get("deprel", ""))

    df = df.sort_values(["language", "cell", "period", "head_lemma"]).reset_index(drop=True)

    out = OUT_DIR / "rq3_pronoun_syntactic_contexts.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("Saved %s  (%d rows)", out.name, len(df))


def export_rq3_collocations() -> None:
    """Export collocation scores and differential collocations for RQ3."""

    # Full collocation scores
    if COLLOCATIONS.is_file():
        df = pd.read_csv(COLLOCATIONS, low_memory=False)
        df["period_label"] = df["period"].map(period_label) if "period" in df.columns else ""
        df["cell_label"] = df["cell"].map({
            "1sg": "1SG (я / я)",
            "1pl": "1PL (ми / мы)",
            "2sg": "2SG (ти / ты)",
            "2pl": "2PL (ви / вы)",
        }) if "cell" in df.columns else ""
        df["relation_readable"] = df["deprel"].map({
            "nsubj": "subject of verb",
            "obj":   "object of verb",
            "obl":   "oblique argument",
            "nmod":  "nominal modifier",
            "conj":  "conjunct",
        }).fillna(df.get("deprel", "")) if "deprel" in df.columns else ""
        df = df.sort_values(
            ["language", "cell", "period", "log_dice"],
            ascending=[True, True, True, False]
        ).reset_index(drop=True)
        out = OUT_DIR / "rq3_collocation_scores_by_cell_period.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("Saved %s  (%d rows)", out.name, len(df))

    # Differential collocations
    if DIFFERENTIAL.is_file():
        df = pd.read_csv(DIFFERENTIAL, low_memory=False)
        df["cell_label"] = df["cell"].map({
            "1sg": "1SG (я / я)",
            "1pl": "1PL (ми / мы)",
            "2sg": "2SG (ти / ты)",
            "2pl": "2PL (ви / вы)",
        }) if "cell" in df.columns else ""
        df["shift_direction"] = df["delta_pmi"].apply(
            lambda x: "strengthened post-invasion" if float(x) > 0 else "weakened post-invasion"
            if pd.notna(x) else ""
        ) if "delta_pmi" in df.columns else ""
        out = OUT_DIR / "rq3_differential_collocations.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("Saved %s  (%d rows)", out.name, len(df))


def export_rq3_top_collocates_summary() -> None:
    """Build a readable top-20 collocates per language × cell × period table."""
    if not COLLOCATIONS.is_file():
        return

    df = pd.read_csv(COLLOCATIONS, low_memory=False)
    required = {"language", "cell", "period", "head_lemma", "log_dice", "cooccurrence"}
    if not required.issubset(df.columns):
        log.warning("collocations file missing expected columns; skipping summary.")
        return

    rows = []
    for (lang, cell, period), grp in df.groupby(["language", "cell", "period"]):
        top = grp.nlargest(20, "log_dice")[
            ["head_lemma", "deprel", "cooccurrence", "log_dice", "pmi"]
        ].copy()
        top.insert(0, "language", lang)
        top.insert(1, "cell", cell)
        top.insert(2, "cell_label", {
            "1sg": "1SG (я / я)", "1pl": "1PL (ми / мы)",
            "2sg": "2SG (ти / ты)", "2pl": "2PL (ви / вы)",
        }.get(str(cell), str(cell)))
        top.insert(3, "period", period)
        top.insert(4, "period_label", period_label(period))
        top.insert(5, "rank", range(1, len(top) + 1))
        rows.append(top)

    if rows:
        result = pd.concat(rows, ignore_index=True)
        result = result.sort_values(["language", "cell", "period", "rank"])
        out = OUT_DIR / "rq3_top_collocates_summary.csv"
        result.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("Saved %s  (%d rows)", out.name, len(result))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    log.info("=== RQ2 exports ===")
    export_rq2_author_counts()
    export_rq2_model_outputs()

    log.info("=== RQ3 exports ===")
    export_rq3_contexts()
    export_rq3_collocations()
    export_rq3_top_collocates_summary()

    log.info("Done. All files saved to: %s", OUT_DIR)
