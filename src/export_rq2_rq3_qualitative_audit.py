"""Export lean qualitative audit CSVs for advisor review (RQ2 and RQ3).

Deliverable A  →  data/qualitative_corpus/rq2_audit_{cell}_poems.csv
                  data/qualitative_corpus/rq2_author_shift_summary_{cell}.csv
Deliverable B  →  data/qualitative_corpus/rq3_audit_{cell}_poems.csv

One row per poem.  Encoding: UTF-8 with BOM (Excel-safe).
Target cells: 1pl, 1sg, 2sg.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "data" / "qualitative_corpus"
OUT.mkdir(parents=True, exist_ok=True)

# ── Input paths ───────────────────────────────────────────────────────────────
ANNOTATION = ROOT / "data" / "Annotated_Source" / "pronoun_annotation_v2.csv"
LAYER0     = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
ROSTER     = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
Q2_SLOPES  = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_author_random_slope_summaries.csv"
CONTEXTS   = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "pronoun_contexts_long.csv"

CELLS = ("1pl", "1sg", "2sg")

# Poets explicitly named in the paper (fig_narr_2 highlighted set + text-named
# negative-1pl cluster).  The qualitative audit uses only these authors so the
# advisor is not presented with 33 largely undiscussed poets.
NAMED_AUTHORS: set[str] = {
    # Eight highlighted in fig_narr_2 (author trajectories) and/or case-study panels
    "Ihor Mitrov",
    "Iya Kiva",
    "Yaryna Chornohuz",
    "Serhiy Zhadan",
    "Boris Khersonsky",
    "Hryhoryi Falkovych",
    "Elizaveta Zharikova",
    "Halyna Kruk",
    # Two additional poets named in body text for the negative-1pl cluster
    "Kateryna Babkina",
    "Ivan Andrusiak",
}

PERIOD_MAP = {
    "2014_2021":    "P1",
    "P1_2014_2021": "P1",
    "post_2022":    "P2",
    "P2_2022_plus": "P2",
}

DEPREL_EN = {
    "nsubj":      "subject-of",
    "nsubj:pass": "subject-of (passive)",
    "obj":        "object-of",
    "iobj":       "indirect-object-of",
    "obl":        "oblique-with",
    "nmod":       "modifier-with",
    "nmod:poss":  "possessor-of",
    "root":       "root",
    "ccomp":      "clausal-complement-of",
    "xcomp":      "open-clausal-complement-of",
    "advcl":      "adverbial-clause-of",
    "conj":       "conjunct-of",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_annotation() -> pd.DataFrame:
    df = pd.read_csv(ANNOTATION, low_memory=False, on_bad_lines="skip")
    df.columns = df.columns.str.lstrip("﻿")
    # Drop repeat posts and translations
    for flag in ("is_repeat", "is_translation"):
        if flag in df.columns:
            df = df[~df[flag].astype(str).str.lower().isin(["true", "1", "yes"])]
    # Normalise person / number → cell
    df["person"] = (
        df["person"].fillna("").astype(str).str.strip()
        .replace({"1st": "1", "2nd": "2", "3rd": "3"})
    )
    df["number"] = (
        df["number"].fillna("").astype(str).str.strip()
        .replace({"Singular": "sg", "Plural": "pl"})
    )
    df["cell"] = (df["person"] + df["number"]).str.lower()
    # Normalise period
    df["period"] = df["temporal_period"].astype(str).str.strip().map(PERIOD_MAP)
    df = df[df["period"].isin(["P1", "P2"])].copy()
    df["year"]    = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    return df


def load_roster() -> set[str]:
    r = pd.read_csv(ROSTER, low_memory=False)
    r.columns = r.columns.str.lstrip("﻿")
    return set(
        r.loc[r["included"].astype(str).str.lower().isin(["true", "1", "yes"]), "author"].astype(str)
    )


def load_full_texts() -> pd.DataFrame:
    df = pd.read_csv(
        LAYER0, low_memory=False,
        usecols=["poem_id", "Poem full text (copy and paste)"],
    )
    df.columns = df.columns.str.lstrip("﻿")
    df = df.rename(columns={"Poem full text (copy and paste)": "full_poem_text"})
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    return df.drop_duplicates("poem_id")


def load_shifts() -> dict[str, pd.DataFrame]:
    """Return {cell: per-author shift DataFrame} with direction and strength columns."""
    if not Q2_SLOPES.is_file():
        log.warning("q2_author_random_slope_summaries.csv not found; shift columns will be empty.")
        return {}
    raw = pd.read_csv(Q2_SLOPES, low_memory=False)
    raw.columns = raw.columns.str.lstrip("﻿")

    result: dict[str, pd.DataFrame] = {}
    for cell in CELLS:
        df = raw[raw["cell"] == cell].copy()
        # Prefer pooled stratum; fall back to any available row
        preferred = "pooled_Ukrainian_Russian"
        pooled = df[df["language_stratum"] == preferred]
        others = df[~df["author"].isin(pooled["author"])]
        df = pd.concat([pooled, others], ignore_index=True).drop_duplicates(subset=["author"])
        df = df[[
            "author",
            "author_total_period_shift_mean_log_mu",
            "author_total_period_shift_rate_ratio_mean",
            "author_total_period_shift_p_direction_gt0",
        ]].rename(columns={
            "author_total_period_shift_mean_log_mu":     "shift_log_mu",
            "author_total_period_shift_rate_ratio_mean": "shift_rate_ratio",
            "author_total_period_shift_p_direction_gt0": "p_increase",
        })

        # shift_direction (plain English)
        df["author_shift_direction"] = df["shift_log_mu"].apply(
            lambda x: "increased" if pd.notna(x) and x > 0.1
            else ("decreased" if pd.notna(x) and x < -0.1 else "stable")
        )

        # shift_strength: compact phrase with directional probability
        def _strength(row: pd.Series) -> str:
            if pd.isna(row["shift_log_mu"]):
                return "unknown"
            direction = row["author_shift_direction"]
            if direction == "stable":
                return f"stable (p_increase={row['p_increase']:.2f})"
            # Use directional probability: p_increase for "increased", 1-p_increase for "decreased"
            p_dir = row["p_increase"] if direction == "increased" else 1.0 - row["p_increase"]
            label = "increase" if direction == "increased" else "decrease"
            strength = "strong" if p_dir >= 0.90 else ("moderate" if p_dir >= 0.75 else "weak")
            return f"{strength} {label} (p={p_dir:.2f})"

        df["author_shift_strength"] = df.apply(_strength, axis=1)
        result[cell] = df
    return result


def load_contexts() -> pd.DataFrame:
    if not CONTEXTS.is_file():
        log.warning("pronoun_contexts_long.csv not found; RQ3 parse columns will be blank.")
        return pd.DataFrame()
    df = pd.read_csv(CONTEXTS, low_memory=False)
    df.columns = df.columns.str.lstrip("﻿")
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df["period"]  = df["period"].astype(str).str.strip().map(PERIOD_MAP).fillna(df["period"])
    df["deprel_en"] = df["deprel"].map(DEPREL_EN).fillna(df["deprel"].astype(str))
    return df


# ── Shared helper ─────────────────────────────────────────────────────────────

def _join_sorted(s: pd.Series) -> str:
    vals = sorted(set(s.dropna().astype(str)))
    return "; ".join(vals) if vals else ""


# ── Deliverable A: RQ2 poem-level audit ──────────────────────────────────────

def export_rq2(
    ann: pd.DataFrame,
    cell: str,
    roster: set[str],
    full_texts: pd.DataFrame,
    shifts: dict[str, pd.DataFrame],
) -> None:
    target = roster & NAMED_AUTHORS
    sub = ann[(ann["cell"] == cell) & (ann["author"].isin(target))].copy()

    poem = (
        sub.groupby("poem_id", as_index=False)
        .agg(
            author       = ("author",       "first"),
            year         = ("year",         "first"),
            period       = ("period",        "first"),
            pronoun_count= ("pronoun_word",  "count"),
            pronoun_forms= ("pronoun_word",  lambda s: "; ".join(sorted(set(s.dropna().astype(str))))),
        )
    )
    poem["cell"] = cell

    # Merge shift info
    shift_df = shifts.get(cell, pd.DataFrame())
    if not shift_df.empty:
        poem = poem.merge(
            shift_df[["author", "author_shift_direction", "author_shift_strength"]],
            on="author", how="left",
        )
    else:
        poem["author_shift_direction"] = ""
        poem["author_shift_strength"]  = ""

    poem["author_shift_direction"] = poem["author_shift_direction"].fillna("no model estimate")
    poem["author_shift_strength"]  = poem["author_shift_strength"].fillna("no model estimate")

    # Merge full poem text
    if not full_texts.empty:
        poem = poem.merge(full_texts[["poem_id", "full_poem_text"]], on="poem_id", how="left")
    else:
        poem["full_poem_text"] = ""
    poem["full_poem_text"] = poem["full_poem_text"].fillna("")

    poem = poem[[
        "author", "poem_id", "year", "period", "cell",
        "pronoun_count", "pronoun_forms",
        "author_shift_direction", "author_shift_strength",
        "full_poem_text",
    ]].sort_values(["author", "period", "year", "poem_id"])

    out_path = OUT / f"rq2_audit_{cell}_poems.csv"
    poem.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(
        "Saved %-36s  %4d poems, %d authors",
        out_path.name, len(poem), poem["author"].nunique(),
    )


# ── Companion: author shift summary ──────────────────────────────────────────

def export_author_summary(
    ann: pd.DataFrame,
    cell: str,
    roster: set[str],
    shifts: dict[str, pd.DataFrame],
) -> None:
    shift_df = shifts.get(cell, pd.DataFrame())
    if shift_df.empty:
        log.warning("No shift data for %s; skipping author summary.", cell)
        return

    target = roster & NAMED_AUTHORS
    sub = ann[(ann["cell"] == cell) & (ann["author"].isin(target))].copy()

    # Poem and token counts per author × period
    counts = (
        sub.groupby(["author", "period"])
        .agg(poems=("poem_id", "nunique"), tokens=("pronoun_word", "count"))
        .unstack("period")
        .fillna(0)
        .astype(int)
    )
    # Flatten MultiIndex columns: (poems, P1) → poems_P1
    counts.columns = [f"{metric}_{p}" for metric, p in counts.columns]
    counts = counts.reset_index()

    summary = shift_df[
        shift_df["author"].isin(target)
    ][["author", "author_shift_direction", "shift_rate_ratio", "p_increase"]].merge(
        counts, on="author", how="left"
    )
    summary["cell"] = cell

    # Guarantee column existence even if a period is absent in data
    for col in ("poems_P1", "poems_P2", "tokens_P1", "tokens_P2"):
        if col not in summary.columns:
            summary[col] = 0

    summary = summary[[
        "author", "cell", "author_shift_direction",
        "shift_rate_ratio", "p_increase",
        "poems_P1", "poems_P2", "tokens_P1", "tokens_P2",
    ]].sort_values("shift_rate_ratio", ascending=False)

    out_path = OUT / f"rq2_author_shift_summary_{cell}.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Saved %-36s  %4d authors", out_path.name, len(summary))


# ── Deliverable B: RQ3 poem-level parse audit ─────────────────────────────────

def _make_syntax_sketch(grp: pd.DataFrame) -> str:
    """Compact per-token annotation: st.N: form [role] → lemma."""
    tokens = []
    for _, row in grp.sort_values("stanza_index").iterrows():
        tokens.append(
            f"st.{int(row['stanza_index'])}: {row['pronoun_form']}"
            f" [{row['deprel_en']}] → {row['head_lemma']}"
        )
    return " | ".join(tokens)


def export_rq3(
    ann: pd.DataFrame,
    cell: str,
    roster: set[str],
    full_texts: pd.DataFrame,
    contexts: pd.DataFrame,
) -> None:
    target = roster & NAMED_AUTHORS
    sub = ann[(ann["cell"] == cell) & (ann["author"].isin(target))].copy()

    # Base: all poems with ≥1 pronoun token for this cell
    all_poems = (
        sub.groupby("poem_id", as_index=False)
        .agg(
            author       = ("author",      "first"),
            year         = ("year",        "first"),
            period       = ("period",       "first"),
            pronoun_count= ("pronoun_word", "count"),
        )
    )
    all_poems["cell"] = cell

    # Build parse aggregation if contexts are available
    if not contexts.empty:
        ctx = contexts[
            (contexts["cell"] == cell) & (contexts["author"].isin(target))
        ].copy()
        parsed_ids = set(ctx["poem_id"].unique())

        def _agg_parse(grp: pd.DataFrame) -> pd.Series:
            nsubj  = grp.loc[grp["deprel"].isin(("nsubj", "nsubj:pass")), "head_lemma"]
            obj    = grp.loc[grp["deprel"].eq("obj"),                      "head_lemma"]
            obl    = grp.loc[grp["deprel"].isin(("obl", "nmod", "nmod:poss")), "head_lemma"]
            return pd.Series({
                "verbs_as_subject": _join_sorted(nsubj),
                "verbs_as_object":  _join_sorted(obj),
                "other_collocates": _join_sorted(obl),
                "syntax_sketch":    _make_syntax_sketch(grp),
            })

        if not ctx.empty:
            parse_agg = ctx.groupby("poem_id", group_keys=False).apply(_agg_parse).reset_index()
            parse_agg["parse_available"] = "yes"
        else:
            parse_agg = pd.DataFrame(columns=["poem_id", "parse_available"])

        poem = all_poems.merge(parse_agg, on="poem_id", how="left")
    else:
        poem = all_poems.copy()
        parse_agg = pd.DataFrame()

    poem["parse_available"] = poem.get("parse_available", pd.Series(dtype=str)).fillna("no")
    for col in ("verbs_as_subject", "verbs_as_object", "other_collocates", "syntax_sketch"):
        if col not in poem.columns:
            poem[col] = ""
        poem[col] = poem[col].fillna("")

    # Merge full poem text
    if not full_texts.empty:
        poem = poem.merge(full_texts[["poem_id", "full_poem_text"]], on="poem_id", how="left")
    else:
        poem["full_poem_text"] = ""
    poem["full_poem_text"] = poem["full_poem_text"].fillna("")

    poem = poem[[
        "author", "poem_id", "year", "period", "cell",
        "pronoun_count",
        "verbs_as_subject", "verbs_as_object", "other_collocates",
        "syntax_sketch", "full_poem_text", "parse_available",
    ]].sort_values(["author", "period", "year", "poem_id"])

    out_path = OUT / f"rq3_audit_{cell}_poems.csv"
    poem.to_csv(out_path, index=False, encoding="utf-8-sig")
    parsed_count = (poem["parse_available"] == "yes").sum()
    log.info(
        "Saved %-36s  %4d poems (%d with parse, %d without)",
        out_path.name, len(poem), parsed_count, len(poem) - parsed_count,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading shared inputs …")
    ann        = load_annotation()
    roster     = load_roster()
    full_texts = load_full_texts()
    shifts     = load_shifts()
    contexts   = load_contexts()

    log.info(
        "Roster: %d authors | Annotation rows: %d | Context rows: %d",
        len(roster), len(ann), len(contexts),
    )

    for cell in CELLS:
        log.info("=== Cell: %s ===", cell.upper())
        export_rq2(ann, cell, roster, full_texts, shifts)
        export_author_summary(ann, cell, roster, shifts)
        # RQ3 is now verb-centric (not poem-level); see export_rq3_verb_shifts.py

    log.info("Done. Output directory: %s", OUT)


if __name__ == "__main__":
    main()
