"""Export verb-centric RQ3 shift tables for advisor review.

For each pronoun cell (1pl, 1sg, 2sg), produces ONE CSV showing the most
significant pre-war → post-war shifts in the verbs (and major nominal heads)
that each pronoun governs or is governed by.

Source: data/rq_corpus_details/rq3_differential_collocations.csv
Output: data/qualitative_corpus/rq3_verb_shifts_{cell}.csv

Rows: one per (language × verb × syntactic role).
Sorted by: language, G² descending.
Encoding: UTF-8 with BOM.
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

DIFF = ROOT / "data" / "rq_corpus_details" / "rq3_differential_collocations.csv"

CELLS = ("1pl", "1sg", "2sg")

# Top-N verbs to show per language × cell (all BH survivors are always included)
TOP_N = 15

# English glosses for lemmas that appear in the paper or are high-frequency
GLOSS: dict[str, str] = {
    # BH survivors and paper-cited verbs
    "мусити":       "must (obligation)",
    "хотіти":       "want",
    "стати":        "become",
    "говорити":     "speak / say",
    "називати":     "name / call",
    "называть":     "name / call",
    "благодарный":  "grateful",
    "казаться":     "seem",
    "казатися":     "seem",
    "написать":     "write (to smb.)",
    "написати":     "write (to smb.)",
    "делать":       "do / make",
    "робити":       "do / make",
    "любить":       "love",
    "любити":       "love",
    # Other directional movers cited in paper
    "знати":        "know",
    "чекати":       "wait",
    "прийти":       "come",
    "бачити":       "see",
    "взяти":        "take",
    "шукати":       "seek",
    "думати":       "think",
    "казати":       "say",
    "жити":         "live",
    "жить":         "live",
    "хотеть":       "want",
    "знать":        "know",
    "видеть":       "see",
    "остаться":     "stay / remain",
    "залишитися":   "stay / remain",
    "іти":          "go",
    "йти":          "go",
    "ждати":        "wait",
    "вміти":        "be able to",
    "мочь":         "be able to",
    "дати":         "give",
    "дать":         "give",
    "боятися":      "fear / be afraid",
    "боятья":       "fear / be afraid",
    "писати":       "write",
    "писать":       "write",
    "пам'ятати":    "remember",
    "помнить":      "remember",
    "сказати":      "say (pfv.)",
    "сказать":      "say (pfv.)",
    "вернутися":    "return",
    "вернуться":    "return",
    "зрозуміти":    "understand",
    "понять":       "understand",
    "стояти":       "stand",
    "стоять":       "stand",
    "ответить":     "answer",
    "відповісти":   "answer",
    "вбити":        "kill",
    "убить":        "kill",
    "плакати":      "weep / cry",
    "плакать":      "weep / cry",
    "вижити":       "survive",
    "выжить":       "survive",
    "захистити":    "defend / protect",
    "защитить":     "defend / protect",
}

DEPREL_EN: dict[str, str] = {
    "nsubj":      "subject-of",
    "nsubj:pass": "subject-of (passive)",
    "obj":        "object-of",
    "iobj":       "indirect-object-of",
    "obl":        "oblique-with",
    "nmod":       "modifier-with",
    "nmod:poss":  "possessor-of",
    "root":       "root",
    "ccomp":      "clausal-complement-of",
    "xcomp":      "complement-of",
    "advcl":      "adverbial-clause-of",
    "conj":       "conjunct-of",
}


def classify_shift(row: pd.Series) -> str:
    p1, p2 = row["count_P1"], row["count_P2"]
    if p1 == 0 and p2 > 0:
        return "new post-war (absent pre-war)"
    if p1 > 0 and p2 == 0:
        return "disappeared post-war"
    raw = str(row.get("shift_direction", "")).strip()
    if "strengthened" in raw:
        return "strengthened post-war"
    if "weakened" in raw:
        return "weakened post-war"
    return "mixed / unclear"


def export_cell(diff: pd.DataFrame, cell: str) -> None:
    sub = diff[diff["cell"] == cell].copy()
    if sub.empty:
        log.warning("No data for cell %s; skipping.", cell)
        return

    # ── Tidy columns ──────────────────────────────────────────────────────────
    sub = sub.rename(columns={
        "cooc_2014_2021":       "count_P1",
        "cooc_post_2022":       "count_P2",
        "g2_period_contrast":   "G2",
        "q_value_bh":           "q_bh",
    })
    sub["count_P1"] = sub["count_P1"].fillna(0).astype(int)
    sub["count_P2"] = sub["count_P2"].fillna(0).astype(int)
    sub["count_change"] = sub["count_P2"] - sub["count_P1"]
    sub["shift_direction"] = sub.apply(classify_shift, axis=1)
    sub["bh_survivor"] = sub["q_bh"].apply(
        lambda q: "yes" if pd.notna(q) and q < 0.10 else "no"
    )
    sub["english_gloss"] = sub["head_lemma"].map(GLOSS).fillna("")
    sub["syntactic_role"] = sub["deprel"].map(DEPREL_EN).fillna(sub["deprel"])

    # ── Select rows: all BH survivors + top-N by G² per language ──────────────
    rows: list[pd.DataFrame] = []
    for lang in ["Ukrainian", "Russian"]:
        lng = sub[sub["language"] == lang].copy()
        if lng.empty:
            continue
        survivors = lng[lng["bh_survivor"] == "yes"]
        top = lng.sort_values("G2", ascending=False).head(TOP_N)
        combined = pd.concat([survivors, top]).drop_duplicates(
            subset=["head_lemma", "deprel"]
        ).sort_values("G2", ascending=False)
        rows.append(combined)

    if not rows:
        return
    out_df = pd.concat(rows, ignore_index=True)

    # ── Final columns ──────────────────────────────────────────────────────────
    out_df = out_df[[
        "language", "cell", "head_lemma", "english_gloss", "syntactic_role",
        "count_P1", "count_P2", "count_change",
        "shift_direction", "G2", "q_bh", "bh_survivor",
    ]].rename(columns={
        "language":      "language_stratum",
        "head_lemma":    "verb_lemma",
        "count_P1":      "count_prewar_P1",
        "count_P2":      "count_wartime_P2",
        "count_change":  "change_P2_minus_P1",
        "G2":            "G2_statistic",
        "q_bh":          "q_bh_corrected",
    })

    out_path = OUT / f"rq3_verb_shifts_{cell}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_surv = (out_df["bh_survivor"] == "yes").sum()
    log.info(
        "Saved %-32s  %3d verbs (%d BH survivors) across %s",
        out_path.name,
        len(out_df),
        n_surv,
        ", ".join(out_df["language_stratum"].unique()),
    )


def main() -> None:
    if not DIFF.is_file():
        log.error("Missing: %s  —  run export_rq2_rq3_corpus_details.py first.", DIFF)
        return

    diff = pd.read_csv(DIFF, low_memory=False)
    diff.columns = diff.columns.str.lstrip("﻿")

    for cell in CELLS:
        export_cell(diff, cell)

    log.info("Done. Output: %s", OUT)


if __name__ == "__main__":
    main()
