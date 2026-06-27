"""Export poem-level audit tables organised by significant verb collocate (RQ3).

For each pronoun cell, produces ONE CSV where rows are poems grouped by verb,
so the advisor can filter to a single verb and compare its P1 vs P2 poems.

Verb selection per language × cell:
  - ALL BH survivors (q_bh < 0.10)
  - Plus TOP_EXTRA non-survivors with largest G² (for context)

Output: data/qualitative_corpus/rq3_poems_by_verb_{cell}.csv
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

DIFF     = ROOT / "data" / "rq_corpus_details" / "rq3_differential_collocations.csv"
CONTEXTS = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "pronoun_contexts_long.csv"
LAYER0   = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
ROSTER   = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"

CELLS      = ("1pl", "1sg", "2sg")
TOP_EXTRA  = 3    # non-BH verbs to include per language × cell
BH_CUTOFF  = 0.10

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
    "advcl":      "adverbial-clause-of",
    "conj":       "conjunct-of",
}

GLOSS: dict[str, str] = {
    "мусити":      "must (obligation)",
    "хотіти":      "want",
    "стати":       "become",
    "говорити":    "speak / say",
    "називати":    "name / call",
    "называть":    "name / call",
    "благодарный": "grateful",
    "казаться":    "seem",
    "написать":    "write (to smb.)",
    "написати":    "write (to smb.)",
    "делать":      "do / make",
    "робити":      "do / make",
    "любить":      "love",
    "любити":      "love",
    "шукати":      "seek",
    "могти":       "be able to / can",
    "залишитися":  "stay / remain",
    "повертатися": "return",
    "знати":       "know",
    "чекати":      "wait",
    "бачити":      "see",
    "думати":      "think",
    "казати":      "say",
    "прийти":      "come",
    "взяти":       "take",
    "мочь":        "be able to / can",
    "жить":        "live",
    "жити":        "live",
    "падать":      "fall",
    "смотреть":    "watch / look",
    "злиться":     "be angry",
    "строить":     "build",
    "ответить":    "answer",
    "остаться":    "stay / remain",
    "спишати":     "hurry",
    "стріляти":    "shoot",
    "забути":      "forget",
    "читати":      "read",
    "збиратися":   "gather / prepare",
    "їхати":       "go (by vehicle)",
    "відбуватися": "take place / happen",
}


def load_roster() -> set[str]:
    r = pd.read_csv(ROSTER, low_memory=False)
    r.columns = r.columns.str.lstrip("﻿")
    return set(r.loc[r["included"].astype(str).str.lower().isin(["true","1","yes"]), "author"].astype(str))


def load_full_texts() -> pd.DataFrame:
    df = pd.read_csv(LAYER0, low_memory=False,
                     usecols=["poem_id", "Poem full text (copy and paste)"])
    df.columns = df.columns.str.lstrip("﻿")
    df = df.rename(columns={"Poem full text (copy and paste)": "full_poem_text"})
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    return df.drop_duplicates("poem_id")


def load_contexts(roster: set[str]) -> pd.DataFrame:
    df = pd.read_csv(CONTEXTS, low_memory=False)
    df.columns = df.columns.str.lstrip("﻿")
    df = df[df["author"].isin(roster)].copy()
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df["period"]  = df["period"].astype(str).str.strip().map(PERIOD_MAP).fillna(df["period"])
    df["deprel_en"] = df["deprel"].map(DEPREL_EN).fillna(df["deprel"].astype(str))
    return df


def select_target_verbs(diff: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Return target verb × language rows for this cell."""
    sub = diff[diff["cell"] == cell].copy()
    sub["bh_survivor"] = sub["q_value_bh"].apply(
        lambda q: "yes" if pd.notna(q) and q < BH_CUTOFF else "no"
    )
    rows: list[pd.DataFrame] = []
    for lang in sub["language"].unique():
        lng = sub[sub["language"] == lang].copy()
        survivors  = lng[lng["bh_survivor"] == "yes"]
        non_surv   = lng[lng["bh_survivor"] == "no"].sort_values(
            "g2_period_contrast", ascending=False
        ).head(TOP_EXTRA)
        combined = pd.concat([survivors, non_surv]).drop_duplicates(
            subset=["head_lemma", "deprel"]
        )
        rows.append(combined)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def classify_shift(row: pd.Series) -> str:
    p1, p2 = int(row["cooc_2014_2021"] or 0), int(row["cooc_post_2022"] or 0)
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


def export_cell(
    cell: str,
    diff: pd.DataFrame,
    ctx: pd.DataFrame,
    full_texts: pd.DataFrame,
) -> None:
    targets = select_target_verbs(diff, cell)
    if targets.empty:
        log.warning("No target verbs for cell %s", cell)
        return

    all_rows: list[pd.DataFrame] = []

    for _, verb_row in targets.iterrows():
        lang      = verb_row["language"]
        lemma     = verb_row["head_lemma"]
        deprel    = verb_row["deprel"]
        g2        = verb_row["g2_period_contrast"]
        q_bh      = verb_row["q_value_bh"]
        bh_surv   = verb_row["bh_survivor"]
        gloss     = GLOSS.get(lemma, "")
        deprel_en = DEPREL_EN.get(deprel, deprel)
        shift_dir = classify_shift(verb_row)
        cooc_p1   = int(verb_row["cooc_2014_2021"] or 0)
        cooc_p2   = int(verb_row["cooc_post_2022"] or 0)

        # Filter contexts to this verb × cell × language
        mask = (
            (ctx["cell"]      == cell)   &
            (ctx["language"]  == lang)   &
            (ctx["head_lemma"]== lemma)  &
            (ctx["deprel"]    == deprel)
        )
        verb_ctx = ctx.loc[mask].copy()
        if verb_ctx.empty:
            log.debug("No context rows for %s %s [%s] %s", lang, cell, deprel, lemma)
            continue

        # Aggregate to one row per poem
        def _agg(grp: pd.DataFrame) -> pd.Series:
            forms = "; ".join(sorted(set(grp["pronoun_form"].dropna().astype(str))))
            sketch = " | ".join(
                f"st.{int(r['stanza_index'])}: {r['pronoun_form']}"
                f" [{r['deprel_en']}] → {r['head_lemma']}"
                for _, r in grp.sort_values("stanza_index").iterrows()
            )
            return pd.Series({
                "author":          grp["author"].iloc[0],
                "period":          grp["period"].iloc[0],
                "pronoun_forms":   forms,
                "verb_token_count": len(grp),
                "syntax_tokens":   sketch,
            })

        poem_agg = (
            verb_ctx.groupby("poem_id", group_keys=False)
            .apply(_agg)
            .reset_index()
        )

        # Merge year from contexts (take first available)
        year_map = (
            verb_ctx.drop_duplicates("poem_id")
            .set_index("poem_id")["stanza_index"]  # placeholder — get year from layer0
        )
        # year is not in contexts; merge from full_texts via poem_id would need layer0
        # Instead carry year from the layer0 join below

        # Attach full poem text
        poem_agg = poem_agg.merge(full_texts[["poem_id", "full_poem_text"]], on="poem_id", how="left")

        # Add verb metadata columns
        poem_agg["language_stratum"]       = lang
        poem_agg["cell"]                   = cell
        poem_agg["verb_lemma"]             = lemma
        poem_agg["english_gloss"]          = gloss
        poem_agg["syntactic_role"]         = deprel_en
        poem_agg["shift_direction"]        = shift_dir
        poem_agg["corpus_count_P1"]        = cooc_p1
        poem_agg["corpus_count_P2"]        = cooc_p2
        poem_agg["G2_statistic"]           = round(g2, 2)
        poem_agg["q_bh_corrected"]         = round(q_bh, 4) if pd.notna(q_bh) else ""
        poem_agg["bh_survivor"]            = bh_surv

        all_rows.append(poem_agg)

    if not all_rows:
        log.warning("No poems found for cell %s", cell)
        return

    out = pd.concat(all_rows, ignore_index=True)

    # Final column order
    out = out[[
        "bh_survivor", "G2_statistic", "language_stratum", "cell",
        "verb_lemma", "english_gloss", "syntactic_role",
        "shift_direction", "corpus_count_P1", "corpus_count_P2",
        "period", "author", "poem_id",
        "pronoun_forms", "verb_token_count", "syntax_tokens",
        "full_poem_text",
    ]]

    # Sort: BH survivors first, then by G² desc, then period, year implicitly by poem_id
    out["_bh_sort"] = (out["bh_survivor"] == "yes").astype(int)
    out = out.sort_values(
        ["_bh_sort", "G2_statistic", "language_stratum", "verb_lemma", "period", "author"],
        ascending=[False, False, True, True, True, True],
    ).drop(columns=["_bh_sort"])

    out_path = OUT / f"rq3_poems_by_verb_{cell}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_verbs  = out[["language_stratum","verb_lemma","syntactic_role"]].drop_duplicates().__len__()
    n_poems  = out["poem_id"].nunique()
    n_bh     = (out["bh_survivor"] == "yes").sum()
    log.info(
        "Saved %-36s  %2d verbs, %3d poems (%d BH-survivor rows)",
        out_path.name, n_verbs, n_poems, n_bh,
    )


def main() -> None:
    roster     = load_roster()
    full_texts = load_full_texts()
    ctx        = load_contexts(roster)
    diff       = pd.read_csv(DIFF, low_memory=False)
    diff.columns = diff.columns.str.lstrip("﻿")

    for cell in CELLS:
        export_cell(cell, diff, ctx, full_texts)

    log.info("Done. Files in: %s", OUT)


if __name__ == "__main__":
    main()
