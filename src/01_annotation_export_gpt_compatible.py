"""Map full-corpus v2 token detections to the downstream GPT-compatible schema.

Reads ``data/Annotated_Source/tokens_v2_full.csv`` (produced by
``01_annotation_source_pronoun_detection.py --full --mode v2``) and the current
canonical GPT annotation (for poem-level metadata the token table does not carry:
``stanza_ukr`` source text, ``is_repeat``, ``is_translation``), and writes

    data/Annotated_Source/pronoun_annotation_v2.csv

with the same columns the modeling/reporting pipeline already consumes.

Mapping rules (see docs/methodology_source_pronoun_detection.md Part C/D):
  * keep only rows with a resolved cell (``cell in {1sg,1pl,2sg,2pl}``); every
    cell-bearing v2 detection is confident, so ``qa_flag`` is set to ``OK`` (passes
    the strict ``qa_clean`` filter). The original v2 flag is kept in ``v2_qa_flag``.
  * ``person``/``number`` already use the GPT encoding (``1st/2nd``,
    ``Singular/Plural``).
  * ``source_mapping`` = lemma; pro-drop rows append ``" (IMPLIED)"`` so
    ``explicit_source_mapping`` and the IMPLIED convention stay consistent.
  * ``vy_register`` = "" for all rows: v2 does not infer politeness, so every
    ``ви/вы`` folds into ``2pl_vy_true_plural`` downstream (polite-singular = 0).
  * English-only columns (``stanza_en``, ``full_shakespeare_text``,
    ``pronoun_word``) are emitted empty so no downstream script KeyErrors.

Run::

    PYTHONPATH=src python src/01_annotation_export_gpt_compatible.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.stage_io import write_csv_artifact
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend=None)

TOKENS_CSV = ROOT / "data" / "Annotated_Source" / "tokens_v2_full.csv"
CANONICAL_GPT_CSV = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
OUT_CSV = ROOT / "data" / "Annotated_Source" / "pronoun_annotation_v2.csv"

PRIMARY_CELLS = ("1sg", "1pl", "2sg", "2pl")

# Output column order: matches the rerun GPT table, then v2 audit extras.
GPT_SCHEMA_COLUMNS = [
    "poem_id", "author", "language", "year", "temporal_period",
    "is_repeat", "is_translation", "stanza_index", "stanza_ukr",
    "stanza_en", "full_shakespeare_text", "pronoun_word",
    "person", "number", "vy_register", "is_pro_drop", "source_mapping", "qa_flag",
]
V2_EXTRA_COLUMNS = [
    "detection_method", "syntactic_role", "verb_tense", "promotion_rule",
    "lemma", "surface_form", "v2_qa_flag",
]


def load_metadata(gpt_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stanza-level stanza_ukr, poem-level is_repeat/is_translation)."""
    cols = ["poem_id", "stanza_index", "stanza_ukr", "is_repeat", "is_translation"]
    df = pd.read_csv(gpt_csv, usecols=cols, low_memory=False)
    df["poem_id"] = df["poem_id"].astype(str)
    df["stanza_index"] = pd.to_numeric(df["stanza_index"], errors="coerce")
    stanza_text = (
        df.dropna(subset=["stanza_index"])
        .assign(stanza_index=lambda d: d["stanza_index"].astype(int))
        .drop_duplicates(["poem_id", "stanza_index"])[["poem_id", "stanza_index", "stanza_ukr"]]
    )
    poem_flags = (
        df.groupby("poem_id")[["is_repeat", "is_translation"]]
        .first()
        .reset_index()
    )
    return stanza_text, poem_flags


def build_export(tokens: pd.DataFrame, stanza_text: pd.DataFrame,
                 poem_flags: pd.DataFrame) -> pd.DataFrame:
    t = tokens.copy()
    t["cell"] = t["cell"].astype(str)
    t = t[t["cell"].isin(PRIMARY_CELLS)].copy()
    t["poem_id"] = t["poem_id"].astype(str)
    t["stanza_index"] = pd.to_numeric(t["stanza_index"], errors="coerce").astype("Int64")

    is_drop = t["is_pro_drop"].astype(str).str.lower().isin(["true", "1", "1.0"])
    lemma = t["lemma"].fillna("").astype(str)
    t["source_mapping"] = lemma.where(~is_drop, lemma + " (IMPLIED)")
    t["v2_qa_flag"] = t["qa_flag"]
    t["qa_flag"] = "OK"
    t["vy_register"] = ""
    t["is_pro_drop"] = is_drop
    t["stanza_en"] = ""
    t["full_shakespeare_text"] = ""
    # pronoun_word carries the detected source surface token. It must be non-empty:
    # downstream filters (e.g. 02a) use ``pronoun_word.notna()`` to select pronoun
    # rows, and an empty string is read back from CSV as NaN.
    t["pronoun_word"] = t["surface_form"].fillna("").astype(str)
    t.loc[t["pronoun_word"].eq(""), "pronoun_word"] = t["lemma"].fillna("").astype(str)

    t = t.merge(stanza_text, on=["poem_id", "stanza_index"], how="left")
    t = t.merge(poem_flags, on="poem_id", how="left")
    for c in ("is_repeat", "is_translation"):
        t[c] = t[c].fillna(False)

    for c in GPT_SCHEMA_COLUMNS + V2_EXTRA_COLUMNS:
        if c not in t.columns:
            t[c] = ""
    return t[GPT_SCHEMA_COLUMNS + V2_EXTRA_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=Path, default=TOKENS_CSV)
    ap.add_argument("--gpt", type=Path, default=CANONICAL_GPT_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    print(f"[export] reading v2 tokens: {args.tokens}")
    tokens = pd.read_csv(args.tokens, low_memory=False)
    print(f"[export] {len(tokens)} token rows; reading metadata from {args.gpt.name}")
    stanza_text, poem_flags = load_metadata(args.gpt)

    out = build_export(tokens, stanza_text, poem_flags)
    write_csv_artifact(out, args.out)

    n_stanza = out[["poem_id", "stanza_index"]].drop_duplicates().shape[0]
    n_poems = out["poem_id"].nunique()
    miss_text = out["stanza_ukr"].isna().sum()
    print(f"[export] wrote {len(out)} rows -> {args.out}")
    print(f"[export] poems={n_poems} stanzas={n_stanza} "
          f"langs={out['language'].value_counts().to_dict()}")
    print(f"[export] cell dist: {out.assign(cell=out['source_mapping']).groupby(['person','number']).size().to_dict()}")
    print(f"[export] pro_drop share: {out['is_pro_drop'].mean():.3f}; missing stanza_ukr: {miss_text}")


if __name__ == "__main__":
    main()
