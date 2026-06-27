"""Re-filter ego-edge tables to content-word neighbours only.

The original ``02_modeling_pronoun_cooccurrence.py`` used a small closed-class
stoplist that left many high-frequency function words (так, ще, від, де, бо,
все, …) in the top neighbours. This is a known limitation of regex tokenisation
plus a hand-curated stoplist on Slavic languages.

This follow-up:
1. Loads ``ego_edges.csv`` from that pipeline.
2. Lemmatises the neighbour column with Stanza (uk / ru) to canonicalise
   morphology.
3. Drops any neighbour whose POS is not NOUN / VERB / ADJ / PROPN (content
   words), removing pronouns, particles, conjunctions, adverbs of degree,
   and prepositions in one pass.
4. Re-ranks per (language, focal, period) by log-Dice and saves the cleaned
   table plus a small markdown comparison report.

Output: ``outputs/02_modeling_pronoun_cooccurrence_content_filtered/``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "outputs" / "02_modeling_pronoun_cooccurrence" / "ego_edges.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_cooccurrence_content_filtered"

CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "PROPN"}


def _build_lemma_table(words: list[str], lang_code: str) -> dict[str, tuple[str, str]]:
    """Return {surface: (lemma, upos)} for every unique surface form."""
    import stanza

    pipe = stanza.Pipeline(
        lang=lang_code,
        processors="tokenize,pos,lemma",
        download_method=None,
        verbose=False,
        tokenize_pretokenized=True,
    )
    out: dict[str, tuple[str, str]] = {}
    batch_size = 128
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        doc = pipe([[w] for w in batch])
        for w_surface, sentence in zip(batch, doc.sentences):
            if not sentence.words:
                continue
            tok = sentence.words[0]
            lemma = (tok.lemma or w_surface).lower()
            upos = tok.upos or "X"
            out[w_surface] = (lemma, upos)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    log.info("Loaded %d ego edges", len(df))

    lemma_tables: dict[str, dict[str, tuple[str, str]]] = {}
    for lang, code in (("Ukrainian", "uk"), ("Russian", "ru")):
        words = sorted(df.loc[df["language"].eq(lang), "neighbour"].astype(str).unique())
        log.info("%s: %d unique neighbour surface forms to lemmatise", lang, len(words))
        lemma_tables[lang] = _build_lemma_table(words, code)

    rows = []
    for _, r in df.iterrows():
        lang = r["language"]
        surface = str(r["neighbour"])
        lemma, upos = lemma_tables[lang].get(surface, (surface, "X"))
        rows.append(
            {
                **{k: r[k] for k in df.columns},
                "neighbour_lemma": lemma,
                "neighbour_upos": upos,
            }
        )
    enriched = pd.DataFrame(rows)
    enriched.to_csv(out_dir / "ego_edges_enriched.csv", index=False)

    content = enriched.loc[enriched["neighbour_upos"].isin(CONTENT_UPOS)].copy()
    log.info(
        "Kept %d / %d edges after content-POS filter", len(content), len(enriched)
    )

    # Collapse to lemma level (sum cooccurrence; recompute log-Dice from there).
    grouped = (
        content.groupby(
            ["language", "focal", "period", "neighbour_lemma", "neighbour_upos"],
            as_index=False,
        )
        .agg(cooccurrence=("cooccurrence", "sum"), log_dice=("log_dice", "max"))
    )
    grouped = grouped.sort_values(["language", "focal", "period", "log_dice"], ascending=[True, True, True, False])
    grouped.to_csv(out_dir / "ego_edges_content_lemmas.csv", index=False)

    md_lines = ["# Content-word ego neighbours (top-k by log-Dice)\n"]
    for (lang, focal, period), grp in grouped.groupby(["language", "focal", "period"]):
        md_lines.append(f"## {lang} · {focal} · {period}\n")
        top = grp.nlargest(args.top_k, "log_dice")[
            ["neighbour_lemma", "neighbour_upos", "cooccurrence", "log_dice"]
        ]
        md_lines.append(top.to_markdown(index=False, floatfmt=".3f"))
        md_lines.append("\n")
    (out_dir / "top_content_neighbours.md").write_text("\n".join(md_lines), encoding="utf-8")

    log.info("Wrote content-filtered outputs to %s", out_dir)


if __name__ == "__main__":
    main()
