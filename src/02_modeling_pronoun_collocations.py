"""Dependency-parsed pronoun collocation pipeline (P1-A + P1-B).

For each Ukrainian and Russian stanza in the annotated corpus, run Stanza's
dependency parser, identify every personal-pronoun token, and emit a
``(pronoun_cell, deprel, head_lemma)`` tuple. Aggregating these tuples by
``(cell, period, language)`` yields the corpus's syntactic context for the
1sg / 1pl / 2sg / 2pl_vy_true_plural cells; aggregating across periods yields
period-contrastive (``differential'') collocations.

Statistical measures
--------------------
For a given ``(cell, language)`` slice and a head-lemma ``w``:

* **Frequency** ``f_w`` = number of times ``w`` is the dependency head of a
  cell-pronoun token in the slice.
* **log-Dice** \citep{rychly2008lexicographer} =
  ``14 + log2( 2 * f_w / (F_cell + F_w_global) )``,
  where ``F_cell`` is the cell's pronoun-token total and ``F_w_global`` is the
  head-lemma's total cell-pronoun co-occurrences across all cells in the slice.
  log-Dice is on a roughly bounded scale that does not over-reward rare words.
* **PMI** = ``log2( P(w | cell) / P(w) )``.
* **Log-likelihood ratio (G²)** for a $2 \times 2$ contingency table
  (cell vs other cells $\times$ head-lemma vs other head-lemmas), the standard
  collocation significance measure since \citet{dunning1993accurate}.

Period-differential measures compare each ``(cell, head_lemma)`` count between
P1 (2014--2021) and P2 (post-2022):

* **ΔPMI** = ``PMI_P2 - PMI_P1``.
* **G² (period contrast)** on a $2 \times 2$ table (P1 vs P2 $\times$
  with-cell-pronoun vs without). BH-FDR within (language, cell) controls
  multiplicity over head-lemmas.

Caching
-------
Stanza parses are expensive; the script caches them as JSONL keyed by
``(poem_id, stanza_index, language, stanza_hash)``. Re-running with the same
inputs reads the cache and skips parsing. Set ``--rebuild-cache`` to ignore
the cache.

Outputs (in ``outputs/02_modeling_pronoun_collocations/``)
----------------------------------------------------------
* ``collocations_by_cell_period.csv``  — long table, one row per
  ``(language, cell, period, deprel, head_lemma)`` with counts and measures.
* ``differential_collocations_top.csv`` — per (language, cell, deprel), the
  top head-lemmas ranked by absolute ΔPMI with G² period-contrast p-values
  (BH-corrected).
* ``collocation_scatter_<language>_<cell>.png`` — log-frequency vs ΔPMI
  scatter plots (Word Sketch Difference style).
* ``parse_cache/parses.jsonl`` — Stanza parse cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.stats_common import bh_adjust
from utils.workspace import canonical_pronoun_annotation_csv, prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = canonical_pronoun_annotation_csv(ROOT)
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_collocations"


# Pronoun lemma → (person, number, vy_register-aware-cell). We use Stanza's
# morphological output for person/number but keep this lookup table to (a)
# recover the cell when Stanza's morph features are noisy and (b) preserve
# the polite-singular vs true-plural ви distinction, which Stanza alone
# cannot decide. For ви we default to ``true_plural`` here because the
# GPT-annotated polite-singular tokens are 23 corpus-wide and dropped from
# frequentist inference.
PRONOUN_LEMMA_TO_CELL: dict[str, str] = {
    # Ukrainian
    "я": "1sg",
    "мене": "1sg",
    "мені": "1sg",
    "мною": "1sg",
    "ми": "1pl",
    "нас": "1pl",
    "нам": "1pl",
    "нами": "1pl",
    "ти": "2sg",
    "тебе": "2sg",
    "тобі": "2sg",
    "тобою": "2sg",
    "ви": "2pl_vy_true_plural",
    "вас": "2pl_vy_true_plural",
    "вам": "2pl_vy_true_plural",
    "вами": "2pl_vy_true_plural",
    # Russian (share most of the orthography but distinct lemmas)
    "мы": "1pl",
    "вы": "2pl_vy_true_plural",
    "тебя": "2sg",
}

# Russian and Ukrainian both have 1sg lemma "я" so the table is intentionally
# language-agnostic; downstream we split by ``language`` column.

PRIMARY_CELLS = ("1sg", "1pl", "2sg", "2pl_vy_true_plural")
# Top-level deprels we report. Others land in "other".
RETAINED_DEPRELS = frozenset(
    {
        "nsubj",
        "nsubj:pass",
        "obj",
        "iobj",
        "obl",
        "nmod",
        "nmod:poss",
        "root",
        "ccomp",
        "xcomp",
        "advcl",
        "conj",
    }
)


@dataclass
class PronounContext:
    """One observation of a personal pronoun in dependency-parsed text."""

    poem_id: str
    stanza_index: int
    language: str
    period: str
    author: str
    pronoun_lemma: str
    pronoun_form: str
    cell: str
    deprel: str
    head_lemma: str
    head_upos: str


def _stable_stanza_hash(text: str) -> str:
    """Short content hash for parse-cache keying."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_stanza_pipelines(use_gpu: bool):
    """Lazy-load Ukrainian and Russian Stanza pipelines on first call."""
    import stanza

    pipelines: dict[str, "stanza.Pipeline"] = {}
    for lang in ("uk", "ru"):
        try:
            pipelines[lang] = stanza.Pipeline(
                lang=lang,
                processors="tokenize,pos,lemma,depparse",
                use_gpu=use_gpu,
                download_method=None,
                verbose=False,
            )
        except Exception as exc:
            log.warning("Could not initialize Stanza %s pipeline: %s", lang, exc)
    return pipelines


def _iter_stanzas(input_df: pd.DataFrame) -> Iterable[dict[str, object]]:
    """Yield one record per unique (poem_id, stanza_index, language) covered by the
    annotated corpus. The GPT annotation table emits one row per pronoun, so we
    deduplicate; poems without pronouns are still present as a single row with
    ``pronoun_word == NaN`` and are included so we can compute per-cell denominators."""
    base_cols = [
        "poem_id",
        "stanza_index",
        "language",
        "temporal_period",
        "author",
        "stanza_ukr",
    ]
    missing = sorted(set(base_cols) - set(input_df.columns))
    if missing:
        raise ValueError(f"Annotation CSV missing required columns: {missing}")
    dedup = input_df.drop_duplicates(["poem_id", "stanza_index"]).copy()
    dedup = dedup.loc[dedup["stanza_ukr"].notna()].copy()
    for _, row in dedup.iterrows():
        lang = str(row["language"]).strip()
        if lang not in ("Ukrainian", "Russian"):
            continue
        yield {
            "poem_id": str(row["poem_id"]),
            "stanza_index": int(row["stanza_index"]),
            "language": lang,
            "period": str(row["temporal_period"]),
            "author": str(row["author"]),
            "stanza_ukr": str(row["stanza_ukr"]),
        }


def _parse_stanza_text(
    pipeline,
    stanza_text: str,
    record: dict[str, object],
) -> list[PronounContext]:
    """Return a list of PronounContext rows for personal pronouns in the parse."""
    out: list[PronounContext] = []
    try:
        doc = pipeline(stanza_text)
    except Exception as exc:
        log.warning(
            "Stanza failed on poem %s stanza %s: %s",
            record["poem_id"],
            record["stanza_index"],
            exc,
        )
        return out

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos != "PRON":
                continue
            lemma = (word.lemma or "").lower()
            if lemma not in PRONOUN_LEMMA_TO_CELL:
                continue
            cell = PRONOUN_LEMMA_TO_CELL[lemma]
            head_word = sent.words[word.head - 1] if word.head and word.head > 0 else None
            head_lemma = (head_word.lemma if head_word else "ROOT") or "ROOT"
            head_upos = (head_word.upos if head_word else "ROOT") or "ROOT"
            deprel = word.deprel or "dep"
            if deprel not in RETAINED_DEPRELS:
                deprel = "other"
            out.append(
                PronounContext(
                    poem_id=str(record["poem_id"]),
                    stanza_index=int(record["stanza_index"]),
                    language=str(record["language"]),
                    period=str(record["period"]),
                    author=str(record["author"]),
                    pronoun_lemma=lemma,
                    pronoun_form=word.text.lower(),
                    cell=cell,
                    deprel=deprel,
                    head_lemma=head_lemma.lower(),
                    head_upos=head_upos,
                )
            )
    return out


def parse_corpus(
    input_df: pd.DataFrame,
    cache_path: Path,
    *,
    rebuild_cache: bool,
    use_gpu: bool,
) -> pd.DataFrame:
    """Parse all stanzas with Stanza (cached). Returns one row per pronoun token."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_keys: set[str] = set()
    rows: list[dict[str, object]] = []
    if cache_path.is_file() and not rebuild_cache:
        with cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                cache_keys.add(rec["__cache_key"])
                rows.extend(rec["contexts"])

    pipelines = None
    pending = list(_iter_stanzas(input_df))
    log.info("Stanzas to consider: %d; cached: %d", len(pending), len(cache_keys))
    new_count = 0
    with cache_path.open("a", encoding="utf-8") as fh:
        for record in pending:
            key = "{}|{}|{}|{}".format(
                record["poem_id"],
                record["stanza_index"],
                record["language"],
                _stable_stanza_hash(record["stanza_ukr"]),
            )
            if key in cache_keys:
                continue
            if pipelines is None:
                pipelines = _load_stanza_pipelines(use_gpu=use_gpu)
            lang_short = "uk" if record["language"] == "Ukrainian" else "ru"
            pipe = pipelines.get(lang_short)
            if pipe is None:
                continue
            ctxs = _parse_stanza_text(pipe, record["stanza_ukr"], record)
            ctx_dicts = [c.__dict__ for c in ctxs]
            rows.extend(ctx_dicts)
            fh.write(
                json.dumps(
                    {"__cache_key": key, "contexts": ctx_dicts}, ensure_ascii=False
                )
                + "\n"
            )
            cache_keys.add(key)
            new_count += 1
            if new_count % 500 == 0:
                log.info("Parsed %d new stanzas (running total %d)", new_count, len(rows))

    log.info("Total pronoun-context rows: %d (newly parsed stanzas: %d)", len(rows), new_count)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _aggregate_collocates(parses: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ``(language, cell, period, deprel, head_lemma)``."""
    grouped = (
        parses.groupby(["language", "cell", "period", "deprel", "head_lemma"], sort=False)
        .size()
        .reset_index(name="cooccurrence")
    )
    return grouped


def _add_collocation_measures(coll: pd.DataFrame, parses: pd.DataFrame) -> pd.DataFrame:
    """Append log-Dice, PMI, and G² to the aggregated collocate table.

    G² uses Dunning's log-likelihood for a 2×2 table:
        head vs other-heads × cell vs other-cells, within the same (language, period) slice.
    """
    out = coll.copy()
    out["cooccurrence"] = out["cooccurrence"].astype(int)

    # Per-slice totals for log-Dice / PMI / G².
    cell_totals = (
        parses.groupby(["language", "period", "cell"], sort=False).size().rename("F_cell").reset_index()
    )
    head_totals = (
        parses.groupby(["language", "period", "head_lemma"], sort=False)
        .size()
        .rename("F_head")
        .reset_index()
    )
    slice_totals = parses.groupby(["language", "period"], sort=False).size().rename("F_slice").reset_index()

    out = out.merge(cell_totals, on=["language", "period", "cell"], how="left")
    out = out.merge(head_totals, on=["language", "period", "head_lemma"], how="left")
    out = out.merge(slice_totals, on=["language", "period"], how="left")

    eps = 1e-12
    out["log_dice"] = 14.0 + np.log2(
        (2.0 * out["cooccurrence"]) / (out["F_cell"] + out["F_head"] + eps)
    )

    p_w_given_cell = out["cooccurrence"] / (out["F_cell"] + eps)
    p_w = out["F_head"] / (out["F_slice"] + eps)
    out["pmi"] = np.log2((p_w_given_cell + eps) / (p_w + eps))

    # G²: 2×2 ( cell vs ¬cell ) × ( head vs ¬head )
    a = out["cooccurrence"].astype(float)
    b = out["F_cell"].astype(float) - a  # cell, ¬head
    c = out["F_head"].astype(float) - a  # ¬cell, head
    d = out["F_slice"].astype(float) - a - b - c
    n = a + b + c + d

    def _xlogx(x: pd.Series, ref: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((x > 0) & (ref > 0), np.log(x / ref), 0.0)
        return x * ratio

    expected_a = (a + b) * (a + c) / n.replace(0, np.nan)
    expected_b = (a + b) * (b + d) / n.replace(0, np.nan)
    expected_c = (c + d) * (a + c) / n.replace(0, np.nan)
    expected_d = (c + d) * (b + d) / n.replace(0, np.nan)
    g2 = 2.0 * (
        _xlogx(a, expected_a)
        + _xlogx(b, expected_b)
        + _xlogx(c, expected_c)
        + _xlogx(d, expected_d)
    )
    out["g2"] = g2.replace([np.inf, -np.inf], np.nan)

    return out


def _period_differential(coll_with_measures: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide-by-period and compute ΔPMI, G² for the P1-vs-P2 contrast."""
    df = coll_with_measures.copy()
    keep_cols = [
        "language",
        "cell",
        "deprel",
        "head_lemma",
        "period",
        "cooccurrence",
        "F_cell",
        "F_head",
        "F_slice",
        "pmi",
    ]
    df = df[keep_cols]
    pivot_count = df.pivot_table(
        index=["language", "cell", "deprel", "head_lemma"],
        columns="period",
        values="cooccurrence",
        fill_value=0,
        aggfunc="sum",
    )
    pivot_pmi = df.pivot_table(
        index=["language", "cell", "deprel", "head_lemma"],
        columns="period",
        values="pmi",
        fill_value=np.nan,
        aggfunc="mean",
    )

    cells = ["2014_2021", "post_2022"]
    for col in cells:
        if col not in pivot_count.columns:
            pivot_count[col] = 0
        if col not in pivot_pmi.columns:
            pivot_pmi[col] = np.nan
    diff = pivot_count[cells].copy()
    diff.columns = [f"cooc_{c}" for c in cells]
    diff[f"pmi_{cells[0]}"] = pivot_pmi[cells[0]]
    diff[f"pmi_{cells[1]}"] = pivot_pmi[cells[1]]
    diff["delta_pmi"] = diff[f"pmi_{cells[1]}"] - diff[f"pmi_{cells[0]}"]
    diff = diff.reset_index()

    # G² for the per-collocate period contrast: 2×2 (period × with-this-head)
    slice_p1 = (
        coll_with_measures.loc[coll_with_measures["period"].eq(cells[0])]
        .groupby(["language", "cell"], sort=False)["cooccurrence"]
        .sum()
        .rename("N_p1")
        .reset_index()
    )
    slice_p2 = (
        coll_with_measures.loc[coll_with_measures["period"].eq(cells[1])]
        .groupby(["language", "cell"], sort=False)["cooccurrence"]
        .sum()
        .rename("N_p2")
        .reset_index()
    )
    diff = diff.merge(slice_p1, on=["language", "cell"], how="left").fillna({"N_p1": 0})
    diff = diff.merge(slice_p2, on=["language", "cell"], how="left").fillna({"N_p2": 0})

    a = diff[f"cooc_{cells[1]}"].astype(float)
    b = diff["N_p2"].astype(float) - a
    c = diff[f"cooc_{cells[0]}"].astype(float)
    d = diff["N_p1"].astype(float) - c
    n = a + b + c + d

    def _xlogx(x: pd.Series, ref: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((x > 0) & (ref > 0), np.log(x / ref), 0.0)
        return x * ratio

    expected_a = (a + b) * (a + c) / n.replace(0, np.nan)
    expected_b = (a + b) * (b + d) / n.replace(0, np.nan)
    expected_c = (c + d) * (a + c) / n.replace(0, np.nan)
    expected_d = (c + d) * (b + d) / n.replace(0, np.nan)
    diff["g2_period_contrast"] = (
        2.0
        * (
            _xlogx(a, expected_a)
            + _xlogx(b, expected_b)
            + _xlogx(c, expected_c)
            + _xlogx(d, expected_d)
        )
    ).replace([np.inf, -np.inf], np.nan)

    # χ²(df=1) approximation for the p-value (G² is asymptotically χ²).
    from scipy.stats import chi2

    diff["p_value_period_contrast"] = 1.0 - chi2.cdf(diff["g2_period_contrast"].fillna(0.0), df=1)

    diff["q_value_bh"] = np.nan
    grouped_q = diff.groupby(["language", "cell"], group_keys=False)["p_value_period_contrast"].apply(
        bh_adjust
    )
    diff.loc[grouped_q.index, "q_value_bh"] = grouped_q
    return diff


def _plot_scatter(diff: pd.DataFrame, out_dir: Path) -> None:
    """Per (language, cell), write a Word Sketch Difference-style scatter."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cells_total = (
        diff.assign(total=diff["cooc_2014_2021"] + diff["cooc_post_2022"])
        .groupby(["language", "cell"])["total"]
        .sum()
    )
    for (lang, cell), grp in diff.groupby(["language", "cell"]):
        grp = grp.assign(total=grp["cooc_2014_2021"] + grp["cooc_post_2022"])
        grp = grp.loc[grp["total"] >= 5].copy()
        if grp.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            np.log10(grp["total"].clip(lower=1)),
            grp["delta_pmi"].fillna(0.0),
            s=8 + np.sqrt(grp["total"]) * 2,
            alpha=0.35,
            c="0.4",
        )
        # Annotate top movers in either direction.
        top_pos = grp.nlargest(8, "delta_pmi")
        top_neg = grp.nsmallest(8, "delta_pmi")
        for _, row in pd.concat([top_pos, top_neg]).iterrows():
            ax.annotate(
                row["head_lemma"],
                (math.log10(max(int(row["total"]), 1)), row["delta_pmi"]),
                fontsize=7,
                alpha=0.85,
            )
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_xlabel("log10(co-occurrence count, P1+P2)")
        ax.set_ylabel("ΔPMI = PMI_post2022 − PMI_pre2022")
        ax.set_title(
            f"Period-differential pronoun collocates · {lang} · cell={cell} "
            f"(N={int(cells_total.get((lang, cell), 0))})"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"collocation_scatter_{lang}_{cell}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Dependency-parsed pronoun collocations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore parse cache and re-run Stanza from scratch.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Pass use_gpu=True to Stanza."
    )
    parser.add_argument(
        "--min-coocc-for-diff",
        type=int,
        default=3,
        help="Minimum total (P1+P2) co-occurrence for inclusion in differential table.",
    )
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "parse_cache" / "parses.jsonl"

    df = pd.read_csv(args.input, low_memory=False, on_bad_lines="skip")
    parses = parse_corpus(
        df,
        cache_path,
        rebuild_cache=args.rebuild_cache,
        use_gpu=args.use_gpu,
    )
    if parses.empty:
        log.warning("No parses produced; aborting.")
        return

    # Restrict downstream tables to the four primary cells and to P1 / P2 periods.
    parses = parses.loc[parses["cell"].isin(PRIMARY_CELLS)].copy()
    parses = parses.loc[parses["period"].isin(("2014_2021", "post_2022"))].copy()
    parses.to_csv(out_dir / "pronoun_contexts_long.csv", index=False)

    coll = _aggregate_collocates(parses)
    coll = _add_collocation_measures(coll, parses)
    coll.to_csv(out_dir / "collocations_by_cell_period.csv", index=False)

    diff = _period_differential(coll)
    diff_filtered = diff.loc[
        (diff["cooc_2014_2021"] + diff["cooc_post_2022"]) >= args.min_coocc_for_diff
    ].copy()
    diff_filtered.to_csv(out_dir / "differential_collocations.csv", index=False)

    # Top movers per (language, cell, deprel): top 25 by |ΔPMI| with q ≤ 0.10.
    top_rows: list[pd.DataFrame] = []
    for (lang, cell, deprel), grp in diff_filtered.groupby(["language", "cell", "deprel"]):
        grp = grp.assign(abs_delta_pmi=grp["delta_pmi"].abs())
        ranked = grp.sort_values("abs_delta_pmi", ascending=False).head(25)
        top_rows.append(ranked)
    if top_rows:
        pd.concat(top_rows, ignore_index=True).to_csv(
            out_dir / "differential_collocations_top.csv", index=False
        )

    _plot_scatter(diff_filtered, out_dir)
    log.info("Wrote collocation outputs to %s", out_dir)


if __name__ == "__main__":
    main()
