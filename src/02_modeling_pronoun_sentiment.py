"""Stanza-level sentiment scoring and pronoun×sentiment mixed-effects model (P1-D).

We score every Ukrainian and Russian stanza in the annotated corpus with the
multilingual sentiment model ``cardiffnlp/twitter-xlm-roberta-base-sentiment``
(three classes: positive / neutral / negative). The scalar sentiment value
$s = P(\text{positive}) - P(\text{negative}) \in [-1, 1]$ is then joined to
each stanza's pronoun-cell tally and the dominant cell label. A linear mixed
model with a random author intercept tests whether sentiment is differently
distributed across cells *post-cutpoint* relative to *pre-cutpoint*.

Why a separate sentiment layer
------------------------------
The Q1/Q2a estimands operate on counts. They tell us how attention is
reallocated between pronominal cells but nothing about the *affective
valence* attached to each cell. The wartime poetic claim that the 1pl shift
is a turn toward solidarity, mourning, or defiance is testable only against
an independent affect signal. We pick a multilingual model that ships with
Ukrainian + Russian training data so we do not have to fine-tune separately
per language stratum.

Caching
-------
Predictions are cached as JSONL keyed by ``(poem_id, stanza_index, hash)``.
Re-running with the same input reads the cache.

Outputs (in ``outputs/02_modeling_pronoun_sentiment/``)
-------------------------------------------------------
* ``stanza_sentiment_predictions.csv``  — per-stanza sentiment score + label.
* ``stanza_sentiment_long.csv``         — joined with cell counts and dominant cell.
* ``sentiment_mixedlm_coefs.csv``       — fitted mixed-LM coefficients.
* ``sentiment_by_cell_period.csv``      — descriptive cell × period table.
* ``sentiment_by_cell_period.png``      — strip / box plot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.pronoun_encoding import (
    POEM_COUNT_CELL_COLUMNS,
    poem_person_cell_column,
)
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_sentiment"
DEFAULT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

PRIMARY_CELLS = ("1sg", "1pl", "2sg", "2pl_vy_true_plural")


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_pipeline(model_name: str, device: int):
    """Lazy import + load: keeps transformers an optional runtime dependency."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
        truncation=True,
        max_length=192,
    )


def _score_sentence_iter(
    stanzas: list[dict[str, object]],
    pipe,
    *,
    batch_size: int,
) -> list[dict[str, object]]:
    """Run sentiment scoring in batches, returning one record per stanza."""
    out: list[dict[str, object]] = []
    texts = [str(s["stanza_ukr"]) for s in stanzas]
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_meta = stanzas[batch_start : batch_start + batch_size]
        try:
            preds = pipe(batch_texts)
        except Exception as exc:
            log.warning(
                "Sentiment batch starting at %d failed: %s — skipping batch",
                batch_start,
                exc,
            )
            continue
        for meta, pred in zip(batch_meta, preds):
            scores = {p["label"].lower(): float(p["score"]) for p in pred}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral", 0.0)
            score = pos - neg
            label = max(scores, key=lambda k: scores[k])
            out.append(
                {
                    "poem_id": meta["poem_id"],
                    "stanza_index": int(meta["stanza_index"]),
                    "language": meta["language"],
                    "author": meta["author"],
                    "temporal_period": meta["temporal_period"],
                    "p_positive": pos,
                    "p_neutral": neu,
                    "p_negative": neg,
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "__cache_key": meta["__cache_key"],
                }
            )
    return out


def score_corpus_sentiment(
    df: pd.DataFrame,
    cache_path: Path,
    *,
    rebuild_cache: bool,
    model_name: str,
    device: int,
    batch_size: int,
) -> pd.DataFrame:
    """Score every (Ukrainian|Russian) stanza, with on-disk caching."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict[str, object]] = {}
    if cache_path.is_file() and not rebuild_cache:
        with cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                cached[rec["__cache_key"]] = rec

    base_cols = [
        "poem_id",
        "stanza_index",
        "language",
        "temporal_period",
        "author",
        "stanza_ukr",
    ]
    missing = sorted(set(base_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Annotation CSV missing columns: {missing}")
    dedup = df.drop_duplicates(["poem_id", "stanza_index"]).copy()
    dedup = dedup.loc[dedup["stanza_ukr"].notna()].copy()
    dedup = dedup.loc[dedup["language"].astype(str).isin(("Ukrainian", "Russian"))].copy()

    pending: list[dict[str, object]] = []
    pre_scored: list[dict[str, object]] = []
    for _, row in dedup.iterrows():
        text = str(row["stanza_ukr"])
        cache_key = f"{row['poem_id']}|{int(row['stanza_index'])}|{_stable_hash(text)}"
        if cache_key in cached:
            pre_scored.append(cached[cache_key])
            continue
        pending.append(
            {
                "poem_id": str(row["poem_id"]),
                "stanza_index": int(row["stanza_index"]),
                "language": str(row["language"]),
                "author": str(row["author"]),
                "temporal_period": str(row["temporal_period"]),
                "stanza_ukr": text,
                "__cache_key": cache_key,
            }
        )

    log.info("Sentiment scoring: %d cached, %d pending", len(pre_scored), len(pending))
    if pending:
        pipe = _load_pipeline(model_name, device=device)
        new_scores = _score_sentence_iter(pending, pipe, batch_size=batch_size)
        with cache_path.open("a", encoding="utf-8") as fh:
            for rec in new_scores:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pre_scored.extend(new_scores)

    if not pre_scored:
        return pd.DataFrame()
    out = pd.DataFrame(pre_scored)
    out = out.drop(columns=["__cache_key"], errors="ignore")
    return out


def _stanza_dominant_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Compute, per (poem_id, stanza_index), the dominant primary-cell label.

    The dominant cell is the cell with the largest count of pronoun tokens in
    that stanza, breaking ties toward the cell with the lower index in
    ``PRIMARY_CELLS`` (deterministic). If no primary-cell pronoun appears the
    dominant label is ``"none"``.
    """
    cell_col = poem_person_cell_column(df)
    df = df.assign(_cell=cell_col)
    sub = df.loc[df["_cell"].isin(PRIMARY_CELLS)].copy()
    sub["_one"] = 1
    counts = (
        sub.groupby(["poem_id", "stanza_index", "_cell"], sort=False)["_one"]
        .sum()
        .unstack(fill_value=0)
    )
    if counts.empty:
        return pd.DataFrame(
            columns=["poem_id", "stanza_index", "dominant_cell", "n_pronouns_primary"]
        )
    counts = counts.reindex(columns=PRIMARY_CELLS, fill_value=0)
    # argmax with deterministic tie-break.
    arr = counts.to_numpy()
    dominant_idx = arr.argmax(axis=1)
    n_max = arr.max(axis=1)
    dominant = [
        PRIMARY_CELLS[i] if total > 0 else "none"
        for i, total in zip(dominant_idx, n_max)
    ]
    counts_reset = counts.reset_index()
    counts_reset["dominant_cell"] = dominant
    counts_reset["n_pronouns_primary"] = arr.sum(axis=1)
    return counts_reset[
        ["poem_id", "stanza_index", "dominant_cell", "n_pronouns_primary"]
    ]


def fit_mixed_model(sentiment_long: pd.DataFrame) -> pd.DataFrame:
    """Fit `sentiment_score ~ period * dominant_cell + (1|author)` via statsmodels."""
    import statsmodels.formula.api as smf

    fit_df = sentiment_long.loc[
        sentiment_long["dominant_cell"].isin(PRIMARY_CELLS)
        & sentiment_long["temporal_period"].isin(("2014_2021", "post_2022"))
    ].copy()
    if fit_df.empty:
        return pd.DataFrame()
    fit_df["dominant_cell"] = pd.Categorical(
        fit_df["dominant_cell"], categories=PRIMARY_CELLS, ordered=False
    )
    fit_df["period"] = pd.Categorical(
        fit_df["temporal_period"], categories=("2014_2021", "post_2022"), ordered=False
    )

    formula = "sentiment_score ~ C(period, Treatment('2014_2021')) * C(dominant_cell)"
    md = smf.mixedlm(formula, data=fit_df, groups=fit_df["author"])
    fit = md.fit(method="lbfgs")

    summary = (
        pd.DataFrame(
            {
                "term": fit.params.index.astype(str),
                "estimate": fit.params.values,
                "se": fit.bse.values,
                "z": fit.tvalues.values,
                "p_value": fit.pvalues.values,
                "ci95_low": fit.conf_int()[0].values,
                "ci95_high": fit.conf_int()[1].values,
            }
        )
        .reset_index(drop=True)
    )
    summary["n_obs"] = int(len(fit_df))
    summary["n_authors"] = int(fit_df["author"].nunique())
    return summary


def _plot_sentiment_box(sentiment_long: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = sentiment_long.loc[
        sentiment_long["dominant_cell"].isin(PRIMARY_CELLS)
        & sentiment_long["temporal_period"].isin(("2014_2021", "post_2022"))
    ].copy()
    if plot_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, lang in zip(axes, ("Ukrainian", "Russian")):
        sub = plot_df.loc[plot_df["language"].eq(lang)]
        if sub.empty:
            ax.set_visible(False)
            continue
        positions = []
        data = []
        labels = []
        for i, cell in enumerate(PRIMARY_CELLS):
            for j, period in enumerate(("2014_2021", "post_2022")):
                vals = sub.loc[
                    (sub["dominant_cell"].eq(cell)) & (sub["temporal_period"].eq(period)),
                    "sentiment_score",
                ].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                positions.append(i * 3 + j)
                data.append(vals)
                labels.append(f"{cell}\n{period}")
        if data:
            ax.boxplot(data, positions=positions, widths=0.7, showfliers=False)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_title(f"{lang}: stanza sentiment by dominant pronoun cell × period")
        ax.set_ylabel("Sentiment score (P(pos) − P(neg))")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Pronoun×sentiment mixed model (P1-D).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="HuggingFace pipeline device: -1=CPU, 0=GPU 0 (default CPU).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "predictions_cache.jsonl"

    df = pd.read_csv(args.input, low_memory=False, on_bad_lines="skip")

    sentiment = score_corpus_sentiment(
        df,
        cache_path,
        rebuild_cache=args.rebuild_cache,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    if sentiment.empty:
        log.warning("No sentiment predictions produced; aborting.")
        return
    sentiment.to_csv(out_dir / "stanza_sentiment_predictions.csv", index=False)

    dominant = _stanza_dominant_cell(df)
    sentiment_long = sentiment.merge(dominant, on=["poem_id", "stanza_index"], how="left")
    sentiment_long["dominant_cell"] = sentiment_long["dominant_cell"].fillna("none")
    sentiment_long["n_pronouns_primary"] = sentiment_long["n_pronouns_primary"].fillna(0).astype(int)
    sentiment_long.to_csv(out_dir / "stanza_sentiment_long.csv", index=False)

    coefs = fit_mixed_model(sentiment_long)
    if not coefs.empty:
        coefs.to_csv(out_dir / "sentiment_mixedlm_coefs.csv", index=False)

    descriptive = (
        sentiment_long.loc[sentiment_long["dominant_cell"].isin(PRIMARY_CELLS)]
        .groupby(["language", "temporal_period", "dominant_cell"], sort=False)["sentiment_score"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    descriptive.to_csv(out_dir / "sentiment_by_cell_period.csv", index=False)
    _plot_sentiment_box(sentiment_long, out_dir / "sentiment_by_cell_period.png")
    log.info("Wrote sentiment outputs to %s", out_dir)


if __name__ == "__main__":
    main()
