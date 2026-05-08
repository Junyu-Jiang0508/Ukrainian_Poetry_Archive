"""Manual audit package for `vy_register` quality assurance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend=None)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "01_annotation_vy_register_audit"


def _normalize_text_col(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _stratified_sample_true_plural(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    if df.empty or n_total <= 0:
        return df.iloc[0:0].copy()
    work = df.copy()
    lang = "language" if "language" in work.columns else None
    period = "temporal_period" if "temporal_period" in work.columns else None
    if lang is None:
        work["_lang"] = "unknown_language"
        lang = "_lang"
    if period is None:
        if "year" in work.columns:
            yy = pd.to_numeric(work["year"], errors="coerce")
            work["_period"] = np.where(yy.ge(2022), "post_2022", "pre_2022")
        else:
            work["_period"] = "unknown_period"
        period = "_period"
    grouped = list(work.groupby([lang, period], dropna=False))
    if not grouped:
        return work.iloc[0:0].copy()
    base = max(1, n_total // len(grouped))
    draws = []
    rng = np.random.default_rng(int(seed))
    for _, g in grouped:
        take = min(len(g), base)
        if take <= 0:
            continue
        draws.append(g.sample(n=take, random_state=int(rng.integers(1, 1_000_000))))
    sampled = pd.concat(draws, ignore_index=False).drop_duplicates()
    remaining = n_total - len(sampled)
    if remaining > 0:
        pool = work.drop(index=sampled.index, errors="ignore")
        if not pool.empty:
            take2 = min(remaining, len(pool))
            sampled = pd.concat(
                [sampled, pool.sample(n=take2, random_state=int(rng.integers(1, 1_000_000)))],
                ignore_index=False,
            )
    return sampled.reset_index(drop=True)


def _compute_agreement(reviewed: pd.DataFrame, out_dir: Path) -> None:
    need = {"vy_register", "reviewer_vy_register"}
    if not need.issubset(reviewed.columns):
        return
    valid = reviewed.loc[_normalize_text_col(reviewed["reviewer_vy_register"]).ne("")].copy()
    if valid.empty:
        return
    model_label = _normalize_text_col(valid["vy_register"])
    human_label = _normalize_text_col(valid["reviewer_vy_register"])
    agree = float((model_label == human_label).mean())
    confusion = (
        pd.crosstab(human_label, model_label, dropna=False)
        .rename_axis(index="human_label", columns="model_label")
        .reset_index()
    )
    confusion.to_csv(out_dir / "vy_register_confusion_matrix.csv", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "n_reviewed", "value": int(len(valid))},
            {"metric": "raw_agreement", "value": agree},
            {
                "metric": "false_positive_rate_polite_singular",
                "value": float(
                    (
                        (model_label == "polite_singular")
                        & (human_label != "polite_singular")
                    ).mean()
                ),
            },
        ]
    )
    summary.to_csv(out_dir / "vy_register_audit_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare/score manual audit package for vy_register labels.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--sample-true-plural", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260508)
    ap.add_argument(
        "--reviewed-csv",
        type=Path,
        default=None,
        help="Optional completed review sheet with `reviewer_vy_register` column for agreement scoring.",
    )
    args = ap.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input.resolve(), low_memory=False)
    if "vy_register" not in df.columns:
        raise SystemExit("Input CSV must contain `vy_register`.")

    base_cols = [c for c in ("poem_id", "author", "year", "temporal_period", "language", "stanza_index", "stanza_ukr", "pronoun_word", "source_mapping", "vy_register") if c in df.columns]
    polite = df.loc[_normalize_text_col(df["vy_register"]).eq("polite_singular"), base_cols].copy()
    polite["reviewer_vy_register"] = ""
    polite["reviewer_notes"] = ""
    polite.to_csv(out_dir / "vy_register_polite_singular_full_manual_review.csv", index=False)

    true_plural = df.loc[_normalize_text_col(df["vy_register"]).eq("genuine_plural"), base_cols].copy()
    sampled = _stratified_sample_true_plural(true_plural, int(args.sample_true_plural), int(args.seed))
    sampled["reviewer_vy_register"] = ""
    sampled["reviewer_notes"] = ""
    sampled.to_csv(out_dir / "vy_register_true_plural_stratified_sample_manual_review.csv", index=False)

    run_info = pd.DataFrame(
        [
            {"artifact": "polite_singular_events", "n_rows": int(len(polite))},
            {"artifact": "true_plural_population", "n_rows": int(len(true_plural))},
            {"artifact": "true_plural_sample", "n_rows": int(len(sampled))},
        ]
    )
    run_info.to_csv(out_dir / "vy_register_audit_run_info.csv", index=False)

    reviewed_path = args.reviewed_csv.resolve() if args.reviewed_csv else None
    if reviewed_path is not None and reviewed_path.is_file():
        reviewed = pd.read_csv(reviewed_path, low_memory=False)
        _compute_agreement(reviewed, out_dir)

    readme = out_dir / "README.md"
    readme.write_text(
        "# vy_register Manual Audit\n\n"
        "- `vy_register_polite_singular_full_manual_review.csv`: full census of model-labeled polite-singular events.\n"
        "- `vy_register_true_plural_stratified_sample_manual_review.csv`: stratified true-plural audit sheet.\n"
        "- Fill `reviewer_vy_register` manually and rerun with `--reviewed-csv` to compute agreement outputs.\n",
        encoding="utf-8",
    )
    print(f"Wrote vy_register audit package to: {out_dir}")


if __name__ == "__main__":
    main()
