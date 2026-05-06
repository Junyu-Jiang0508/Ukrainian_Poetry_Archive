"""Explore GPT annotation fields: distributions, missingness, period crosstabs."""

from pathlib import Path

from utils.annotation_cohort import load_gpt_annotation_derived
from utils.annotation_derived_columns import CORE_TEMPORAL_PERIOD_ORDER
from utils.workspace import gpt_public_annotation_detailed_csv, prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INPUT_CSV = gpt_public_annotation_detailed_csv(ROOT)
OUTPUT_DIR = ROOT / "outputs/12_gpt_annotation_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEMANTIC_FIELDS = [
    "we_inclusivity",
    "addressee_type",
    "dominant_referent_category",
    "poem_perspective_primary",
    "poem_perspective_secondary",
    "polyphony_type",
]

# Crosstabs / contrasts that need dated bins (exclude unknown; pre_2014 appendix-only)
KNOWN_DATED_PERIODS = ["pre_2014", "2014_2021", "post_2022"]
CORE_TEMPORAL_PERIODS = CORE_TEMPORAL_PERIOD_ORDER


def load_data() -> pd.DataFrame:
    df = load_gpt_annotation_derived(INPUT_CSV)
    df["temporal_period_for_analysis"] = df["temporal_period_reconciled"]
    print(f"Loaded {len(df)} rows, {df['original_id'].nunique()} unique poems")
    return df


def field_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute value counts, missing rate, and unique count for each semantic field."""
    records = []
    for field in SEMANTIC_FIELDS + ["qa_flag"]:
        if field not in df.columns:
            continue
        total = len(df)
        missing = df[field].isna().sum()
        empty_str = (df[field].astype(str).str.strip() == "").sum() - missing
        vc = df[field].dropna().value_counts()
        for val, cnt in vc.items():
            records.append({
                "field": field,
                "value": val,
                "count": cnt,
                "pct": cnt / total * 100,
            })
        records.append({
            "field": field,
            "value": "__MISSING__",
            "count": missing,
            "pct": missing / total * 100,
        })
        if empty_str > 0:
            records.append({
                "field": field,
                "value": "__EMPTY_STRING__",
                "count": int(empty_str),
                "pct": empty_str / total * 100,
            })
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "field_value_distributions.csv", index=False)
    print(f"\nField distributions saved ({len(result)} rows)")
    return result


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    all_fields = SEMANTIC_FIELDS + ["qa_flag", "person", "number", "pronoun_word",
                                     "temporal_period", "year", "source_mapping"]
    rows = []
    for f in all_fields:
        if f not in df.columns:
            continue
        total = len(df)
        na = df[f].isna().sum()
        rows.append({"field": f, "total": total, "missing": na,
                      "missing_pct": na / total * 100})
    result = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    result.to_csv(OUTPUT_DIR / "missing_summary.csv", index=False)
    print("\nMissing summary:")
    print(result.to_string(index=False))
    return result


def crosstab_by_period(df: pd.DataFrame):
    """Cross-tabulate each semantic field by reconciled temporal period. Save CSV + heatmap."""
    tcol = "temporal_period_for_analysis"
    df_known = df[df[tcol].isin(KNOWN_DATED_PERIODS)].copy()
    actual_periods = [p for p in KNOWN_DATED_PERIODS if p in df_known[tcol].unique()]
    df_known[tcol] = pd.Categorical(
        df_known[tcol], categories=actual_periods, ordered=True,
    )

    for field in SEMANTIC_FIELDS:
        if field not in df_known.columns:
            continue
        ct_count = pd.crosstab(df_known[field], df_known[tcol], margins=True)
        ct_count.to_csv(OUTPUT_DIR / f"crosstab_{field}_count.csv")

        ct_pct = pd.crosstab(df_known[field], df_known[tcol], normalize="columns") * 100
        ct_pct.to_csv(OUTPUT_DIR / f"crosstab_{field}_pct.csv")

        fig, ax = plt.subplots(figsize=(10, max(4, len(ct_pct) * 0.5)))
        sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title(f"{field} distribution (%) by period (reconciled)")
        ax.set_ylabel(field)
        ax.set_xlabel("Period")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"heatmap_{field}.png", dpi=150)
        plt.close(fig)
        print(f"  {field}: {ct_count.shape[0]-1} categories")


def pronoun_class_by_period(df: pd.DataFrame):
    """Summarize pronoun person/number by reconciled period (core = 2014_2021 vs post_2022)."""
    tcol = "temporal_period_for_analysis"
    df_known = df[df[tcol].isin(KNOWN_DATED_PERIODS)].copy()
    df_valid = df_known.dropna(subset=["person", "number"])

    def classify(row):
        p, n = str(row["person"]), str(row["number"])
        if "1st" in p and "Sing" in n:
            return "1sg"
        if "1st" in p and "Plur" in n:
            return "1pl"
        if "2nd" in p:
            return "2"
        if "3rd" in p and "Sing" in n:
            return "3sg"
        if "3rd" in p and "Plur" in n:
            return "3pl"
        return "other"

    df_valid = df_valid.copy()
    df_valid["pronoun_class"] = df_valid.apply(classify, axis=1)
    ct = pd.crosstab(df_valid["pronoun_class"], df_valid[tcol])
    ct_pct = pd.crosstab(df_valid["pronoun_class"], df_valid[tcol], normalize="columns") * 100
    ct.to_csv(OUTPUT_DIR / "pronoun_class_by_period_count.csv")
    ct_pct.to_csv(OUTPUT_DIR / "pronoun_class_by_period_pct.csv")
    df_core = df_valid[df_valid[tcol].isin(CORE_TEMPORAL_PERIODS)]
    ct_core = pd.crosstab(df_core["pronoun_class"], df_core[tcol])
    ct_core_pct = pd.crosstab(df_core["pronoun_class"], df_core[tcol], normalize="columns") * 100
    ct_core.to_csv(OUTPUT_DIR / "pronoun_class_core_temporal_count.csv")
    ct_core_pct.to_csv(OUTPUT_DIR / "pronoun_class_core_temporal_pct.csv")
    print("\nPronoun class by period (%):")
    print(ct_pct.round(1).to_string())


def qa_flag_summary(df: pd.DataFrame):
    vc = df["qa_flag"].value_counts()
    total = len(df)
    print(f"\nQA flag summary (total={total}):")
    for v, c in vc.items():
        print(f"  {v}: {c} ({c/total*100:.1f}%)")
    vc.to_csv(OUTPUT_DIR / "qa_flag_summary.csv")


def poem_level_summary(df: pd.DataFrame):
    """One row per poem: perspective, pronoun counts, period."""
    tcol = "temporal_period_for_analysis"
    poems = df.groupby("original_id").agg(
        author=("author", "first"),
        year=("year", "first"),
        temporal_period=(tcol, "first"),
        perspective_primary=("poem_perspective_primary", "first"),
        perspective_secondary=("poem_perspective_secondary", "first"),
        n_pronoun_rows=("pronoun_word", "count"),
        n_pronoun_detected=("pronoun_word", lambda x: x.notna().sum()),
    ).reset_index()
    poems.to_csv(OUTPUT_DIR / "poem_level_summary.csv", index=False)
    print(f"\nPoem-level summary: {len(poems)} poems")

    df_known = poems[poems["temporal_period"].isin(KNOWN_DATED_PERIODS)]
    ct = pd.crosstab(df_known["perspective_primary"], df_known["temporal_period"])
    ct.to_csv(OUTPUT_DIR / "perspective_primary_by_period.csv")
    print("Perspective primary by period:")
    print(ct.to_string())
    return poems


def temporal_period_overview(df: pd.DataFrame):
    print("\nTemporal period overview (reconciled from year when available):")
    tcol = "temporal_period_for_analysis"
    vc = df[tcol].value_counts()
    for v, c in vc.items():
        n_poems = df[df[tcol] == v]["original_id"].nunique()
        print(f"  {v}: {c} rows, {n_poems} poems")


def main():
    df = load_data()
    temporal_period_overview(df)
    missing_summary(df)
    field_distributions(df)
    print("\nCross-tabulations by period:")
    crosstab_by_period(df)
    pronoun_class_by_period(df)
    qa_flag_summary(df)
    poem_level_summary(df)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
