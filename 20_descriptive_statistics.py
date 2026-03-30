"""Descriptive stats: pre_2022 vs post_2022; corpus, pronouns, GPT fields, crosstabs."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

INPUT_CSV = Path("data/data/processed/gpt_annotation_public_run/gpt_annotation_detailed.csv")
OUTPUT_DIR = Path("outputs/20_descriptive_statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEMANTIC_FIELDS = [
    "we_type",
    "addressee_type",
    "dominant_referent_level",
    "poem_perspective_primary",
    "poem_perspective_secondary",
]

PERIODS = ["pre_2022", "post_2022"]


# ---------------------------------------------------------------------------
# Data loading & period remapping
# ---------------------------------------------------------------------------

def load_and_remap() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, low_memory=False, on_bad_lines="skip")
    raw_counts = df["temporal_period"].value_counts()
    print("Raw temporal_period distribution:")
    for v, c in raw_counts.items():
        print(f"  {v}: {c} rows ({df[df['temporal_period']==v]['original_id'].nunique()} poems)")

    period_map = {
        "pre_2014": "pre_2022",
        "2014_2021": "pre_2022",
        "post_2022": "post_2022",
    }
    df["period"] = df["temporal_period"].map(period_map)
    n_dropped = df["period"].isna().sum()
    df = df.dropna(subset=["period"])
    print(f"\nDropped {n_dropped} rows (unknown period)")
    print(f"Final: {len(df)} rows, {df['original_id'].nunique()} poems")
    for p in PERIODS:
        sub = df[df["period"] == p]
        print(f"  {p}: {len(sub)} rows, {sub['original_id'].nunique()} poems")
    return df


def classify_pronoun(row) -> str:
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


# ---------------------------------------------------------------------------
# Module A: Corpus overview
# ---------------------------------------------------------------------------

def module_a_corpus_overview(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODULE A: Corpus Overview")
    print("=" * 60)
    rows = []
    for p in PERIODS + ["Total"]:
        sub = df if p == "Total" else df[df["period"] == p]
        rows.append({
            "period": p,
            "n_poems": sub["original_id"].nunique(),
            "n_authors": sub["author"].nunique(),
            "n_sentence_rows": len(sub),
            "n_pronouns_detected": sub["pronoun_word"].notna().sum(),
            "pronoun_missing_pct": sub["pronoun_word"].isna().mean() * 100,
            "qa_ok_pct": (sub["qa_flag"] == "OK").mean() * 100,
        })
    overview = pd.DataFrame(rows)
    overview.to_csv(OUTPUT_DIR / "A_corpus_overview.csv", index=False)
    print(overview.to_string(index=False))
    return overview


# ---------------------------------------------------------------------------
# Module B: Pronoun class distribution
# ---------------------------------------------------------------------------

def module_b_pronoun_distribution(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODULE B: Pronoun Class Distribution")
    print("=" * 60)

    df_valid = df.dropna(subset=["person", "number"]).copy()
    df_valid["pronoun_class"] = df_valid.apply(classify_pronoun, axis=1)
    df_valid = df_valid[df_valid["pronoun_class"] != "other"]

    ct = pd.crosstab(df_valid["pronoun_class"], df_valid["period"])
    ct = ct.reindex(columns=[p for p in PERIODS if p in ct.columns])
    ct.to_csv(OUTPUT_DIR / "B_pronoun_class_count.csv")

    ct_pct = pd.crosstab(df_valid["pronoun_class"], df_valid["period"],
                          normalize="columns") * 100
    ct_pct = ct_pct.reindex(columns=[p for p in PERIODS if p in ct_pct.columns])
    ct_pct.to_csv(OUTPUT_DIR / "B_pronoun_class_pct.csv")
    print("\nPronoun class by period (%):")
    print(ct_pct.round(1).to_string())

    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    chi_result = pd.DataFrame([{
        "test": "pronoun_class x period",
        "chi2": chi2, "dof": dof, "p_value": p_val,
        "n": n, "cramers_v": cramers_v,
    }])
    chi_result.to_csv(OUTPUT_DIR / "B_pronoun_class_chi2.csv", index=False)
    print(f"\nChi2 = {chi2:.3f}, df = {dof}, p = {p_val:.6f}, Cramer's V = {cramers_v:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ct_pct.index))
    width = 0.35
    for i, p in enumerate(PERIODS):
        if p in ct_pct.columns:
            ax.bar(x + i * width, ct_pct[p].values, width, label=p)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ct_pct.index)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Pronoun Class Distribution: pre_2022 vs post_2022")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, p in enumerate(PERIODS):
        if p in ct_pct.columns:
            for j, v in enumerate(ct_pct[p].values):
                ax.text(j + i * width, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "B_pronoun_class_bar.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Module C: Semantic field frequency tables + bar charts
# ---------------------------------------------------------------------------

def module_c_field_frequencies(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODULE C: GPT Semantic Field Frequencies")
    print("=" * 60)

    for field in SEMANTIC_FIELDS:
        print(f"\n--- {field} ---")
        vc_all = df[field].value_counts(dropna=False)
        total = len(df)
        freq = pd.DataFrame({
            "value": vc_all.index,
            "count": vc_all.values,
            "pct": vc_all.values / total * 100,
        })
        freq.to_csv(OUTPUT_DIR / f"C_{field}_freq.csv", index=False)

        for p in PERIODS:
            sub = df[df["period"] == p]
            vc = sub[field].value_counts(dropna=False)
            sub_total = len(sub)
            pf = pd.DataFrame({
                "value": vc.index,
                "count": vc.values,
                "pct": vc.values / sub_total * 100,
            })
            pf.to_csv(OUTPUT_DIR / f"C_{field}_freq_{p}.csv", index=False)

        valid = df[df[field].notna() & (df[field].astype(str).str.strip() != "")]
        ct_pct = pd.crosstab(valid[field], valid["period"], normalize="columns") * 100
        ct_pct = ct_pct.reindex(columns=[p for p in PERIODS if p in ct_pct.columns])

        fig, ax = plt.subplots(figsize=(12, max(4, len(ct_pct) * 0.45)))
        ct_pct.plot(kind="barh", ax=ax)
        ax.set_title(f"{field} distribution by period (%)")
        ax.set_xlabel("Proportion (%)")
        ax.legend(title="Period")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"C_{field}_bar.png", dpi=150)
        plt.close(fig)

        print(f"  {valid[field].nunique()} unique values")


# ---------------------------------------------------------------------------
# Module D: Cross-tabulation + heatmaps + chi-square
# ---------------------------------------------------------------------------

def module_d_crosstabs(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODULE D: Cross-tabulations + Chi-square")
    print("=" * 60)

    chi_results = []

    for field in SEMANTIC_FIELDS:
        valid = df[df[field].notna() & (df[field].astype(str).str.strip() != "")]

        ct_count = pd.crosstab(valid[field], valid["period"], margins=True)
        ct_count.to_csv(OUTPUT_DIR / f"D_{field}_crosstab_count.csv")

        ct_pct = pd.crosstab(valid[field], valid["period"],
                              normalize="columns") * 100
        ct_pct = ct_pct.reindex(columns=[p for p in PERIODS if p in ct_pct.columns])
        ct_pct.to_csv(OUTPUT_DIR / f"D_{field}_crosstab_pct.csv")

        fig, ax = plt.subplots(figsize=(8, max(4, len(ct_pct) * 0.5)))
        sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5)
        ax.set_title(f"{field} (% by period)")
        ax.set_ylabel(field)
        ax.set_xlabel("Period")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"D_{field}_heatmap.png", dpi=150)
        plt.close(fig)

        ct_test = pd.crosstab(valid[field], valid["period"])
        ct_test = ct_test.loc[ct_test.sum(axis=1) >= 5]
        if ct_test.shape[0] >= 2 and ct_test.shape[1] >= 2:
            chi2, p_val, dof, _ = stats.chi2_contingency(ct_test)
            n = ct_test.values.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(ct_test.shape) - 1)))
        else:
            chi2 = dof = p_val = cramers_v = np.nan
            n = 0

        chi_results.append({
            "field": field,
            "chi2": chi2,
            "dof": dof,
            "p_value": p_val,
            "n": n,
            "cramers_v": cramers_v,
        })
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {field}: chi2={chi2:.2f}, df={dof}, p={p_val:.6f}, V={cramers_v:.4f} {sig}")

    chi_df = pd.DataFrame(chi_results)
    chi_df.to_csv(OUTPUT_DIR / "D_chi_square_summary.csv", index=False)
    print("\nChi-square summary saved.")


# ---------------------------------------------------------------------------
# Module E: Poem-level summary
# ---------------------------------------------------------------------------

def module_e_poem_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODULE E: Poem-Level Summary")
    print("=" * 60)

    df_pron = df.dropna(subset=["person", "number"]).copy()
    df_pron["pronoun_class"] = df_pron.apply(classify_pronoun, axis=1)
    df_pron = df_pron[df_pron["pronoun_class"] != "other"]

    def dominant_class(series):
        vc = series.value_counts()
        return vc.index[0] if len(vc) > 0 else np.nan

    pron_agg = df_pron.groupby("original_id").agg(
        n_pronouns=("pronoun_class", "size"),
        dominant_pronoun_class=("pronoun_class", dominant_class),
        frac_1sg=("pronoun_class", lambda x: (x == "1sg").mean()),
        frac_1pl=("pronoun_class", lambda x: (x == "1pl").mean()),
        frac_2=("pronoun_class", lambda x: (x == "2").mean()),
        frac_3sg=("pronoun_class", lambda x: (x == "3sg").mean()),
        frac_3pl=("pronoun_class", lambda x: (x == "3pl").mean()),
    )

    poem_meta = df.groupby("original_id").agg(
        author=("author", "first"),
        period=("period", "first"),
        year=("year", "first"),
        perspective_primary=("poem_perspective_primary", "first"),
        perspective_secondary=("poem_perspective_secondary", "first"),
        n_sentence_rows=("original_id", "size"),
    )

    poems = poem_meta.join(pron_agg, how="left")
    poems["n_pronouns"] = poems["n_pronouns"].fillna(0).astype(int)
    poems["pronoun_density"] = poems["n_pronouns"] / poems["n_sentence_rows"]
    poems = poems.reset_index()
    poems.to_csv(OUTPUT_DIR / "E_poem_level_summary.csv", index=False)

    print(f"Total poems: {len(poems)}")
    for p in PERIODS:
        sub = poems[poems["period"] == p]
        print(f"\n  {p} ({len(sub)} poems):")
        print(f"    Mean pronouns/poem: {sub['n_pronouns'].mean():.1f}")
        print(f"    Mean pronoun density: {sub['pronoun_density'].mean():.3f}")
        print(f"    Dominant class distribution:")
        vc = sub["dominant_pronoun_class"].value_counts(normalize=True) * 100
        for cls, pct in vc.items():
            print(f"      {cls}: {pct:.1f}%")

    density_stats = poems.groupby("period")["pronoun_density"].agg(["mean", "std", "count"])
    density_stats.to_csv(OUTPUT_DIR / "E_pronoun_density_by_period.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    poems.boxplot(column="pronoun_density", by="period", ax=axes[0])
    axes[0].set_title("Pronoun Density by Period")
    axes[0].set_ylabel("Pronouns / Sentence Rows")
    axes[0].set_xlabel("Period")
    plt.sca(axes[0])
    plt.title("Pronoun Density by Period")

    dom_ct = pd.crosstab(poems["dominant_pronoun_class"], poems["period"],
                          normalize="columns") * 100
    dom_ct = dom_ct.reindex(columns=[p for p in PERIODS if p in dom_ct.columns])
    dom_ct.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Dominant Pronoun Class by Period (%)")
    axes[1].set_ylabel("% of poems")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="Period")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "E_poem_level_charts.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_and_remap()
    module_a_corpus_overview(df)
    module_b_pronoun_distribution(df)
    module_c_field_frequencies(df)
    module_d_crosstabs(df)
    module_e_poem_summary(df)

    print("\n" + "=" * 60)
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR}/")
    n_csv = len(list(OUTPUT_DIR.glob("*.csv")))
    n_png = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"  {n_csv} CSV files, {n_png} PNG files")
    print("=" * 60)


if __name__ == "__main__":
    main()
