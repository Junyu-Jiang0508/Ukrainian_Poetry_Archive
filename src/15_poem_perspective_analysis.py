"""Poem-level perspective primary/secondary by period."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.chdir(Path(__file__).resolve().parent.parent)

INPUT_CSV = Path("data/processed/gpt_annotation_public_run/gpt_annotation_detailed.csv")
OUTPUT_DIR = Path("outputs/15_poem_perspective")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_PERIODS = ["pre_2014", "2014_2021", "post_2022"]
PERSPECTIVE_ORDER = [
    "1st person singular",
    "1st person plural",
    "2nd person singular",
    "2nd person plural",
    "3rd person singular",
    "3rd person plural",
    "Mixed",
    "Impersonal/Other",
]


def load_poem_data():
    df = pd.read_csv(INPUT_CSV, low_memory=False, on_bad_lines="skip")
    poems = df.groupby("original_id").agg(
        author=("author", "first"),
        year=("year", "first"),
        temporal_period=("temporal_period", "first"),
        primary=("poem_perspective_primary", "first"),
        secondary=("poem_perspective_secondary", "first"),
        n_rows=("original_id", "size"),
    ).reset_index()
    poems = poems[poems["temporal_period"].isin(KNOWN_PERIODS)]
    print(f"Poems (known periods): {len(poems)}")
    return poems, df


def primary_perspective_analysis(poems: pd.DataFrame):
    """Distribution of primary perspective by period."""
    ct = pd.crosstab(poems["primary"], poems["temporal_period"])
    ct_pct = pd.crosstab(poems["primary"], poems["temporal_period"], normalize="columns") * 100
    ct.to_csv(OUTPUT_DIR / "primary_perspective_by_period_count.csv")
    ct_pct.to_csv(OUTPUT_DIR / "primary_perspective_by_period_pct.csv")
    print("\nPrimary perspective by period (%):")
    print(ct_pct.round(1).to_string())

    df_test = poems[poems["temporal_period"].isin(["2014_2021", "post_2022"])]
    ct_test = pd.crosstab(df_test["primary"], df_test["temporal_period"])
    ct_test = ct_test.loc[ct_test.sum(axis=1) >= 5]
    chi2, p, dof, _ = stats.chi2_contingency(ct_test)
    print(f"\nChi-square (primary x period): Chi2={chi2:.3f}, df={dof}, p={p:.6f}")

    d1 = df_test[df_test["temporal_period"] == "2014_2021"]
    d2 = df_test[df_test["temporal_period"] == "post_2022"]
    n1, n2 = len(d1), len(d2)
    results = []
    for persp in ct_test.index:
        c1 = (d1["primary"] == persp).sum()
        c2 = (d2["primary"] == persp).sum()
        p1, p2 = c1 / n1, c2 / n2
        p_pool = (c1 + c2) / (n1 + n2)
        if p_pool in (0, 1):
            z, pval = 0, 1
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p2 - p1) / se
            pval = 2 * (1 - stats.norm.cdf(abs(z)))
        results.append({
            "perspective": persp,
            "pct_2014_2021": p1 * 100,
            "pct_post_2022": p2 * 100,
            "diff_pp": (p2 - p1) * 100,
            "z": z,
            "p_value": pval,
        })
    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / "primary_perspective_z_tests.csv", index=False)
    print("\nPrimary perspective z-tests:")
    print(res.to_string(index=False))
    return ct_pct


def secondary_perspective_analysis(poems: pd.DataFrame):
    """Analyze secondary perspective presence and type."""
    poems = poems.copy()
    poems["has_secondary"] = poems["secondary"].notna() & (poems["secondary"].astype(str).str.strip() != "")

    for period in KNOWN_PERIODS:
        d = poems[poems["temporal_period"] == period]
        pct = d["has_secondary"].mean() * 100
        print(f"  {period}: {pct:.1f}% have secondary perspective (n={len(d)})")

    df_test = poems[poems["temporal_period"].isin(["2014_2021", "post_2022"])]
    d1 = df_test[df_test["temporal_period"] == "2014_2021"]
    d2 = df_test[df_test["temporal_period"] == "post_2022"]
    ct = pd.crosstab(df_test["has_secondary"], df_test["temporal_period"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f"\nSecondary presence Chi2={chi2:.3f}, p={p:.6f}")

    df_with_sec = poems[poems["has_secondary"]].copy()
    if len(df_with_sec) > 0:
        ct_sec = pd.crosstab(df_with_sec["secondary"], df_with_sec["temporal_period"])
        ct_sec_pct = pd.crosstab(
            df_with_sec["secondary"], df_with_sec["temporal_period"], normalize="columns"
        ) * 100
        ct_sec.to_csv(OUTPUT_DIR / "secondary_perspective_by_period_count.csv")
        ct_sec_pct.to_csv(OUTPUT_DIR / "secondary_perspective_by_period_pct.csv")
        print("\nSecondary perspective distribution (among poems with secondary):")
        print(ct_sec_pct.round(1).to_string())


def perspective_combination(poems: pd.DataFrame):
    """Analyze primary+secondary perspective combinations."""
    poems = poems.copy()
    poems["combo"] = poems["primary"].astype(str) + " + " + poems["secondary"].fillna("None").astype(str)
    combo_vc = poems.groupby(["temporal_period", "combo"]).size().reset_index(name="count")
    combo_vc.to_csv(OUTPUT_DIR / "perspective_combinations.csv", index=False)

    top_combos = poems["combo"].value_counts().head(15).index
    df_top = poems[poems["combo"].isin(top_combos)]
    ct = pd.crosstab(df_top["combo"], df_top["temporal_period"], normalize="columns") * 100
    ct.to_csv(OUTPUT_DIR / "top_perspective_combos_pct.csv")

    fig, ax = plt.subplots(figsize=(12, 8))
    ct_sorted = ct.reindex(columns=[p for p in KNOWN_PERIODS if p in ct.columns])
    sns.heatmap(ct_sorted, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Top 15 perspective combinations by period (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "perspective_combos_heatmap.png", dpi=150)
    plt.close(fig)


def visualize_primary(ct_pct: pd.DataFrame):
    ct_pct = ct_pct.reindex(columns=[p for p in KNOWN_PERIODS if p in ct_pct.columns])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ct_pct.T.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set3")
    axes[0].set_title("Primary poem perspective by period")
    axes[0].set_ylabel("% of poems")
    axes[0].legend(title="Perspective", bbox_to_anchor=(1.02, 1), fontsize=7)
    axes[0].tick_params(axis="x", rotation=0)

    for persp in ["1st person singular", "1st person plural", "Mixed"]:
        if persp in ct_pct.index:
            axes[1].plot(ct_pct.columns, ct_pct.loc[persp].values, marker="o", label=persp)
    axes[1].set_title("Key perspective trends")
    axes[1].set_ylabel("% of poems")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "primary_perspective_by_period.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    poems, df = load_poem_data()
    ct_pct = primary_perspective_analysis(poems)
    print("\nSecondary perspective presence by period:")
    secondary_perspective_analysis(poems)
    perspective_combination(poems)
    visualize_primary(ct_pct)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
