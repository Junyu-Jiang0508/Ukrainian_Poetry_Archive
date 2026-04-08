"""RQ2: addressee_type and 2nd-person over time."""

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
INTERVALS_CSV = Path("outputs/08_change_point_detection/adaptive_intervals.csv")
OUTPUT_DIR = Path("outputs/14_rq2_addressee")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_PERIODS = ["pre_2014", "2014_2021", "post_2022"]
ADDRESSEE_TYPES_SUBSTANTIVE = [
    "lyric_self_2nd",
    "specific_individual",
    "collective_nation",
    "enemy_other",
    "god_nature_abstract",
    "absent_beloved",
    "europe_world",
]


def load_data():
    df = pd.read_csv(INPUT_CSV, low_memory=False, on_bad_lines="skip")
    df_2nd = df[df["person"].astype(str).str.contains("2nd", na=False)].copy()
    df_2nd = df_2nd[df_2nd["temporal_period"].isin(KNOWN_PERIODS)]
    print(f"Total 2nd-person rows (known periods): {len(df_2nd)}")
    print(f"  Poems with 2nd: {df_2nd['original_id'].nunique()}")
    return df, df_2nd


def chi_square_and_proportions(df_2nd: pd.DataFrame):
    """Chi-square test + proportion z-tests for addressee_type across periods."""
    df_test = df_2nd[df_2nd["temporal_period"].isin(["2014_2021", "post_2022"])]

    ct = pd.crosstab(df_test["addressee_type"], df_test["temporal_period"])
    ct_valid = ct.loc[ct.sum(axis=1) >= 5]
    chi2, p, dof, _ = stats.chi2_contingency(ct_valid)
    print(f"\nChi-square (addressee_type x period, 2014_2021 vs post_2022):")
    print(f"  Chi2 = {chi2:.3f}, df = {dof}, p = {p:.6f}, n = {ct_valid.values.sum()}")

    ct_all = pd.crosstab(df_2nd["addressee_type"], df_2nd["temporal_period"])
    ct_pct = pd.crosstab(df_2nd["addressee_type"], df_2nd["temporal_period"], normalize="columns") * 100
    ct_all.to_csv(OUTPUT_DIR / "addressee_type_by_period_count.csv")
    ct_pct.to_csv(OUTPUT_DIR / "addressee_type_by_period_pct.csv")

    d1 = df_test[df_test["temporal_period"] == "2014_2021"]
    d2 = df_test[df_test["temporal_period"] == "post_2022"]
    n1, n2 = len(d1), len(d2)
    results = []
    for at in ADDRESSEE_TYPES_SUBSTANTIVE:
        c1 = (d1["addressee_type"] == at).sum()
        c2 = (d2["addressee_type"] == at).sum()
        p1, p2 = c1 / n1, c2 / n2
        p_pool = (c1 + c2) / (n1 + n2)
        if p_pool in (0, 1):
            z, pval = 0, 1
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p2 - p1) / se
            pval = 2 * (1 - stats.norm.cdf(abs(z)))
        results.append({
            "addressee_type": at,
            "pct_2014_2021": p1 * 100,
            "pct_post_2022": p2 * 100,
            "diff_pp": (p2 - p1) * 100,
            "z": z,
            "p_value": pval,
        })
    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / "addressee_proportion_z_tests.csv", index=False)
    print("\nProportion z-tests (addressee_type):")
    print(res.to_string(index=False))
    return res


def referent_cross_analysis(df_2nd: pd.DataFrame):
    """Cross-tabulate addressee_type x dominant_referent_level for 2nd-person tokens."""
    df_test = df_2nd[df_2nd["temporal_period"].isin(["2014_2021", "post_2022"])]

    for period in ["2014_2021", "post_2022"]:
        d = df_test[df_test["temporal_period"] == period]
        ct = pd.crosstab(d["addressee_type"], d["dominant_referent_level"])
        ct.to_csv(OUTPUT_DIR / f"addressee_x_referent_{period}.csv")

    ct_diff_rows = []
    for at in ADDRESSEE_TYPES_SUBSTANTIVE:
        for rl in df_test["dominant_referent_level"].unique():
            d1 = df_test[(df_test["temporal_period"] == "2014_2021")]
            d2 = df_test[(df_test["temporal_period"] == "post_2022")]
            c1 = ((d1["addressee_type"] == at) & (d1["dominant_referent_level"] == rl)).sum()
            c2 = ((d2["addressee_type"] == at) & (d2["dominant_referent_level"] == rl)).sum()
            ct_diff_rows.append({
                "addressee_type": at,
                "referent_level": rl,
                "count_2014_2021": c1,
                "count_post_2022": c2,
                "pct_2014_2021": c1 / len(d1) * 100,
                "pct_post_2022": c2 / len(d2) * 100,
            })
    pd.DataFrame(ct_diff_rows).to_csv(OUTPUT_DIR / "addressee_referent_cross_comparison.csv", index=False)


def visualize(df_2nd: pd.DataFrame):
    """Bar charts and heatmaps for addressee_type by period."""
    ct_pct = pd.crosstab(
        df_2nd["addressee_type"], df_2nd["temporal_period"], normalize="columns"
    ) * 100
    ct_pct = ct_pct.reindex(columns=[p for p in KNOWN_PERIODS if p in ct_pct.columns])
    substantive = ct_pct.drop("not_applicable", errors="ignore")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    substantive.T.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10")
    axes[0].set_title("Addressee-type distribution by period\n(2nd person, excl. not_applicable)")
    axes[0].set_ylabel("% of 2nd-person tokens")
    axes[0].legend(title="addressee_type", bbox_to_anchor=(1.02, 1), fontsize=7)
    axes[0].tick_params(axis="x", rotation=0)

    for at in ADDRESSEE_TYPES_SUBSTANTIVE:
        if at in ct_pct.index:
            axes[1].plot(ct_pct.columns, ct_pct.loc[at].values, marker="o", label=at)
    axes[1].set_title("Addressee-type trends across periods")
    axes[1].set_ylabel("% of all 2nd-person tokens")
    axes[1].legend(fontsize=7)
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "addressee_type_by_period.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    among = substantive.div(substantive.sum()) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(among, annot=True, fmt=".1f", cmap="Blues", ax=ax)
    ax.set_title("Addressee-type composition (% among substantive 2nd-person)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "addressee_type_heatmap.png", dpi=150)
    plt.close(fig)
    among.to_csv(OUTPUT_DIR / "addressee_among_substantive_pct.csv")


def formal_informal_split(df_2nd: pd.DataFrame):
    """Analyze singular (ty) vs plural (vy) 2nd-person split by period."""
    df_test = df_2nd[df_2nd["temporal_period"].isin(["2014_2021", "post_2022"])].copy()
    df_valid = df_test.dropna(subset=["number"])

    ct = pd.crosstab(df_valid["number"], df_valid["temporal_period"])
    ct_pct = pd.crosstab(df_valid["number"], df_valid["temporal_period"], normalize="columns") * 100
    ct.to_csv(OUTPUT_DIR / "2nd_person_number_by_period.csv")
    ct_pct.to_csv(OUTPUT_DIR / "2nd_person_number_by_period_pct.csv")
    print("\n2nd-person Singular vs Plural by period (%):")
    print(ct_pct.round(1).to_string())

    for num_val in ["Singular", "Plural"]:
        d = df_valid[df_valid["number"] == num_val]
        if len(d) > 0:
            ct_addr = pd.crosstab(d["addressee_type"], d["temporal_period"], normalize="columns") * 100
            ct_addr.to_csv(OUTPUT_DIR / f"addressee_by_{num_val.lower()}_2nd.csv")


def poem_level_addressee(df_all: pd.DataFrame, df_2nd: pd.DataFrame):
    """Poem-level: dominant addressee type for poems with 2nd-person."""
    poems_2nd = df_2nd[df_2nd["addressee_type"] != "not_applicable"].groupby("original_id").agg(
        n_2nd=("addressee_type", "size"),
        dominant_addressee=("addressee_type", lambda x: x.value_counts().index[0]),
        temporal_period=("temporal_period", "first"),
        author=("author", "first"),
    ).reset_index()
    poems_2nd.to_csv(OUTPUT_DIR / "poem_level_addressee.csv", index=False)

    ct = pd.crosstab(poems_2nd["dominant_addressee"], poems_2nd["temporal_period"])
    ct_pct = pd.crosstab(poems_2nd["dominant_addressee"], poems_2nd["temporal_period"], normalize="columns") * 100
    ct.to_csv(OUTPUT_DIR / "poem_dominant_addressee_by_period.csv")
    ct_pct.to_csv(OUTPUT_DIR / "poem_dominant_addressee_by_period_pct.csv")
    print(f"\nPoem-level dominant addressee ({len(poems_2nd)} poems):")
    print(ct_pct.round(1).to_string())


def main():
    df_all, df_2nd = load_data()
    chi_square_and_proportions(df_2nd)
    referent_cross_analysis(df_2nd)
    visualize(df_2nd)
    formal_informal_split(df_2nd)
    poem_level_addressee(df_all, df_2nd)
    print(f"\nAll RQ2 outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
