"""RQ1: we_type over time (tests + plots)."""

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
TIMESERIES_CSV = Path("outputs/08_change_point_detection/pronoun_timeseries_adaptive.csv")
OUTPUT_DIR = Path("outputs/13_rq1_we_type")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_PERIODS = ["pre_2014", "2014_2021", "post_2022"]
WE_TYPE_ORDER = [
    "exclusive_ingroup",
    "inclusive_addressee",
    "speaker_exclusive",
    "generic_universal",
    "mixed_we",
    "not_applicable",
]
WE_TYPES_SUBSTANTIVE = [
    "exclusive_ingroup",
    "inclusive_addressee",
    "speaker_exclusive",
    "generic_universal",
    "mixed_we",
]

BREAK_2014 = pd.Timestamp("2014-02-01")
BREAK_2022 = pd.Timestamp("2022-02-01")


def load_data():
    df = pd.read_csv(INPUT_CSV, low_memory=False, on_bad_lines="skip")
    df_we = df[
        (df["person"].astype(str).str.contains("1st", na=False)) &
        (df["number"].astype(str).str.contains("Plur", na=False))
    ].copy()
    df_we = df_we[df_we["temporal_period"].isin(KNOWN_PERIODS)]
    print(f"Total 1pl rows (known periods): {len(df_we)}")
    print(f"  Poems with 1pl: {df_we['original_id'].nunique()}")
    return df, df_we


def chi_square_test(df_we: pd.DataFrame):
    """Chi-square test: we_type x temporal_period (excluding pre_2014 if too sparse)."""
    df_test = df_we[df_we["temporal_period"].isin(["2014_2021", "post_2022"])]
    ct = pd.crosstab(df_test["we_type"], df_test["temporal_period"])
    ct = ct.loc[ct.sum(axis=1) >= 5]  # drop categories with < 5 total

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    result = {
        "test": "chi2",
        "statistic": chi2,
        "p_value": p,
        "dof": dof,
        "n": ct.values.sum(),
    }
    print(f"\nChi-square test (we_type x period, 2014_2021 vs post_2022):")
    print(f"  Chi2 = {chi2:.3f}, df = {dof}, p = {p:.6f}, n = {ct.values.sum()}")

    ct_full = pd.crosstab(df_we["we_type"], df_we["temporal_period"])
    ct_pct = pd.crosstab(df_we["we_type"], df_we["temporal_period"], normalize="columns") * 100
    ct_full.to_csv(OUTPUT_DIR / "we_type_by_period_count.csv")
    ct_pct.to_csv(OUTPUT_DIR / "we_type_by_period_pct.csv")
    return result


def proportion_z_tests(df_we: pd.DataFrame):
    """Z-tests for each we_type proportion between 2014_2021 and post_2022."""
    d1 = df_we[df_we["temporal_period"] == "2014_2021"]
    d2 = df_we[df_we["temporal_period"] == "post_2022"]
    n1, n2 = len(d1), len(d2)
    results = []
    for wt in WE_TYPES_SUBSTANTIVE:
        c1 = (d1["we_type"] == wt).sum()
        c2 = (d2["we_type"] == wt).sum()
        p1, p2 = c1 / n1, c2 / n2
        p_pool = (c1 + c2) / (n1 + n2)
        if p_pool == 0 or p_pool == 1:
            z, pval = 0, 1
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p2 - p1) / se
            pval = 2 * (1 - stats.norm.cdf(abs(z)))
        results.append({
            "we_type": wt,
            "pct_2014_2021": p1 * 100,
            "pct_post_2022": p2 * 100,
            "diff_pp": (p2 - p1) * 100,
            "z": z,
            "p_value": pval,
            "n_2014_2021": c1,
            "n_post_2022": c2,
        })
    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / "we_type_proportion_z_tests.csv", index=False)
    print("\nProportion z-tests (2014_2021 → post_2022):")
    print(res[["we_type", "pct_2014_2021", "pct_post_2022", "diff_pp", "z", "p_value"]].to_string(index=False))
    return res


def poem_level_we_analysis(df_all: pd.DataFrame, df_we: pd.DataFrame):
    """Poem-level aggregation: what fraction of each poem's pronouns are 1pl, and what we_type?"""
    df_known = df_all[df_all["temporal_period"].isin(KNOWN_PERIODS)].copy()
    df_with_pron = df_known.dropna(subset=["person"])

    poem_total = df_with_pron.groupby("original_id").size().rename("n_pronouns")
    poem_1pl = df_we.groupby("original_id").size().rename("n_1pl")

    poems = pd.DataFrame({"n_pronouns": poem_total}).join(poem_1pl, how="left").fillna(0)
    poems["frac_1pl"] = poems["n_1pl"] / poems["n_pronouns"]

    meta = df_known.groupby("original_id").agg(
        temporal_period=("temporal_period", "first"),
        year=("year", "first"),
        author=("author", "first"),
    )
    poems = poems.join(meta)

    we_type_dom = (
        df_we[df_we["we_type"] != "not_applicable"]
        .groupby("original_id")["we_type"]
        .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else np.nan)
        .rename("dominant_we_type")
    )
    poems = poems.join(we_type_dom, how="left")
    poems.to_csv(OUTPUT_DIR / "poem_level_we_stats.csv")
    print(f"\nPoem-level: {len(poems)} poems, {(poems['n_1pl'] > 0).sum()} with 1pl")
    return poems


def stacked_area_by_period(df_we: pd.DataFrame):
    """Stacked bar chart of we_type proportions by period."""
    ct_pct = pd.crosstab(
        df_we["we_type"], df_we["temporal_period"], normalize="columns"
    ) * 100
    ct_pct = ct_pct.reindex(columns=[p for p in KNOWN_PERIODS if p in ct_pct.columns])
    substantive = ct_pct.drop("not_applicable", errors="ignore")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    substantive.T.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2")
    axes[0].set_title("We-type distribution by period\n(excluding not_applicable)")
    axes[0].set_ylabel("Percentage of 1pl tokens (%)")
    axes[0].set_xlabel("")
    axes[0].legend(title="we_type", bbox_to_anchor=(1.02, 1), fontsize=8)
    axes[0].tick_params(axis="x", rotation=0)

    for wt in WE_TYPES_SUBSTANTIVE:
        if wt in ct_pct.index:
            vals = ct_pct.loc[wt]
            axes[1].plot(vals.index, vals.values, marker="o", label=wt)
    axes[1].set_title("We-type trends across periods")
    axes[1].set_ylabel("% of all 1pl tokens")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "we_type_by_period.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    among_substantive = substantive.div(substantive.sum()) * 100
    among_substantive.to_csv(OUTPUT_DIR / "we_type_among_substantive_pct.csv")


def temporal_trend_with_intervals(df_all: pd.DataFrame, df_we: pd.DataFrame):
    """Map we_type data onto adaptive intervals for finer temporal resolution."""
    try:
        intervals = pd.read_csv(INTERVALS_CSV)
    except FileNotFoundError:
        print("Warning: adaptive_intervals.csv not found, skipping interval-level analysis")
        return

    intervals["start_date"] = pd.to_datetime(intervals["start_date"])
    intervals["end_date"] = pd.to_datetime(intervals["end_date"])

    df_dated = df_we.copy()
    df_dated["year_num"] = pd.to_numeric(df_dated["year"], errors="coerce")
    df_dated = df_dated.dropna(subset=["year_num"])

    all_dated = df_all.dropna(subset=["person"]).copy()
    all_dated["year_num"] = pd.to_numeric(all_dated["year"], errors="coerce")
    all_dated = all_dated.dropna(subset=["year_num"])

    rows = []
    for _, iv in intervals.iterrows():
        mask_we = (df_dated["year_num"] >= iv["start_date"].year) & \
                  (df_dated["year_num"] <= iv["end_date"].year)
        mask_all = (all_dated["year_num"] >= iv["start_date"].year) & \
                   (all_dated["year_num"] <= iv["end_date"].year)

        chunk_we = df_dated[mask_we]
        chunk_all = all_dated[mask_all]
        n_total = len(chunk_all)
        n_we = len(chunk_we)

        row = {
            "interval_id": iv["interval_id"],
            "interval_label": iv["interval_label"],
            "n_all_pronouns": n_total,
            "n_1pl": n_we,
            "frac_1pl": n_we / n_total if n_total > 0 else 0,
        }
        for wt in WE_TYPES_SUBSTANTIVE:
            cnt = (chunk_we["we_type"] == wt).sum()
            row[f"n_{wt}"] = cnt
            row[f"frac_{wt}"] = cnt / n_we if n_we > 0 else 0
        rows.append(row)

    ts = pd.DataFrame(rows)
    ts.to_csv(OUTPUT_DIR / "we_type_timeseries_by_interval.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = ts["interval_id"]

    axes[0].plot(x, ts["frac_1pl"] * 100, "k-o", markersize=4, label="1pl fraction")
    axes[0].axvline(x=2, color="orange", ls="--", alpha=0.7, label="2014 break")
    axes[0].axvline(x=24, color="red", ls="--", alpha=0.7, label="2022 break")
    axes[0].set_ylabel("1pl as % of all pronouns")
    axes[0].set_title("First-person plural proportion over time")
    axes[0].legend(fontsize=8)

    for wt in ["exclusive_ingroup", "inclusive_addressee", "mixed_we"]:
        col = f"frac_{wt}"
        if col in ts.columns:
            axes[1].plot(x, ts[col] * 100, marker="o", markersize=3, label=wt)
    axes[1].axvline(x=2, color="orange", ls="--", alpha=0.7)
    axes[1].axvline(x=24, color="red", ls="--", alpha=0.7)
    axes[1].set_xlabel("Adaptive interval ID")
    axes[1].set_ylabel("% of 1pl tokens")
    axes[1].set_title("We-type sub-categories over time (among 1pl)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "we_type_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"\nInterval-level timeseries saved ({len(ts)} intervals)")
    return ts


def exclusive_ingroup_regression(df_we: pd.DataFrame):
    """Simple WLS breakpoint regression on exclusive_ingroup proportion (poem-level)."""
    try:
        from statsmodels.regression.linear_model import WLS
    except ImportError:
        print("statsmodels not available, skipping regression")
        return

    df_dated = df_we.copy()
    df_dated["year_num"] = pd.to_numeric(df_dated["year"], errors="coerce")
    df_dated = df_dated.dropna(subset=["year_num"])

    poems_we = df_dated.groupby("original_id").agg(
        n_1pl=("we_type", "size"),
        n_exclusive=("we_type", lambda x: (x == "exclusive_ingroup").sum()),
        year=("year_num", "first"),
    )
    poems_we["frac_exclusive"] = poems_we["n_exclusive"] / poems_we["n_1pl"]
    poems_we = poems_we[poems_we["n_1pl"] >= 2]

    poems_we["time"] = poems_we["year"].rank(method="dense")
    poems_we["post_2014"] = (poems_we["year"] >= 2014).astype(int)
    poems_we["post_2022"] = (poems_we["year"] >= 2022).astype(int)
    poems_we["time_post_2014"] = poems_we["time"] * poems_we["post_2014"]
    poems_we["time_post_2022"] = poems_we["time"] * poems_we["post_2022"]

    X = poems_we[["time", "post_2014", "post_2022", "time_post_2014", "time_post_2022"]]
    X = X.assign(const=1)
    y = poems_we["frac_exclusive"]
    w = np.sqrt(poems_we["n_1pl"])

    try:
        model = WLS(y, X, weights=w).fit()
        summary_text = model.summary().as_text()
        with open(OUTPUT_DIR / "exclusive_ingroup_regression_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)

        coefs = model.params.to_frame("coef").join(model.pvalues.to_frame("p_value"))
        coefs.to_csv(OUTPUT_DIR / "exclusive_ingroup_regression_coefs.csv")
        r2 = model.rsquared
        p2022 = model.pvalues.get("post_2022", float("nan"))
        c2022 = model.params.get("post_2022", float("nan"))
        print(f"\nExclusive-ingroup regression:")
        print(f"  R-squared = {r2:.3f}")
        print(f"  post_2022 coef = {c2022:.4f}, p = {p2022:.6f}")
    except Exception as e:
        print(f"Regression failed: {e}")


def main():
    df_all, df_we = load_data()
    chi_square_test(df_we)
    proportion_z_tests(df_we)
    poems = poem_level_we_analysis(df_all, df_we)
    stacked_area_by_period(df_we)
    temporal_trend_with_intervals(df_all, df_we)
    exclusive_ingroup_regression(df_we)
    print(f"\nAll RQ1 outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
