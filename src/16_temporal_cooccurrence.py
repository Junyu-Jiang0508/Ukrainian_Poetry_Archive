"""RQ3: pronoun–concept PMI by period, heatmaps, bootstrap."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import os

os.chdir(Path(__file__).resolve().parent.parent)

INPUT_CSV = Path("outputs/02_pronoun_cooccurrence/pronoun_cooccurrence_with_date.csv")
OUTPUT_DIR = Path("outputs/16_temporal_cooccurrence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BREAK_2014 = pd.Timestamp("2014-02-01")
BREAK_2022 = pd.Timestamp("2022-02-01")

CORE_PRONOUNS = ["я", "ми", "ти", "він", "вона", "вони"]
PRONOUN_FORMS = {
    "я":    ["я", "мене", "мені", "мною", "мій"],
    "ми":   ["ми", "нас", "нам", "нами", "наш"],
    "ти":   ["ти", "тебе", "тобі", "тобою", "твій"],
    "він":  ["він"],
    "вона": ["вона"],
    "вони": ["вони"],
}

CONCEPT_GROUPS = {
    "land_space":     ["земля", "місто", "дім", "світ", "небо"],
    "war_conflict":   ["війна", "смерть", "кров", "ворог", "зброя"],
    "kin_social":     ["дитина", "мати", "бог", "друг", "чоловік"],
    "emotion_exist":  ["любити", "жити", "життя", "час", "серце"],
    "agency_voice":   ["казати", "знати", "могти", "писати", "говорити"],
}
ALL_CONCEPTS = [w for words in CONCEPT_GROUPS.values() for w in words]


def load_data():
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    print(f"Loaded {len(df)} co-occurrence records")
    return df


def assign_period(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df["date"] < BREAK_2014,
        (df["date"] >= BREAK_2014) & (df["date"] < BREAK_2022),
        df["date"] >= BREAK_2022,
    ]
    choices = ["pre_2014", "2014_2021", "post_2022"]
    df["period"] = np.select(conditions, choices, default="unknown")
    for p in choices:
        n = df[df["period"] == p]["count"].sum()
        print(f"  {p}: {n} total co-occurrences")
    return df


def map_pronoun_lemma(pronoun: str) -> str:
    """Map inflected pronoun forms to lemma."""
    for lemma, forms in PRONOUN_FORMS.items():
        if pronoun in forms:
            return lemma
    return pronoun


def compute_pmi(df_period: pd.DataFrame, core_pronouns, concepts):
    """Compute PMI for pronoun-concept pairs within a period."""
    df_period = df_period.copy()
    df_period["pronoun_lemma"] = df_period["pronoun"].apply(map_pronoun_lemma)

    total = df_period["count"].sum()
    if total == 0:
        return pd.DataFrame()

    pron_totals = df_period.groupby("pronoun_lemma")["count"].sum()
    word_totals = df_period.groupby("word")["count"].sum()

    rows = []
    for pron in core_pronouns:
        for concept in concepts:
            pair_mask = (df_period["pronoun_lemma"] == pron) & (df_period["word"] == concept)
            pair_count = df_period.loc[pair_mask, "count"].sum()

            p_pron = pron_totals.get(pron, 0) / total
            p_word = word_totals.get(concept, 0) / total
            p_pair = pair_count / total

            if p_pair > 0 and p_pron > 0 and p_word > 0:
                pmi = np.log2(p_pair / (p_pron * p_word))
                npmi = pmi / (-np.log2(p_pair)) if p_pair < 1 else 0
            else:
                pmi = 0
                npmi = 0

            rows.append({
                "pronoun": pron,
                "concept": concept,
                "pair_count": pair_count,
                "pmi": pmi,
                "npmi": npmi,
            })
    return pd.DataFrame(rows)


def temporal_pmi_analysis(df: pd.DataFrame):
    """Compute PMI for each period and compare."""
    periods = ["pre_2014", "2014_2021", "post_2022"]
    all_pmi = {}

    for period in periods:
        df_p = df[df["period"] == period]
        pmi_df = compute_pmi(df_p, CORE_PRONOUNS, ALL_CONCEPTS)
        pmi_df["period"] = period
        all_pmi[period] = pmi_df
        pmi_df.to_csv(OUTPUT_DIR / f"pmi_{period}.csv", index=False)

    combined = pd.concat(all_pmi.values(), ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "pmi_all_periods.csv", index=False)

    pivot_pmi = combined.pivot_table(
        index=["pronoun", "concept"], columns="period", values="npmi", fill_value=0
    )
    pivot_pmi = pivot_pmi.reindex(columns=[p for p in periods if p in pivot_pmi.columns])
    pivot_pmi.to_csv(OUTPUT_DIR / "npmi_pivot.csv")

    if "2014_2021" in pivot_pmi.columns and "post_2022" in pivot_pmi.columns:
        pivot_pmi["delta_2022"] = pivot_pmi["post_2022"] - pivot_pmi["2014_2021"]
        top_increase = pivot_pmi.nlargest(15, "delta_2022")
        top_decrease = pivot_pmi.nsmallest(15, "delta_2022")
        top_increase.to_csv(OUTPUT_DIR / "top_npmi_increases_2022.csv")
        top_decrease.to_csv(OUTPUT_DIR / "top_npmi_decreases_2022.csv")
        print("\nTop NPMI increases (2014_2021 -> post_2022):")
        print(top_increase[["delta_2022"]].to_string())
        print("\nTop NPMI decreases:")
        print(top_decrease[["delta_2022"]].to_string())

    return combined


def pmi_heatmaps(combined: pd.DataFrame):
    """Generate heatmaps for each pronoun across concepts and periods."""
    periods = ["pre_2014", "2014_2021", "post_2022"]

    for pron in CORE_PRONOUNS:
        d = combined[(combined["pronoun"] == pron) & (combined["concept"].isin(ALL_CONCEPTS))]
        if len(d) == 0:
            continue
        pivot = d.pivot_table(index="concept", columns="period", values="npmi", fill_value=0)
        pivot = pivot.reindex(columns=[p for p in periods if p in pivot.columns])

        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.35)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
        ax.set_title(f"NPMI: {pron} x concepts across periods")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"npmi_heatmap_{pron}.png", dpi=150)
        plt.close(fig)

    for group_name, concepts in CONCEPT_GROUPS.items():
        for period in periods:
            d = combined[(combined["period"] == period) & (combined["concept"].isin(concepts))]
            pivot = d.pivot_table(index="concept", columns="pronoun", values="npmi", fill_value=0)
            if len(pivot) == 0:
                continue
            pivot = pivot.reindex(columns=[p for p in CORE_PRONOUNS if p in pivot.columns])
            fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.5)))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
            ax.set_title(f"{group_name} NPMI ({period})")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"npmi_{group_name}_{period}.png", dpi=150)
            plt.close(fig)


def concept_group_summary(combined: pd.DataFrame):
    """Aggregate NPMI by concept group for a cleaner comparison."""
    periods = ["pre_2014", "2014_2021", "post_2022"]
    rows = []
    for pron in CORE_PRONOUNS:
        for group_name, concepts in CONCEPT_GROUPS.items():
            for period in periods:
                d = combined[
                    (combined["pronoun"] == pron)
                    & (combined["period"] == period)
                    & (combined["concept"].isin(concepts))
                ]
                mean_npmi = d["npmi"].mean() if len(d) > 0 else 0
                rows.append({
                    "pronoun": pron,
                    "concept_group": group_name,
                    "period": period,
                    "mean_npmi": mean_npmi,
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "concept_group_mean_npmi.csv", index=False)

    pivot = summary.pivot_table(
        index=["pronoun", "concept_group"],
        columns="period",
        values="mean_npmi",
    )
    pivot = pivot.reindex(columns=[p for p in periods if p in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0, ax=ax)
    ax.set_title("Mean NPMI by pronoun x concept-group x period")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "concept_group_npmi_summary.png", dpi=150)
    plt.close(fig)

    return summary


def bootstrap_pmi_diff(df: pd.DataFrame, n_bootstrap=1000):
    """Bootstrap test for NPMI difference between 2014_2021 and post_2022 for key pairs."""
    key_pairs = [
        ("ми", "війна"), ("ми", "земля"), ("ми", "дім"),
        ("я", "писати"), ("я", "любити"),
        ("ти", "бог"), ("ти", "смерть"),
        ("вони", "земля"), ("вони", "дитина"),
    ]

    df1 = df[df["period"] == "2014_2021"]
    df2 = df[df["period"] == "post_2022"]
    results = []

    for pron, concept in key_pairs:
        pmi1 = compute_pmi(df1, [pron], [concept])
        pmi2 = compute_pmi(df2, [pron], [concept])
        if len(pmi1) == 0 or len(pmi2) == 0:
            continue

        obs_diff = pmi2.iloc[0]["npmi"] - pmi1.iloc[0]["npmi"]

        df_combined = pd.concat([df1, df2])
        boot_diffs = []
        for _ in range(n_bootstrap):
            sample = df_combined.sample(frac=1, replace=True)
            half = len(sample) // 2
            s1, s2 = sample.iloc[:half], sample.iloc[half:]
            b1 = compute_pmi(s1, [pron], [concept])
            b2 = compute_pmi(s2, [pron], [concept])
            if len(b1) > 0 and len(b2) > 0:
                boot_diffs.append(b2.iloc[0]["npmi"] - b1.iloc[0]["npmi"])

        if len(boot_diffs) > 0:
            ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
            p_val = np.mean(np.array(boot_diffs) >= abs(obs_diff)) + \
                    np.mean(np.array(boot_diffs) <= -abs(obs_diff))
        else:
            ci_low = ci_high = p_val = np.nan

        results.append({
            "pronoun": pron,
            "concept": concept,
            "npmi_2014_2021": pmi1.iloc[0]["npmi"],
            "npmi_post_2022": pmi2.iloc[0]["npmi"],
            "diff": obs_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_p": p_val,
        })

    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / "bootstrap_npmi_diff.csv", index=False)
    print("\nBootstrap NPMI difference tests:")
    print(res.to_string(index=False))
    return res


def main():
    df = load_data()
    df = assign_period(df)
    combined = temporal_pmi_analysis(df)
    pmi_heatmaps(combined)
    concept_group_summary(combined)
    print("\nRunning bootstrap tests (may take a minute)...")
    bootstrap_pmi_diff(df, n_bootstrap=500)
    print(f"\nAll RQ3 outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
