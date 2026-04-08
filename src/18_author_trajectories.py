"""Author-level pronoun trajectories and clustering."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

os.chdir(Path(__file__).resolve().parent.parent)

INPUT_CSV = Path("data/processed/gpt_annotation_public_run/gpt_annotation_detailed.csv")
OUTPUT_DIR = Path("outputs/18_author_trajectories")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_PERIODS = ["pre_2014", "2014_2021", "post_2022"]
MIN_POEMS = 10
MIN_POEMS_PER_PERIOD = 3


def load_data():
    df = pd.read_csv(INPUT_CSV, low_memory=False, on_bad_lines="skip")
    df = df[df["temporal_period"].isin(KNOWN_PERIODS)]
    df = df.dropna(subset=["person", "number"])

    def classify(row):
        p, n = str(row["person"]), str(row["number"])
        if "1st" in p and "Sing" in n: return "1sg"
        if "1st" in p and "Plur" in n: return "1pl"
        if "2nd" in p: return "2"
        if "3rd" in p and "Sing" in n: return "3sg"
        if "3rd" in p and "Plur" in n: return "3pl"
        return "other"

    df["pronoun_class"] = df.apply(classify, axis=1)
    df = df[df["pronoun_class"] != "other"]
    return df


def author_period_profiles(df: pd.DataFrame):
    """Compute pronoun proportions per author per period."""
    profiles = df.groupby(["author", "temporal_period", "pronoun_class"]).size().reset_index(name="count")
    totals = df.groupby(["author", "temporal_period"]).size().reset_index(name="total")
    profiles = profiles.merge(totals, on=["author", "temporal_period"])
    profiles["proportion"] = profiles["count"] / profiles["total"]

    poem_counts = df.groupby(["author", "temporal_period"])["original_id"].nunique().reset_index(name="n_poems")
    profiles = profiles.merge(poem_counts, on=["author", "temporal_period"])

    profiles.to_csv(OUTPUT_DIR / "author_period_profiles.csv", index=False)
    return profiles


def identify_prolific_authors(df: pd.DataFrame):
    """Find authors with enough poems across periods."""
    author_poems = df.groupby("author")["original_id"].nunique().reset_index(name="total_poems")
    author_poems = author_poems.sort_values("total_poems", ascending=False)

    period_poems = df.groupby(["author", "temporal_period"])["original_id"].nunique().unstack(fill_value=0)

    prolific = author_poems[author_poems["total_poems"] >= MIN_POEMS]
    cross_period = []
    for _, row in prolific.iterrows():
        author = row["author"]
        if author in period_poems.index:
            periods_active = (period_poems.loc[author] >= MIN_POEMS_PER_PERIOD).sum()
            if periods_active >= 2:
                cross_period.append(author)

    prolific.to_csv(OUTPUT_DIR / "prolific_authors.csv", index=False)
    print(f"Total authors: {len(author_poems)}")
    print(f"Prolific (>={MIN_POEMS} poems): {len(prolific)}")
    print(f"Cross-period (>={MIN_POEMS_PER_PERIOD} poems in >=2 periods): {len(cross_period)}")
    return prolific, cross_period


def trajectory_analysis(profiles: pd.DataFrame, cross_period_authors: list):
    """Analyze how each author's 1pl proportion changes across periods."""
    results = []
    for author in cross_period_authors:
        ap = profiles[(profiles["author"] == author) & (profiles["pronoun_class"] == "1pl")]
        if len(ap) < 2:
            ap_all = profiles[profiles["author"] == author].groupby("temporal_period").agg(
                total=("total", "first")
            ).reset_index()
            for _, r in ap_all.iterrows():
                results.append({
                    "author": author,
                    "temporal_period": r["temporal_period"],
                    "frac_1pl": 0,
                    "n_poems": 0,
                })
            continue

        for _, r in ap.iterrows():
            results.append({
                "author": author,
                "temporal_period": r["temporal_period"],
                "frac_1pl": r["proportion"],
                "n_poems": r["n_poems"],
            })

    traj = pd.DataFrame(results)
    if len(traj) == 0:
        return traj

    pivot = traj.pivot_table(index="author", columns="temporal_period",
                              values="frac_1pl", fill_value=0)
    cols = [c for c in KNOWN_PERIODS if c in pivot.columns]
    pivot = pivot.reindex(columns=cols)

    if "2014_2021" in pivot.columns and "post_2022" in pivot.columns:
        pivot["delta_2022"] = pivot["post_2022"] - pivot["2014_2021"]
        pivot = pivot.sort_values("delta_2022", ascending=False)

        pioneers = pivot[pivot["delta_2022"] > 0.05].index.tolist()
        resisters = pivot[pivot["delta_2022"] < -0.05].index.tolist()
        print(f"\nPioneers (1pl increase > 5pp after 2022): {len(pioneers)}")
        print(f"Resisters (1pl decrease > 5pp after 2022): {len(resisters)}")
    else:
        pioneers, resisters = [], []

    pivot.to_csv(OUTPUT_DIR / "author_1pl_trajectory.csv")
    traj.to_csv(OUTPUT_DIR / "author_trajectory_long.csv", index=False)
    return traj


def author_clustering(profiles: pd.DataFrame, cross_period_authors: list):
    """Cluster authors by pronoun usage patterns."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    feature_rows = []
    for author in cross_period_authors:
        ap = profiles[profiles["author"] == author]
        row = {"author": author}
        for period in KNOWN_PERIODS:
            for pc in ["1sg", "1pl", "2", "3sg", "3pl"]:
                val = ap[(ap["temporal_period"] == period) & (ap["pronoun_class"] == pc)]
                row[f"{pc}_{period}"] = val["proportion"].values[0] if len(val) > 0 else 0
        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows).set_index("author")
    feat_df = feat_df.fillna(0)

    if len(feat_df) < 4:
        print("Too few cross-period authors for clustering")
        feat_df.to_csv(OUTPUT_DIR / "author_features.csv")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)

    inertias = []
    K_range = range(2, min(8, len(feat_df)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    n_clusters = 3 if len(feat_df) >= 6 else 2
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat_df["cluster"] = km.fit_predict(X)
    feat_df.to_csv(OUTPUT_DIR / "author_clusters.csv")

    cluster_summary = feat_df.groupby("cluster").mean()
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_mean_profiles.csv")
    print(f"\nAuthor clusters ({n_clusters}):")
    print(cluster_summary.round(3).to_string())


def visualize_trajectories(traj: pd.DataFrame, top_n=12):
    """Small multiples of top author trajectories."""
    if len(traj) == 0:
        return

    pivot = traj.pivot_table(index="author", columns="temporal_period", values="frac_1pl", fill_value=0)
    cols = [c for c in KNOWN_PERIODS if c in pivot.columns]
    pivot = pivot.reindex(columns=cols)

    if "post_2022" in pivot.columns and "2014_2021" in pivot.columns:
        pivot["delta"] = abs(pivot["post_2022"] - pivot["2014_2021"])
        top_authors = pivot.nlargest(top_n, "delta").index
    else:
        top_authors = pivot.head(top_n).index

    n_cols = 4
    n_rows = (len(top_authors) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    for i, author in enumerate(top_authors):
        ax = axes_flat[i]
        d = traj[traj["author"] == author]
        d = d.set_index("temporal_period").reindex(KNOWN_PERIODS)
        ax.plot(range(len(d)), d["frac_1pl"].fillna(0) * 100, "o-", color="#F44336", linewidth=2)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([p[:8] for p in d.index], fontsize=7, rotation=45)
        ax.set_title(author[:20], fontsize=9)
        ax.grid(True, alpha=0.3)
        if i % n_cols == 0:
            ax.set_ylabel("1pl %")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Author-Level 1pl Trajectory (Top Movers)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "author_trajectories_small_multiples.png", dpi=150)
    plt.close(fig)
    print(f"Author trajectory visualization saved ({len(top_authors)} authors)")


def all_pronoun_author_heatmap(profiles: pd.DataFrame, cross_period_authors: list):
    """Heatmap showing all pronoun types for cross-period authors."""
    import seaborn as sns

    data = profiles[
        (profiles["author"].isin(cross_period_authors[:20])) &
        (profiles["temporal_period"] == "post_2022")
    ]
    pivot = data.pivot_table(index="author", columns="pronoun_class", values="proportion", fill_value=0)
    cols_order = [c for c in ["1sg", "1pl", "2", "3sg", "3pl"] if c in pivot.columns]
    pivot = pivot.reindex(columns=cols_order)

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title("Pronoun Distribution (%) for Top Authors (post-2022)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "author_pronoun_heatmap_post2022.png", dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    profiles = author_period_profiles(df)
    prolific, cross_period = identify_prolific_authors(df)
    traj = trajectory_analysis(profiles, cross_period)
    visualize_trajectories(traj)

    try:
        author_clustering(profiles, cross_period)
    except ImportError:
        print("sklearn not available, skipping clustering")
    except Exception as e:
        print(f"Clustering failed: {e}")

    try:
        all_pronoun_author_heatmap(profiles, cross_period)
    except Exception as e:
        print(f"Heatmap failed: {e}")

    print(f"\nAll author trajectory outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
