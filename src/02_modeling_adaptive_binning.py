"""Adaptive time bins for pronoun time series (min poems per bin)."""

from pathlib import Path

from utils.workspace import prepare_analysis_environment

prepare_analysis_environment(__file__, matplotlib_backend=None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.adaptive_temporal_binning import (              
    INITIAL_MONTHS,
    MIN_POEMS_PER_INTERVAL,
    TARGET_POEMS,
    adaptive_binning,
)

INPUT_CSV = Path("outputs/01_annotation_pronoun_detection/ukrainian_pronouns_projection_final.csv")
OUTPUT_DIR = Path("outputs/02_modeling_adaptive_binning")


def smart_merge_intervals(
    df: pd.DataFrame,
    min_poems: int = None,
    target_poems: int = None,
    initial_months: int = None,
) -> pd.DataFrame:
    min_poems = min_poems or MIN_POEMS_PER_INTERVAL
    target_poems = target_poems or TARGET_POEMS
    initial_months = initial_months or INITIAL_MONTHS

    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_period"] = df["_date"].dt.to_period(f"{initial_months}M")

    intervals = (
        df.groupby("_period")
        .agg(
            n_poems=("ID", "nunique"),
            n_total=("ID", "count"),
            start=("_date", "min"),
            end=("_date", "max"),
        )
        .reset_index()
        .sort_values("_period")
    )
    intervals["interval_id"] = range(1, len(intervals) + 1)

    changed = True
    while changed and len(intervals) > 1:
        changed = False
        small_idx = intervals["n_poems"].idxmin()
        if intervals.loc[small_idx, "n_poems"] < min_poems:
            changed = True
            if small_idx == intervals.index[0]:
                merge_with = intervals.index[1]
            elif small_idx == intervals.index[-1]:
                merge_with = intervals.index[-2]
            else:
                pos = list(intervals.index).index(small_idx)
                prev_idx = intervals.index[pos - 1]
                next_idx = intervals.index[pos + 1]
                cost_prev = abs(intervals.loc[prev_idx, "n_poems"] - target_poems) + abs(
                    intervals.loc[small_idx, "n_poems"] - target_poems
                )
                cost_next = abs(intervals.loc[next_idx, "n_poems"] - target_poems) + abs(
                    intervals.loc[small_idx, "n_poems"] - target_poems
                )
                merge_with = prev_idx if cost_prev <= cost_next else next_idx

            keep_idx = min(small_idx, merge_with)
            remove_idx = max(small_idx, merge_with)
            intervals.loc[keep_idx, "end"] = max(
                intervals.loc[keep_idx, "end"], intervals.loc[remove_idx, "end"]
            )
            intervals.loc[keep_idx, "start"] = min(
                intervals.loc[keep_idx, "start"], intervals.loc[remove_idx, "start"]
            )
            intervals.loc[keep_idx, "n_poems"] += intervals.loc[remove_idx, "n_poems"]
            intervals.loc[keep_idx, "n_total"] += intervals.loc[remove_idx, "n_total"]
            intervals = intervals.drop(remove_idx).reset_index(drop=True)
            intervals["interval_id"] = range(1, len(intervals) + 1)

    return intervals


def build_pronoun_timeseries_adaptive(
    df: pd.DataFrame,
    min_poems: int = None,
    initial_months: int = None,
    use_smart_merge: bool = False,
) -> pd.DataFrame:
    min_poems = min_poems or MIN_POEMS_PER_INTERVAL
    initial_months = initial_months or INITIAL_MONTHS

    if use_smart_merge:
        intervals = smart_merge_intervals(df, min_poems=min_poems, initial_months=initial_months)
        df = df.copy()
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["_date"])
        intervals = intervals.rename(columns={"start": "start_date", "end": "end_date"})

        def _assign(row):
            for _, r in intervals.iterrows():
                if r["start_date"] <= row["_date"] <= r["end_date"]:
                    return r["interval_id"]
            return np.nan

        df["interval_id"] = df.apply(_assign, axis=1)
        df = df.dropna(subset=["interval_id"])
        df_with_int = df
        interval_df = intervals
    else:
        df_with_int, interval_df = adaptive_binning(
            df, min_poems=min_poems, initial_months=initial_months
        )

    def _agg(g):
        n = len(g)
        if n == 0:
            return pd.Series({"1sg": 0, "1pl": 0, "2": 0, "3sg": 0, "3pl": 0, "n_tokens": 0})
        p1 = (g["person"] == "1") | (g["person"].astype(str) == "1")
        p2 = (g["person"] == "2") | (g["person"].astype(str) == "2")
        p3 = (g["person"] == "3") | (g["person"].astype(str) == "3")
        sing = (g["number"] == "Sing").fillna(False)
        plur = (g["number"] == "Plur").fillna(False)
        return pd.Series({
            "1sg": ((p1) & (sing)).sum() / n,
            "1pl": ((p1) & (plur)).sum() / n,
            "2": (p2).sum() / n,
            "3sg": ((p3) & (sing)).sum() / n,
            "3pl": ((p3) & (plur)).sum() / n,
            "n_tokens": n,
        })

    ts = (
        df_with_int.groupby("interval_id", group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    period_dt = interval_df["start_date"] if "start_date" in interval_df.columns else interval_df["start"]
    ts = ts.merge(
        interval_df[["interval_id"]].drop_duplicates().assign(period_dt=period_dt),
        on="interval_id",
    )
    if "n_poems" in interval_df.columns:
        ts = ts.merge(
            interval_df[["interval_id", "n_poems"]].drop_duplicates(),
            on="interval_id",
        )
    ts["period"] = ts["interval_id"].astype(str)
    return ts


def visualize_intervals(intervals: pd.DataFrame, out_path: Path = None):
    """Visualize adaptive intervals: poem count and duration per interval."""
    intervals = intervals.copy()
    if "start_date" not in intervals.columns and "start" in intervals.columns:
        intervals["start_date"] = intervals["start"]
        intervals["end_date"] = intervals["end"]

    intervals["mid_date"] = intervals["start_date"] + (intervals["end_date"] - intervals["start_date"]) / 2
    intervals["duration_months"] = (intervals["end_date"] - intervals["start_date"]).dt.days / 30

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    colors = ["#2ecc71" if n >= MIN_POEMS_PER_INTERVAL else "#e74c3c" for n in intervals["n_poems"]]
    ax1.bar(intervals["interval_id"], intervals["n_poems"], color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axhline(MIN_POEMS_PER_INTERVAL, color="gray", linestyle="--", alpha=0.5, label=f"min={MIN_POEMS_PER_INTERVAL}")
    ax1.set_ylabel("Poems per interval")
    ax1.set_xlabel("Interval ID")
    ax1.set_title(f"Adaptive intervals (min_poems={MIN_POEMS_PER_INTERVAL})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(intervals["interval_id"], intervals["duration_months"], color="steelblue", alpha=0.7)
    ax2.set_xlabel("Interval ID")
    ax2.set_ylabel("Duration (months)")
    ax2.set_title("Interval time span")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_path or OUTPUT_DIR / "adaptive_intervals.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    import sys
    df = pd.read_csv(INPUT_CSV, low_memory=False, on_bad_lines="skip", dtype={"person": "object", "number": "object"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["person"] = df["person"].astype(str).str.strip()
    df = df[df["person"].isin(["1", "2", "3"])]
    df["number"] = df["number"].astype(str).str.strip().replace("", np.nan)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    min_poems = int(args[0]) if args else MIN_POEMS_PER_INTERVAL
    use_smart = "--smart" in sys.argv

    print(f"Adaptive binning: min_poems={min_poems}, smart_merge={use_smart}")
    print(f"Total poems: {df['ID'].nunique()}, pronoun tokens: {len(df)}")

    df_with_int, intervals = adaptive_binning(df, min_poems=min_poems)
    print(f"\nIntervals: {len(intervals)}")
    print(intervals[["interval_id", "interval_label", "n_poems", "start_date", "end_date"]])

    ts = build_pronoun_timeseries_adaptive(df, min_poems=min_poems, use_smart_merge=use_smart)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts.to_csv(OUTPUT_DIR / "pronoun_timeseries_adaptive.csv", index=False)
    intervals.to_csv(OUTPUT_DIR / "adaptive_intervals.csv", index=False)
    visualize_intervals(intervals)
    print("\nSaved: pronoun_timeseries_adaptive.csv, adaptive_intervals.csv, ada...")


if __name__ == "__main__":
    main()
