"""Publication figures and LaTeX tables (RQ1–RQ3)."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = Path("outputs/19_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESERIES_CSV = Path("outputs/08_change_point_detection/pronoun_timeseries_adaptive.csv")
INTERVALS_CSV = Path("outputs/08_change_point_detection/adaptive_intervals.csv")
TABLE1_CSV = Path("outputs/09_breakpoint_regression/table1_descriptive_by_period.csv")
TABLE2_CSV = Path("outputs/09_breakpoint_regression/table2_breakpoint_regression.csv")
TABLE3_CSV = Path("outputs/09_breakpoint_regression/table3_changepoints_detected.csv")
PREPARED_CSV = Path("outputs/09_breakpoint_regression/prepared_timeseries.csv")

WE_TYPE_PCT = Path("outputs/13_rq1_we_type/we_type_by_period_pct.csv")
WE_TYPE_TS = Path("outputs/13_rq1_we_type/we_type_timeseries_by_interval.csv")
ADDR_PCT = Path("outputs/14_rq2_addressee/addressee_type_by_period_pct.csv")
NPMI_SUMMARY = Path("outputs/16_temporal_cooccurrence/concept_group_mean_npmi.csv")
NPMI_PIVOT = Path("outputs/16_temporal_cooccurrence/npmi_pivot.csv")

PRONOUN_COLS = ["1sg", "1pl", "2", "3sg", "3pl"]
PRONOUN_LABELS = {
    "1sg": "1st Singular (I)",
    "1pl": "1st Plural (We)",
    "2": "2nd Person (You)",
    "3sg": "3rd Singular (He/She)",
    "3pl": "3rd Plural (They)",
}
PRONOUN_COLORS = {
    "1sg": "#2196F3",
    "1pl": "#F44336",
    "2": "#4CAF50",
    "3sg": "#FF9800",
    "3pl": "#9C27B0",
}


def figure1_timeseries():
    """Multi-panel time series with breakpoint lines."""
    ts = pd.read_csv(TIMESERIES_CSV)
    intervals = pd.read_csv(INTERVALS_CSV)
    intervals["mid_date"] = pd.to_datetime(intervals["start_date"]) + \
        (pd.to_datetime(intervals["end_date"]) - pd.to_datetime(intervals["start_date"])) / 2
    ts = ts.merge(intervals[["interval_id", "mid_date"]], on="interval_id")

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    for ax, col in zip(axes, PRONOUN_COLS):
        color = PRONOUN_COLORS[col]
        ax.plot(ts["mid_date"], ts[col] * 100, "o-", color=color,
                markersize=4, linewidth=1.5, label=PRONOUN_LABELS[col])
        ax.fill_between(ts["mid_date"], 0, ts[col] * 100, alpha=0.1, color=color)

        ax.axvline(pd.Timestamp("2014-02-01"), color="orange", ls="--", lw=1.5, alpha=0.8)
        ax.axvline(pd.Timestamp("2022-02-24"), color="red", ls="--", lw=1.5, alpha=0.8)

        ax.set_ylabel("Proportion (%)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        sizes = ts["n_poems"]
        for _, row in ts.iterrows():
            ax.scatter(row["mid_date"], row[col] * 100, s=row["n_poems"] / 2,
                       color=color, alpha=0.3, zorder=1)

    axes[0].annotate("Euromaidan", xy=(pd.Timestamp("2014-02-01"), axes[0].get_ylim()[1]),
                     fontsize=8, ha="right", color="orange")
    axes[0].annotate("Full-scale invasion", xy=(pd.Timestamp("2022-02-24"), axes[0].get_ylim()[1]),
                     fontsize=8, ha="right", color="red")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")

    fig.suptitle("Pronoun Proportion Time Series (35 Adaptive Intervals)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "figure1_timeseries.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "figure1_timeseries.pdf")
    plt.close(fig)
    print("Figure 1 saved")


def figure2_changepoint_1pl():
    """Focused 1pl panel with PELT change point highlighted."""
    ts = pd.read_csv(TIMESERIES_CSV)
    intervals = pd.read_csv(INTERVALS_CSV)
    intervals["mid_date"] = pd.to_datetime(intervals["start_date"]) + \
        (pd.to_datetime(intervals["end_date"]) - pd.to_datetime(intervals["start_date"])) / 2
    ts = ts.merge(intervals[["interval_id", "mid_date", "interval_label"]], on="interval_id")

    changepoints = pd.read_csv(TABLE3_CSV)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts["mid_date"], ts["1pl"] * 100, "o-", color="#F44336",
            markersize=5, linewidth=2, label="1st Plural (We)")
    ax.fill_between(ts["mid_date"], 0, ts["1pl"] * 100, alpha=0.15, color="#F44336")

    ax.axvline(pd.Timestamp("2014-02-01"), color="orange", ls="--", lw=1.5, alpha=0.8,
               label="Euromaidan (2014-02)")
    ax.axvline(pd.Timestamp("2022-02-24"), color="darkred", ls="--", lw=1.5, alpha=0.8,
               label="Full-scale invasion (2022-02)")

    for _, cp in changepoints.iterrows():
        if cp["pronoun"] == "1pl":
            cp_date = pd.Timestamp(cp["date"])
            ax.axvline(cp_date, color="green", ls="-.", lw=2, alpha=0.8,
                       label=f"PELT changepoint ({cp_date.strftime('%Y-%m')})")
            ax.annotate(f"Changepoint\n{cp_date.strftime('%Y-%m')}",
                        xy=(cp_date, ax.get_ylim()[1] * 0.9),
                        fontsize=9, color="green", ha="center")

    ax.set_ylabel("1pl Proportion (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title('First-Person Plural "We" (1pl) with PELT Change Points', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure2_1pl_changepoint.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "figure2_1pl_changepoint.pdf")
    plt.close(fig)
    print("Figure 2 saved")


def tables_latex():
    """Format Table 1 and Table 2 as LaTeX."""
    t1 = pd.read_csv(TABLE1_CSV)
    t2 = pd.read_csv(TABLE2_CSV)

    def fmt_p(p):
        if p < 0.001:
            return "< 0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        elif p < 0.05:
            return f"{p:.3f}*"
        elif p < 0.1:
            return f"{p:.3f}+"
        else:
            return f"{p:.3f}"

    with open(OUTPUT_DIR / "table1_descriptive.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Descriptive Statistics by Period}\n")
        f.write("\\begin{tabular}{llrrr}\n\\hline\n")
        f.write("Pronoun & Period & Mean & SD & N \\\\\n\\hline\n")
        for _, row in t1.iterrows():
            sd = f"{row['std']:.3f}" if pd.notna(row['std']) else "---"
            f.write(f"{row['pronoun']} & {row['period']} & {row['mean']:.3f} & {sd} & {int(row['n'])} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    with open(OUTPUT_DIR / "table2_regression.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Weighted Breakpoint Regression Results}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\hline\n")
        f.write("Pronoun & $\\beta_{2014}$ & $p_{2014}$ & $\\beta_{2022}$ & $p_{2022}$ & $R^2$ \\\\\n\\hline\n")
        for _, row in t2.iterrows():
            f.write(f"{row['pronoun']} & {row['beta_2014_level']:.3f} & {fmt_p(row['p_2014_level'])} "
                    f"& {row['beta_2022_level']:.3f} & {fmt_p(row['p_2022_level'])} "
                    f"& {row['R2']:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\n")
        f.write("\\footnotesize{* $p < 0.05$, + $p < 0.10$}\n")
        f.write("\\end{table}\n")

    print("LaTeX tables saved")


def figure3_we_type():
    """RQ1 we_type evolution figure."""
    try:
        we_ts = pd.read_csv(WE_TYPE_TS)
    except FileNotFoundError:
        print("we_type timeseries not found, skipping Figure 3")
        return

    intervals = pd.read_csv(INTERVALS_CSV)
    intervals["mid_date"] = pd.to_datetime(intervals["start_date"]) + \
        (pd.to_datetime(intervals["end_date"]) - pd.to_datetime(intervals["start_date"])) / 2
    we_ts = we_ts.merge(intervals[["interval_id", "mid_date"]], on="interval_id")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(we_ts["mid_date"], we_ts["frac_1pl"] * 100, "o-", color="#F44336",
                 markersize=4, linewidth=2)
    axes[0].axvline(pd.Timestamp("2014-02-01"), color="orange", ls="--", lw=1.5)
    axes[0].axvline(pd.Timestamp("2022-02-24"), color="red", ls="--", lw=1.5)
    axes[0].set_ylabel("1pl as % of all pronouns")
    axes[0].set_title("(A) First-Person Plural Proportion Over Time")
    axes[0].grid(True, alpha=0.3)

    type_colors = {"exclusive_ingroup": "#D32F2F", "inclusive_addressee": "#1976D2",
                   "mixed_we": "#388E3C", "generic_universal": "#F57C00",
                   "speaker_exclusive": "#7B1FA2"}
    for wt, color in type_colors.items():
        col = f"frac_{wt}"
        if col in we_ts.columns:
            axes[1].plot(we_ts["mid_date"], we_ts[col] * 100, "o-", color=color,
                         markersize=3, linewidth=1.5, label=wt.replace("_", " ").title())
    axes[1].axvline(pd.Timestamp("2014-02-01"), color="orange", ls="--", lw=1.5)
    axes[1].axvline(pd.Timestamp("2022-02-24"), color="red", ls="--", lw=1.5)
    axes[1].set_ylabel("% of 1pl tokens")
    axes[1].set_xlabel("Date")
    axes[1].set_title('(B) "We"-Type Sub-Categories Over Time')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure3_we_type.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "figure3_we_type.pdf")
    plt.close(fig)
    print("Figure 3 saved")


def figure4_addressee():
    """RQ2 addressee_type comparison."""
    try:
        addr = pd.read_csv(ADDR_PCT, index_col=0)
    except FileNotFoundError:
        print("addressee data not found, skipping Figure 4")
        return

    addr = addr.drop("not_applicable", errors="ignore")
    ordered_cols = [c for c in ["pre_2014", "2014_2021", "post_2022"] if c in addr.columns]
    addr = addr.reindex(columns=ordered_cols)

    fig, ax = plt.subplots(figsize=(12, 6))
    addr.T.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Second-Person Addressee Types by Period (Excluding Not Applicable)")
    ax.set_ylabel("% of 2nd-person tokens")
    ax.set_xlabel("Period")
    ax.legend(title="Addressee Type", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure4_addressee.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure4_addressee.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 saved")


def figure5_npmi():
    """RQ3 NPMI concept group summary heatmap."""
    try:
        npmi = pd.read_csv(NPMI_SUMMARY)
    except FileNotFoundError:
        print("NPMI summary not found, skipping Figure 5")
        return

    import seaborn as sns

    periods = ["pre_2014", "2014_2021", "post_2022"]
    core_pronouns = ["я", "ми", "ти", "він", "вона", "вони"]

    npmi_2periods = npmi[npmi["period"].isin(["2014_2021", "post_2022"])]
    pivot = npmi_2periods.pivot_table(
        index=["pronoun", "concept_group"], columns="period", values="mean_npmi"
    )
    if "2014_2021" in pivot.columns and "post_2022" in pivot.columns:
        pivot["delta"] = pivot["post_2022"] - pivot["2014_2021"]
    pivot = pivot.reindex(columns=[p for p in periods if p in pivot.columns] + ["delta"])

    delta_wide = pivot["delta"].unstack("concept_group") if "delta" in pivot.columns else None

    if delta_wide is not None:
        delta_wide = delta_wide.reindex(index=[p for p in core_pronouns if p in delta_wide.index])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(delta_wide, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Change in Mean NPMI (post_2022 - 2014_2021)\nPronoun x Concept Group")
        ax.set_ylabel("Pronoun")
        ax.set_xlabel("Concept Group")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure5_npmi_delta.png", dpi=200)
        fig.savefig(OUTPUT_DIR / "figure5_npmi_delta.pdf")
        plt.close(fig)
    print("Figure 5 saved")


def main():
    figure1_timeseries()
    figure2_changepoint_1pl()
    tables_latex()
    figure3_we_type()
    figure4_addressee()
    figure5_npmi()
    print(f"\nAll publication figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
