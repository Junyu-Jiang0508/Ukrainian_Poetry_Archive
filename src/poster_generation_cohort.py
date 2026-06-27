"""
Poster figure: generation-cohort composition (pre vs post 2022).

Transparent background, poster-friendly palette.
Usage : python src/poster_generation_cohort.py
Output: docs/Ukranian_Poetr_Report/figures/poster_generation_cohort.{pdf,png,svg}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
POEM_CSV   = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_poem_cell_counts_12.csv"
ROSTER_CSV = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
COV_CSV    = ROOT / "data" / "author_covariates.csv"
OUT        = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"

PRE_PERIOD  = "P1_2014_2021"
POST_PERIOD = "P2_2022_plus"
QUARTET     = ("1sg", "1pl", "2sg", "2pl")

# ── poster palette ───────────────────────────────────────────────────────
NAV   = "#1C3557"
BLU   = "#005BBB"
GOLD  = "#FFD500"
DARK  = "#2C2000"
FONT  = "Times New Roman"

# Cell colors — poster-friendly, legible on both light and dark backgrounds
CELL_COLORS = {
    "1sg": "#5A7D9A",          # slate blue
    "1pl": "#C87A20",          # warm amber (the star cell)
    "2sg": "#E8C547",          # gold
    "2pl": "#2E6B5E",          # deep teal
}
CELL_LABELS = {
    "1sg": "1sg «I»",
    "1pl": "1pl «we»",
    "2sg": "2sg «thou»",
    "2pl": "2pl «you (pl.)»",
}

GEN_COHORT_ORDER = ("pre_1970", "1970s", "1980s", "1990s", "2000s_plus")
GEN_COHORT_LABEL = {
    "pre_1970": "pre-1970",
    "1970s": "1970s",
    "1980s": "1980s",
    "1990s": "1990s",
    "2000s_plus": "2000s+",
}


def _load_data():
    df = pd.read_csv(POEM_CSV)
    df = df[df["year_int"].notna()].copy()
    df["year"] = df["year_int"].astype(int)
    df = df[df[list(QUARTET)].sum(axis=1) > 0].copy()

    roster = pd.read_csv(ROSTER_CSV)
    roster_set = set(roster.loc[roster["included"] == True, "author"].astype(str))
    cov = pd.read_csv(COV_CSV)

    d = df[df["author"].isin(roster_set)].copy()
    d = d[d["period3"].isin([PRE_PERIOD, POST_PERIOD])]
    d = d.merge(cov[["author", "generation_cohort"]], on="author", how="left")

    rows = []
    for (coh, per), sub in d.groupby(["generation_cohort", "period3"]):
        denom = sub[list(QUARTET)].to_numpy().sum()
        if denom <= 0:
            continue
        sums = sub[list(QUARTET)].sum()
        rows.append({
            "generation_cohort": coh,
            "period": per,
            "n_authors": sub["author"].nunique(),
            "n_poems": len(sub),
            **{f"share_{c}": float(sums[c]) / denom for c in QUARTET},
        })
    comp = pd.DataFrame(rows)
    comp = comp[comp["generation_cohort"].isin(GEN_COHORT_ORDER)]
    return comp


def main():
    comp = _load_data()
    if comp.empty:
        print("No cohort data — aborting.")
        return

    cohorts = [c for c in GEN_COHORT_ORDER if c in comp["generation_cohort"].unique()]
    n_coh = len(cohorts)

    BAR_H = 0.34
    GAP_WITHIN = 0.10
    GAP_BETWEEN = 0.62

    # compute y positions
    y_positions: dict[tuple[str, str], float] = {}
    cohort_centers: dict[str, float] = {}
    y_cursor = 0.0
    for coh in cohorts:
        y_pre = y_cursor
        y_post = y_cursor - (BAR_H + GAP_WITHIN)
        y_positions[(coh, PRE_PERIOD)] = y_pre
        y_positions[(coh, POST_PERIOD)] = y_post
        cohort_centers[coh] = (y_pre + y_post) / 2
        y_cursor -= (BAR_H * 2 + GAP_WITHIN + GAP_BETWEEN)

    fig, ax = plt.subplots(figsize=(11.5, max(5.6, 0.55 * abs(y_cursor) + 1.4)))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    LEFT_LABEL_X = -0.21
    LEFT_N_X = -0.018

    for coh in cohorts:
        for period in (PRE_PERIOD, POST_PERIOD):
            row = comp[(comp["generation_cohort"] == coh) & (comp["period"] == period)]
            if row.empty:
                continue
            y = y_positions[(coh, period)]
            left = 0.0
            for cell in QUARTET:
                w = float(row[f"share_{cell}"].iloc[0])
                ax.barh(
                    y, w, height=BAR_H, left=left,
                    color=CELL_COLORS[cell], edgecolor="white", linewidth=0.7,
                )
                if w >= 0.04:
                    # text color: white on dark cells, dark on light cells
                    txt_c = "white" if cell in ("1sg", "1pl", "2pl") else DARK
                    ax.text(
                        left + w / 2, y, f"{w * 100:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontfamily=FONT, color=txt_c,
                    )
                left += w

            label_short = "pre-2022" if period == PRE_PERIOD else "post-2022"
            n_a = int(row["n_authors"].iloc[0])
            n_p = int(row["n_poems"].iloc[0])
            ax.text(
                LEFT_N_X, y,
                f"{label_short}  ·  {n_a} authors · {n_p} poems",
                ha="right", va="center", fontsize=9,
                fontfamily=FONT, color=NAV,
            )

        # cohort label
        ax.text(
            LEFT_LABEL_X, cohort_centers[coh],
            GEN_COHORT_LABEL[coh],
            ha="left", va="center", fontsize=13, fontweight="bold",
            fontfamily=FONT, color=NAV,
            transform=ax.get_yaxis_transform(),
        )

    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=10, fontfamily=FONT, color=NAV,
    )
    ax.set_xlabel(
        "Share of the four-cell first/second-person attention quartet",
        fontsize=10.5, fontfamily=FONT, color=NAV, labelpad=8,
    )
    ax.set_ylim(y_cursor + GAP_BETWEEN - 0.30, BAR_H + 0.30)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(NAV)
    ax.tick_params(axis="x", colors=NAV, length=3)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.85, bottom=0.16)

    fig.suptitle(
        "Generation cohorts: pronoun composition before vs after the invasion",
        fontsize=14, fontweight="bold", fontfamily=FONT,
        color=NAV, y=0.965, x=0.555,
    )

    handles = [
        mpatches.Patch(color=CELL_COLORS[c], label=CELL_LABELS[c])
        for c in QUARTET
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.555, 0.925), ncol=4, frameon=False,
        fontsize=10, prop={"family": FONT},
    )

    fig.text(
        0.555, 0.025,
        "Each pair compares pre-2022 (top) vs post-2022 (bottom) corpus composition. "
        "The 1990s cohort is the only one whose 1pl share more than doubles after the invasion.",
        ha="center", fontsize=8.5, fontfamily=FONT,
        color="#6F665B", fontstyle="italic",
    )

    # ── save ─────────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(
            OUT / f"poster_generation_cohort.{ext}",
            dpi=300, bbox_inches="tight",
            facecolor="none", edgecolor="none",
            transparent=True, format=ext,
        )
        print(f"saved → {OUT}/poster_generation_cohort.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
