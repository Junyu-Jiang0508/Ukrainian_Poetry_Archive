"""
Appendix figure: breakpoint regression + PELT change-point detection, redrawn
for legibility.

Replaces the original interval-indexed 5-panel scatter (x-axis was the
meaningless bin id, no fitted trend drawn). This version uses a calendar-time
x-axis shared across panels, draws the segmented weighted-least-squares fit as
explicit level-shift steps at the 2014 and 2022 knots, marks the PELT
change-points on the 1pl series, sizes each binned point by its poem count, and
annotates each cell with its 2022 level-shift coefficient and p-value.

Usage : python src/fig_appendix_breakpoint_timeseries.py
Output: docs/Ukranian_Poetr_Report/figures/fig1_breakpoint_timeseries.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parent.parent
DATA = ROOT / "outputs" / "02_modeling_breakpoint_regression" / "prepared_timeseries.csv"
OUT  = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"

NAV   = "#1C3557"
BLU   = "#005BBB"
FIT   = "#1C3557"
PT    = "#5B9BD5"
KNOT  = "#9A1750"
PELT  = "#C77A00"
GREY  = "#6F665B"

KNOT_2014 = pd.Timestamp("2014-02-01")
KNOT_2022 = pd.Timestamp("2022-02-24")
PELT_DATES = [pd.Timestamp("2022-03-01"), pd.Timestamp("2023-02-10")]

# 2022 level-shift coefficient and p-value from table2_breakpoint_regression.csv
CELLS = {
    "1pl": ("1pl  «ми»  (we)",       0.202, 0.051),
    "1sg": ("1sg  «я»  (I)",         0.126, 0.220),
    "2":   ("2  «ти / ви»  (you)",   0.020, 0.865),
    "3sg": ("3sg  (he/she/it)",     -0.357, 0.017),
    "3pl": ("3pl  (they)",           0.009, 0.903),
}


def _fit(d, col):
    """Weighted segmented OLS; returns fitted values aligned to rows of d."""
    X = np.column_stack([
        np.ones(len(d)), d.time, d.post_2014, d.post_2022,
        d.time_post2014, d.time_post2022,
    ])
    W = np.sqrt(d.n_poems.values.astype(float))
    beta, *_ = np.linalg.lstsq(X * W[:, None], d[col].values * W, rcond=None)
    return X @ beta


def _panel(ax, d, col, title, beta22, p22, show_pelt=False):
    mid = d["mid"]
    yhat = _fit(d, col)

    # binned observations, size ~ poem count
    sizes = 14 + 1.4 * d.n_poems.values
    ax.scatter(mid, d[col], s=sizes, color=PT, alpha=0.55,
               edgecolor="white", linewidth=0.4, zorder=3)

    # segmented fit drawn per regime so level steps are not bridged
    for mask in (d.post_2014 == 0, (d.post_2014 == 1) & (d.post_2022 == 0), d.post_2022 == 1):
        m = mask.values
        if m.sum() >= 1:
            ax.plot(mid[m], yhat[m], color=FIT, lw=2.4, zorder=4,
                    solid_capstyle="round")

    # knot lines
    for k in (KNOT_2014, KNOT_2022):
        ax.axvline(k, color=KNOT, lw=1.1, ls=(0, (4, 3)), alpha=0.8, zorder=2)
    if show_pelt:
        for pd_ in PELT_DATES:
            ax.axvline(pd_, color=PELT, lw=1.0, ls=":", alpha=0.9, zorder=2)

    # title + annotation
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAV, loc="left", pad=4)
    sig = "marginal" if 0.05 <= p22 < 0.10 else ("sig." if p22 < 0.05 else "n.s.")
    ax.text(0.985, 0.93, rf"$\hat{{\beta}}_{{2022}}={beta22:+.2f}$,  $p={p22:.3f}$  ({sig})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color=NAV, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                 ec=GREY, lw=0.5, alpha=0.85), zorder=6)

    ax.set_ylabel("share", fontsize=9, color=GREY)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=8.5, colors=GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.grid(axis="y", color=GREY, alpha=0.15, lw=0.5)
    ax.set_xlim(d["mid"].min() - pd.Timedelta(days=90),
                d["mid"].max() + pd.Timedelta(days=90))


def main():
    d = pd.read_csv(DATA)
    s = pd.to_datetime(d.start_date, format="%Y/%m/%d")
    e = pd.to_datetime(d.end_date, format="%Y/%m/%d")
    d["mid"] = s + (e - s) / 2

    fig = plt.figure(figsize=(11, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.16)

    ax_head = fig.add_subplot(gs[0, :])
    _panel(ax_head, d, "1pl", CELLS["1pl"][0], CELLS["1pl"][1], CELLS["1pl"][2],
           show_pelt=True)

    others = ["1sg", "2", "3sg", "3pl"]
    for i, col in enumerate(others):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        _panel(ax, d, col, CELLS[col][0], CELLS[col][1], CELLS[col][2])

    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PT,
               markeredgecolor="white", markersize=8, label="binned rate (size ∝ poems)"),
        Line2D([0], [0], color=FIT, lw=2.4, label="segmented WLS fit"),
        Line2D([0], [0], color=KNOT, lw=1.1, ls=(0, (4, 3)), label="a-priori knots (2014, 2022)"),
        Line2D([0], [0], color=PELT, lw=1.0, ls=":", label="PELT change-points (1pl)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Per-pronoun temporal trajectories with segmented fits and change-points",
                 fontsize=13.5, fontweight="bold", color=NAV, y=0.985)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_breakpoint_timeseries.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig1_breakpoint_timeseries.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("saved → fig1_breakpoint_timeseries.{pdf,png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
