"""
Poster figure: focused RQ2 1pl caterpillar — top-5 and bottom-5 authors only.

Encoding:
  * COLOUR = remained in Ukraine vs left / exile
  * SHAPE  = mobilized (triangle) vs not mobilized (circle)

Usage : python src/poster_rq2_caterpillar.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq2_caterpillar.{pdf,png,svg}
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.wartime_location_encoding import (
    is_mobilized,
    location_color,
    legend_handles,
    mobilization_marker,
)

ZERO, POPC, DARK, RULE = "#444444", "#7A7A7A", "#2C2000", "#9F947E"
POST = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_author_random_slope_summaries.csv"
COV  = ROOT / "data" / "author_covariates_paper_roster_n33.csv"
OUT  = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"
MEAN = "author_total_period_shift_mean_log_mu"
LO95 = "author_total_period_shift_hdi95_low"
HI95 = "author_total_period_shift_hdi95_high"
RR   = "author_total_period_shift_rate_ratio_mean"
Z95, Z50 = 1.959964, 0.674490


def _style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "savefig.facecolor": "none", "figure.facecolor": "none",
                         "axes.facecolor": "none", "savefig.dpi": 300})


def load():
    post = pd.read_csv(POST)
    d = post[(post.language_stratum == "pooled_Ukrainian_Russian") & (post.cell == "1pl")].copy()
    cov = pd.read_csv(COV, dtype=str, keep_default_na=False)
    d = d.merge(cov[["author", "mobilized", "in_ukraine_wartime", "region_at_archive_freeze"]],
                on="author", how="left")
    sd = (d[HI95] - d[LO95]) / (2 * Z95)
    d["q25"] = d[MEAN] - Z50 * sd
    d["q75"] = d[MEAN] + Z50 * sd
    return d.sort_values(MEAN).reset_index(drop=True)


def _row(ax, yy, r):
    c = location_color(r["mobilized"], r["in_ukraine_wartime"], r.get("region_at_archive_freeze"))
    mk = mobilization_marker(r["mobilized"])
    ax.hlines(yy, r["q25"], r["q75"], color=c, lw=5.0, alpha=0.85, zorder=3,
              capstyle="round")
    ax.plot(r[MEAN], yy, mk, color=c, ms=10 if mk == "^" else 8.5,
            mec="white", mew=1.2, zorder=5)


def main():
    _style()
    d = load()
    pop_mean = float(d[MEAN].mean())
    bot = d.head(5).reset_index(drop=True)
    top = d.tail(5).reset_index(drop=True)
    n_mid = len(d) - 10

    y_bot = list(range(5))
    y_gap = 5
    y_top = list(range(6, 11))
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    ax.axvline(0.0, color=ZERO, lw=1.6, zorder=2)
    ax.axvline(pop_mean, color=POPC, ls="--", lw=1.0, zorder=1)
    ax.text(pop_mean, 10.7, f" pop. mean {pop_mean:+.2f}", color="#5A5A5A",
            fontsize=8.5, fontstyle="italic", va="center", ha="left")

    for yy, (_, r) in zip(y_bot, bot.iterrows()): _row(ax, yy, r)
    for yy, (_, r) in zip(y_top, top.iterrows()): _row(ax, yy, r)

    ax.text(0.0, y_gap, f"···  {n_mid} middle authors omitted  ···", ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#8A8275", zorder=4)

    labels = ([f"{r['author']}   (RR {r[RR]:.2f})" for _, r in bot.iterrows()] + [""]
              + [f"{r['author']}   (RR {r[RR]:.2f})" for _, r in top.iterrows()])
    colors = [location_color(r["mobilized"], r["in_ukraine_wartime"], r.get("region_at_archive_freeze"))
              for _, r in bot.iterrows()] + ["none"] \
             + [location_color(r["mobilized"], r["in_ukraine_wartime"], r.get("region_at_archive_freeze"))
                for _, r in top.iterrows()]
    mob_flags = [is_mobilized(r["mobilized"]) for _, r in bot.iterrows()] + [False] \
                + [is_mobilized(r["mobilized"]) for _, r in top.iterrows()]
    ax.set_yticks(y_bot + [y_gap] + y_top)
    ax.set_yticklabels(labels, fontsize=10.5)
    for lab, col, mob in zip(ax.get_yticklabels(), colors, mob_flags):
        lab.set_color(col)
        if mob:
            lab.set_fontweight("semibold")
    ax.tick_params(left=False)

    lo = min(d["q25"].min(), 0) - 0.2
    hi = max(d["q75"].max(), 0) + 0.2
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, 11.0)
    ax.set_xlabel(r"Per-author 1pl period-shift posterior  (log rate, post $-$ pre)",
                  fontsize=11.5, color=DARK)
    ax.tick_params(axis="x", labelsize=10, colors=DARK)

    sec = ax.secondary_xaxis("top", functions=(lambda x: np.exp(x),
                                               lambda x: np.log(np.clip(x, 1e-6, None))))
    sec.set_xlabel("rate ratio (post / pre)", fontsize=10.5, color=DARK)
    sec.set_xticks([0.25, 0.5, 1, 2, 4]); sec.tick_params(labelsize=9.5, colors=DARK)

    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)

    ax.legend(handles=legend_handles(include_mobilization=True),
              loc="lower right", fontsize=9, frameon=False,
              title="colour = location; shape = mobilization", title_fontsize=9)

    fig.text(0.5, -0.02, "Pooled stratum, 33 authors (top-5 & bottom-5 shown).  "
             "Thick band = inner 50% credible interval; solid vertical = no shift, dashed = population mean.",
             ha="center", fontsize=8.5, fontstyle="italic", color="#6B6257")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_rq2_caterpillar.{ext}", bbox_inches="tight",
                    transparent=True, format=ext)
        print(f"saved → poster_rq2_caterpillar.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
