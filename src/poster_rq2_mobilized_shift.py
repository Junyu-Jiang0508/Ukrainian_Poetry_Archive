"""
Poster figure: per-author 1pl period-shift by wartime location.

  * COLOUR = remained in Ukraine vs left / exile
  * SHAPE  = mobilized (triangle) vs not mobilized (circle)

Usage : python src/poster_rq2_mobilized_shift.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq2_mobilized_shift.{pdf,png,svg}
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
    LEFT,
    REMAINED,
    location,
    location_color,
    location_label,
    legend_handles,
    mobilization_marker,
)

POST = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_author_random_slope_summaries.csv"
COV  = ROOT / "data" / "author_covariates_paper_roster_n33.csv"
OUT  = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"
MEAN = "author_total_period_shift_mean_log_mu"
DARK, RULE = "#2C2000", "#9F947E"


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
    d["wartime_loc"] = [
        location(m, i, r) for m, i, r in zip(d["mobilized"], d["in_ukraine_wartime"],
                                               d["region_at_archive_freeze"])
    ]
    return d


def main():
    _style()
    d = load()
    groups = [("remained", location_label("remained"), REMAINED),
              ("left", location_label("left"), LEFT)]
    fig, ax = plt.subplots(figsize=(6.8, 7.2))
    ax.axhline(0.0, color="#444444", lw=1.1, ls="--", zorder=1)

    rng = np.random.default_rng(7)
    xpos = {"remained": 0, "left": 1}
    for key, _, col in groups:
        g = d[d["wartime_loc"] == key]
        if g.empty:
            continue
        x0 = xpos[key]
        ys = g[MEAN].to_numpy(float)
        if len(ys) >= 3:
            ax.boxplot(ys, positions=[x0], widths=0.5, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor="none", edgecolor=col, lw=1.6),
                       medianprops=dict(color=col, lw=2.2),
                       whiskerprops=dict(color=col, lw=1.2),
                       capprops=dict(color=col, lw=1.2), zorder=2)
        jit = rng.uniform(-0.13, 0.13, len(g))
        for (_, row), x, y in zip(g.iterrows(), x0 + jit, ys):
            ax.scatter([x], [y],
                       color=location_color(row["mobilized"], row["in_ukraine_wartime"],
                                            row.get("region_at_archive_freeze")),
                       marker=mobilization_marker(row["mobilized"]),
                       s=78 if mobilization_marker(row["mobilized"]) == "^" else 64,
                       edgecolor="white", linewidth=0.8, zorder=4)
        med = float(np.median(ys))
        ax.annotate(f"median RR {np.exp(med):.2f}", (x0, med), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, fontstyle="italic", color=col, fontweight="semibold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{lab}\n(n={(d['wartime_loc'] == key).sum()})" for key, lab, _ in groups],
                       fontsize=10.5, color=DARK)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(r"Per-author 1pl period-shift posterior  (log rate, post $-$ pre)",
                  fontsize=11, color=DARK)
    ax.tick_params(axis="y", labelsize=10, colors=DARK)

    sec = ax.secondary_yaxis("right", functions=(lambda x: np.exp(x),
                                                 lambda x: np.log(np.clip(x, 1e-6, None))))
    sec.set_ylabel("rate ratio (post / pre)", fontsize=10.5, color=DARK)
    sec.set_yticks([0.5, 1, 2, 4]); sec.tick_params(labelsize=10, colors=DARK)

    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(RULE)

    ax.legend(handles=legend_handles(include_mobilization=True),
              loc="upper right", fontsize=8.5, frameon=False,
              title="colour = location; shape = mobilization", title_fontsize=9)

    fig.text(0.5, -0.01, "N = 33 roster authors, pooled stratum.  Dashed line = no shift; "
             "box = IQR with median.  Mobilized and civilian poets who remained in Ukraine "
             "are grouped together.",
             ha="center", fontsize=9, fontstyle="italic", color="#6B6257")
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_rq2_mobilized_shift.{ext}", bbox_inches="tight", transparent=True, format=ext)
        print(f"saved → poster_rq2_mobilized_shift.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
