"""
Poster figure: author trajectories in the (1sg, 1pl) plane.

Each roster author's pre-2022 mean position is drawn with an arrow to their
wartime (P2, 2022+) position.
  * COLOUR = remained in Ukraine vs left / exile
  * SHAPE  = mobilized (triangle) vs not mobilized (circle) on the pre-war dot

Usage : python src/poster_rq2_trajectories.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq2_trajectories.{pdf,png,svg}
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.wartime_location_encoding import (
    location_color,
    legend_handles,
    mobilization_marker,
)

DARK, RULE = "#2C2000", "#9F947E"
POEM = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_poem_cell_counts_12.csv"
COV  = ROOT / "data" / "author_covariates_paper_roster_n33.csv"
OUT  = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"
QUARTET = ["1sg", "1pl", "2sg", "2pl"]
ANCHORS = [
    "Ihor Mitrov", "Yaryna Chornohuz", "Elizaveta Zharikova", "Artur Dron", "Serhiy Zhadan",
    "Eva Tur", "Borys Humeniuk", "Dmytro Lazutkin",
    "Mykhailo Zharzhailo", "Iya Kiva", "Boris Khersonsky", "Hryhoryi Falkovych", "Kateryna Babkina",
]
ANCHOR_NUM = {a: i + 1 for i, a in enumerate(ANCHORS)}


def _style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "savefig.facecolor": "none", "figure.facecolor": "none",
                         "axes.facecolor": "none", "savefig.dpi": 300})


def load():
    d = pd.read_csv(POEM)
    d = d[d.period3.isin(["P1_2014_2021", "P2_2022_plus"])].copy()
    q = d[QUARTET].sum(axis=1)
    d = d[q > 0].copy()
    d["s1sg"] = d["1sg"] / d[QUARTET].sum(axis=1)
    d["s1pl"] = d["1pl"] / d[QUARTET].sum(axis=1)
    d["period"] = np.where(d.period3 == "P1_2014_2021", "pre", "post")
    g = (d.groupby(["author", "period"])[["s1sg", "s1pl"]].mean().reset_index())
    piv = g.pivot(index="author", columns="period", values=["s1sg", "s1pl"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.dropna().reset_index()
    cov = pd.read_csv(COV, dtype=str, keep_default_na=False)
    return piv.merge(cov[["author", "mobilized", "in_ukraine_wartime", "region_at_archive_freeze"]],
                     on="author", how="left")


def main():
    _style()
    d = load()
    fig, ax = plt.subplots(figsize=(9.6, 8.2))

    for t in (0.2, 0.4, 0.6, 0.8):
        ax.plot([0, t], [t, 0], color="#C9BFA8", lw=0.6, ls=(0, (4, 4)), zorder=0)

    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    for _, r in d.iterrows():
        c = location_color(r["mobilized"], r["in_ukraine_wartime"], r.get("region_at_archive_freeze"))
        mk = mobilization_marker(r["mobilized"])
        x0, y0, x1, y1 = r["s1sg_pre"], r["s1pl_pre"], r["s1sg_post"], r["s1pl_post"]
        anchor = r["author"] in ANCHOR_NUM
        lw, al, dotsz = (2.0, 0.95, 56) if anchor else (1.0, 0.28, 32)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, alpha=al,
                                    shrinkA=3, shrinkB=2), zorder=(4 if anchor else 2))
        za = 6 if anchor else 3
        ax.scatter([x0], [y0], color=c, marker=mk, s=dotsz, zorder=za, edgecolor="white",
                   linewidth=0.8, alpha=al if not anchor else 1.0)
        if anchor:
            ax.annotate(str(ANCHOR_NUM[r["author"]]), (x0, y0), fontsize=9.5, color=c,
                        fontweight="bold", xytext=(5, 5), textcoords="offset points",
                        zorder=7, path_effects=halo)

    ax.set_xlabel("share of 1sg «I» within the quartet", fontsize=11, color=DARK)
    ax.set_ylabel("share of 1pl «we» within the quartet", fontsize=11, color=DARK)
    ax.tick_params(labelsize=10, colors=DARK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(RULE)
    ax.set_xlim(-0.02, 0.86); ax.set_ylim(-0.03, 1.0)

    d_idx = d.set_index("author")
    ax.text(0.60, 0.985, "Key  (colour = location; shape = mobilization)",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=DARK, va="top")
    for k, name in enumerate(ANCHORS):
        row = d_idx.loc[name]
        c = location_color(row["mobilized"], row["in_ukraine_wartime"], row.get("region_at_archive_freeze"))
        ax.text(0.60, 0.95 - 0.039 * k, f"{k+1}.  {name}", transform=ax.transAxes,
                fontsize=8.6, color=c, va="top",
                fontweight=("semibold" if mobilization_marker(row["mobilized"]) == "^" else "normal"))

    ax.legend(handles=legend_handles(include_mobilization=True, include_arrow=True),
              loc="lower right", fontsize=8.5, frameon=False,
              title="colour = location; shape = mobilization", title_fontsize=9)

    fig.text(0.5, -0.01, f"N = {len(d)} roster authors with poems in both periods.  "
             "Numbered = narrative anchors; pale arrows = context cloud.  "
             "Dashed diagonals mark constant total 1st-person share.",
             ha="center", fontsize=9, fontstyle="italic", color="#6B6257")
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_rq2_trajectories.{ext}", bbox_inches="tight", transparent=True, format=ext)
        print(f"saved → poster_rq2_trajectories.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
