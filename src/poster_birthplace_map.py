"""
Poster figure: Ukraine birth-region tile map — wartime Δ1pl share.

White background, poster-friendly palette (Ukrainian blue diverging scale).
Standalone script that reads the same data as the narrative figure pipeline.

Usage : python src/poster_birthplace_map.py
Output: docs/Ukranian_Poetr_Report/figures/poster_birthplace_map.{pdf,png,svg}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
BG       = "none"             # transparent background for poster overlay
NAV      = "#1C3557"          # dark navy (titles, text)
BLU      = "#005BBB"          # Ukrainian blue
GOLD     = "#FFD500"          # Ukrainian yellow (accents)
DARK     = "#2C2000"

# Diverging scale: blue (decline) → cream → warm gold-rust (growth)
DIV_NEG  = "#005BBB"          # Ukrainian blue = decline
DIV_MID  = "#F5F0E6"          # warm cream = neutral
DIV_POS  = "#C87A20"          # warm amber-rust = growth

TILE_EDGE = "#8E8E8E"
FONT      = "Times New Roman"

# ── geography ────────────────────────────────────────────────────────────
REGION_LAYOUT = {
    "west_ukraine":    (0, 0),
    "kyiv":            (1, 0),
    "east_ukraine":    (2, 0),
    "central_ukraine": (1, 1),
    "south_ukraine":   (1, 2),
    "crimea":          (2, 2),
}
REGION_LABEL = {
    "west_ukraine":    "West",
    "central_ukraine": "Central",
    "kyiv":            "Kyiv",
    "east_ukraine":    "East",
    "south_ukraine":   "South",
    "crimea":          "Crimea",
    "born_abroad":     "Born abroad",
    "diaspora":        "Diaspora",
}


def _load_data():
    df = pd.read_csv(POEM_CSV)
    df = df[df["year_int"].notna()].copy()
    df["year"] = df["year_int"].astype(int)
    # keep only rows with at least one quartet pronoun
    df = df[df[list(QUARTET)].sum(axis=1) > 0].copy()

    roster = pd.read_csv(ROSTER_CSV)
    roster_set = set(roster.loc[roster["included"] == True, "author"].astype(str))

    cov = pd.read_csv(COV_CSV)

    d = df[df["author"].isin(roster_set)].copy()
    d = d[d["period3"].isin([PRE_PERIOD, POST_PERIOD])]
    denom = d[list(QUARTET)].sum(axis=1)
    d["share_1pl"] = d["1pl"] / denom.where(denom > 0, np.nan)

    per_author = (
        d.groupby(["author", "period3"])
        .agg(mean_1pl_share=("share_1pl", lambda s: float(np.nanmean(s))))
        .reset_index()
        .pivot(index="author", columns="period3", values="mean_1pl_share")
        .dropna()
        .reset_index()
    )
    per_author["delta"] = per_author[POST_PERIOD] - per_author[PRE_PERIOD]
    per_author = per_author.merge(cov[["author", "region_of_birth"]], on="author", how="left")
    per_author["region_of_birth"] = per_author["region_of_birth"].fillna("(missing)")

    region_stats = (
        per_author.groupby("region_of_birth")
        .agg(
            mean_delta=("delta", "mean"),
            n_authors=("delta", "size"),
            author_list=("author", lambda s: sorted(s.tolist())),
        )
        .reset_index()
    )
    return region_stats


def main():
    region_stats = _load_data()
    abs_max = max(0.05, float(np.nanmax(np.abs(region_stats["mean_delta"]))))

    cmap = LinearSegmentedColormap.from_list("poster_div", [DIV_NEG, DIV_MID, DIV_POS])
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    fig = plt.figure(figsize=(10.5, 8.5))
    fig.patch.set_alpha(0.0)

    grid = fig.add_gridspec(
        2, 2,
        height_ratios=[10, 1],
        width_ratios=[3.0, 1.0],
        hspace=0.22, wspace=0.15,
    )
    ax_map  = fig.add_subplot(grid[0, 0])
    ax_side = fig.add_subplot(grid[0, 1])
    ax_cbar = fig.add_subplot(grid[1, 0])

    for a in (ax_map, ax_side, ax_cbar):
        a.set_facecolor("none")

    # ── tile map ─────────────────────────────────────────────────────────
    tile_w, tile_h = 1.0, 0.80
    gap_x, gap_y = 0.08, 0.20

    for region, (col, row) in REGION_LAYOUT.items():
        s = region_stats[region_stats["region_of_birth"] == region]
        if s.empty:
            color = "#E8E2D5"
            mean_d = np.nan
            n = 0
        else:
            mean_d = float(s["mean_delta"].iloc[0])
            n = int(s["n_authors"].iloc[0])
            color = cmap(norm(mean_d))

        x = col * (tile_w + gap_x)
        y = -row * (tile_h + gap_y)

        rect = mpatches.FancyBboxPatch(
            (x, y), tile_w, tile_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=color, edgecolor=TILE_EDGE, linewidth=1.0,
        )
        ax_map.add_patch(rect)

        # choose text color: light text on dark tiles, dark text on light tiles
        is_dark = abs(mean_d) > abs_max * 0.50 if not np.isnan(mean_d) else False
        txt_main = WHITE if is_dark else NAV
        txt_sub  = "#E8E2D5" if is_dark else "#4A4A4A"

        # region name
        ax_map.text(
            x + tile_w / 2, y + tile_h - 0.10,
            REGION_LABEL.get(region, region),
            ha="center", va="top", fontsize=13, fontweight="bold",
            fontfamily=FONT, color=txt_main,
        )

        if not np.isnan(mean_d):
            # delta value
            ax_map.text(
                x + tile_w / 2, y + tile_h / 2 - 0.02,
                f"Δ1pl = {mean_d * 100:+.1f} pp",
                ha="center", va="center", fontsize=12, fontweight="bold",
                fontfamily=FONT, color=txt_main,
            )
            # author count
            ax_map.text(
                x + tile_w / 2, y + 0.10,
                f"n = {n}",
                ha="center", va="bottom", fontsize=9.5,
                fontfamily=FONT, fontstyle="italic", color=txt_sub,
            )
        else:
            ax_map.text(
                x + tile_w / 2, y + tile_h / 2,
                "no authors\nin roster",
                ha="center", va="center", fontsize=9, fontstyle="italic",
                fontfamily=FONT, color="#8E8E8E",
            )

    ax_map.set_xlim(-0.15, 3 * (tile_w + gap_x))
    ax_map.set_ylim(-2 * (tile_h + gap_y) - 0.25, tile_h + 0.30)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    ax_map.set_title(
        "Wartime Δ1pl share by author birth region",
        fontsize=15, fontweight="bold", fontfamily=FONT,
        color=NAV, pad=18, y=1.02,
    )

    # ── side panel: off-map cohorts ──────────────────────────────────────
    ax_side.axis("off")
    ax_side.set_title(
        "Off-map cohorts", fontsize=12, fontweight="bold",
        fontfamily=FONT, color=NAV, pad=10,
    )

    off_map = [
        r for r in ("born_abroad", "diaspora", "(missing)")
        if (region_stats["region_of_birth"] == r).any()
    ]
    yb = 0.92
    for special in off_map:
        s = region_stats[region_stats["region_of_birth"] == special]
        mean_d = float(s["mean_delta"].iloc[0])
        n = int(s["n_authors"].iloc[0])

        ax_side.text(
            0.02, yb,
            REGION_LABEL.get(special, special.replace("_", " ").title()),
            fontsize=12, fontweight="bold", fontfamily=FONT, color=NAV,
            transform=ax_side.transAxes,
        )
        ax_side.text(
            0.02, yb - 0.065,
            f"Δ1pl = {mean_d * 100:+.1f} pp  ·  n = {n}",
            fontsize=10, fontstyle="italic", fontfamily=FONT, color="#4A4A4A",
            transform=ax_side.transAxes,
        )
        # color stripe
        stripe = mpatches.FancyBboxPatch(
            (0.02, yb - 0.12), 0.90, 0.020,
            boxstyle="round,pad=0.003",
            facecolor=cmap(norm(mean_d)), edgecolor=TILE_EDGE, linewidth=0.5,
            transform=ax_side.transAxes, clip_on=False,
        )
        ax_side.add_patch(stripe)
        yb -= 0.26

    # ── colorbar ─────────────────────────────────────────────────────────
    gradient = np.linspace(-1, 1, 256).reshape(1, -1)
    ax_cbar.imshow(
        gradient, aspect="auto", cmap=cmap,
        extent=[-abs_max, abs_max, 0, 1],
    )
    ax_cbar.set_yticks([])
    ax_cbar.set_xticks([-abs_max, 0, abs_max])
    ax_cbar.set_xticklabels(
        [f"{-abs_max * 100:+.0f} pp", "0", f"{abs_max * 100:+.0f} pp"],
        fontsize=10, fontfamily=FONT, color=NAV,
    )
    ax_cbar.set_xlabel(
        "Mean change in 1pl share (post-2022 minus pre-2022)",
        fontsize=10, fontfamily=FONT, color="#4A4A4A",
        fontstyle="italic", labelpad=6,
    )
    for sp in ax_cbar.spines.values():
        sp.set_visible(False)
    ax_cbar.tick_params(colors=NAV, length=3)

    # ── footer ───────────────────────────────────────────────────────────
    n_total = int(region_stats["n_authors"].sum())
    fig.text(
        0.5, -0.01,
        f"Cool tones = wartime decline in 1pl share · warm tones = wartime growth · "
        f"N = {n_total} roster authors with both-period poems",
        ha="center", fontsize=8.5, fontfamily=FONT,
        color="#6F665B", fontstyle="italic",
    )

    # ── save ─────────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(
            OUT / f"poster_birthplace_map.{ext}",
            dpi=300, bbox_inches="tight",
            facecolor="none", edgecolor="none",
            transparent=True, format=ext,
        )
        print(f"saved → {OUT}/poster_birthplace_map.{ext}")
    plt.close(fig)


# constant needed for text color logic
WHITE = "#FFFFFF"

if __name__ == "__main__":
    main()
