"""Narrative publication figures for a literary-studies audience.

Replaces the HDI-forest figure family with five intuitive, story-driven
visualizations of the wartime pronoun reallocation. Each figure tells one
clear story and is built directly from the corpus + author covariates, not
from posterior summaries:

1. ``fig_narr_1_time_series_we_rises.{pdf,png}`` — the rise of 1pl in the
   first/second-person attention quartet by composition year, two lines
   (Ukrainian, Russian), with a 2022 invasion marker and prose annotations.
2. ``fig_narr_2_author_trajectories.{pdf,png}`` — each roster author drawn
   as a tail-and-arrow from their pre-2022 to post-2022 position in the
   2-D plane of (1sg-share, 1pl-share within the four-cell quartet). Named
   labels for 8 narratively important poets.
3. ``fig_narr_3_ukraine_birthplace_map.{pdf,png}`` — stylized geographic
   tile map of the seven `region_of_birth` categories, colored by the
   mean wartime change in 1pl share among authors born in that region.
4. ``fig_narr_4_generation_cohort_composition.{pdf,png}`` — for each
   generation cohort, paired horizontal stacked bars (pre / post) of the
   four-cell first/second-person attention composition.
5. ``fig_narr_5_case_study_poets.{pdf,png}`` — small-multiples panel of
   six narratively chosen poets, each showing per-year 1pl share with
   the 2022 invasion line and a brief biographic tagline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils.author_covariates import load_author_covariates
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_POEM_CELL = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_poem_cell_counts_12.csv"
DEFAULT_AUTHORS = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "q2_author_random_slope_summaries.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_q2_hierarchical" / "figures"

QUARTET_CELLS = ("1sg", "1pl", "2sg", "2pl_vy_true_plural")
CELL_HUMAN = {
    "1sg": "1sg «I»",
    "1pl": "1pl «we»",
    "2sg": "2sg «you»",
    "2pl_vy_true_plural": "2pl «you (pl.)»",
}

PRE_PERIOD = "P1_2014_2021"
POST_PERIOD = "P2_2022_plus"

# Literary palette: muted, serif-friendly, no garish primaries
COLOR_UA = "#1F4E5F"           # deep teal
COLOR_RU = "#9A4036"           # muted brick
COLOR_INVASION = "#7A0F2A"     # deep wine
COLOR_RIBBON = "#D8C9B0"       # warm cream (accent only)
COLOR_RULE = "#5F5F5F"
COLOR_ANNOT = "#2C2C2C"
# (COLOR_FACE replaced by literal "white" throughout — kept here as documentation:
#  the figure background is white to match the LaTeX page.)
COLOR_PRE = "#7A8FA6"          # cool slate (pre-war)
COLOR_POST = "#9A1750"         # burgundy (post-2022)
COLOR_GRID = "#E8E8E8"         # very light neutral grid
COLOR_MUTED = "#B8B8B8"        # neutral grey for unhighlighted

# Sequential map for the Ukraine map (positive shift = warm, negative = cool)
DIVERGING_NEG = "#3B6F8C"   # cool blue
DIVERGING_MID = "#E8E2D5"   # cream
DIVERGING_POS = "#A33B2A"   # warm rust

GEN_COHORT_ORDER = ("pre_1970", "1970s", "1980s", "1990s", "2000s_plus")
GEN_COHORT_LABEL = {
    "pre_1970": "pre-1970",
    "1970s": "1970s",
    "1980s": "1980s",
    "1990s": "1990s",
    "2000s_plus": "2000s+",
}

# Stylized geographic layout (col, row) for the Ukraine birth-region tile map.
# Rows go top→bottom; columns go left→right roughly mirroring Ukrainian geography.
REGION_LAYOUT: dict[str, tuple[int, int]] = {
    "west_ukraine":    (0, 0),
    "kyiv":            (1, 0),
    "east_ukraine":    (2, 0),
    "central_ukraine": (1, 1),
    "south_ukraine":   (1, 2),
    "crimea":          (2, 2),
}
REGION_LABEL = {
    "west_ukraine": "West",
    "central_ukraine": "Central",
    "kyiv": "Kyiv",
    "east_ukraine": "East",
    "south_ukraine": "South",
    "crimea": "Crimea",
    "born_abroad": "Born abroad",
    "diaspora": "Diaspora",
}

CASE_STUDY_POETS: list[tuple[str, str]] = [
    ("Ihor Mitrov",        "b. 1991 Kerch; 95th Air Assault Brigade since Mar 2022"),
    ("Iya Kiva",           "b. 1984 Donetsk; bilingual UA/RU; Lviv since 2022"),
    ("Yaryna Chornohuz",   "b. 1995 Kyiv; 140th Marine Recon medic; Shevchenko Prize 2024"),
    ("Serhiy Zhadan",      "b. 1974 Starobilsk; Kharkiv; 13th Khartia Brigade since May 2024"),
    ("Boris Khersonsky",   "b. 1950 Chernivtsi; Odesa; Russian-primary; PEN member"),
    ("Hryhoryi Falkovych", "b. 1940 Kyiv; Jewish-Ukrainian; evacuated to Kolomyia in 2022"),
]


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "Times", "serif"],
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "regular",
            "axes.labelsize": 10.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.6,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.4,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _quartet_share(df: pd.DataFrame, cell: str) -> pd.Series:
    """Return the per-row share of one cell within the closed four-cell quartet."""
    denom = df[list(QUARTET_CELLS)].sum(axis=1)
    return df[cell] / denom.where(denom > 0, np.nan)


def _read_inputs(poem_path: Path, roster_path: Path, authors_path: Path):
    df = pd.read_csv(poem_path)
    df = df[df["year_int"].notna()]
    df["year"] = df["year_int"].astype(int)
    df = df[df[list(QUARTET_CELLS)].sum(axis=1) > 0].copy()
    roster = pd.read_csv(roster_path)
    authors = pd.read_csv(authors_path)
    cov = load_author_covariates()
    return df, roster, authors, cov


# --- Figure 1: Time series of 1pl share ----------------------------------

def fig1_time_series_we_rises(df: pd.DataFrame, out_dir: Path) -> None:
    """Two-line yearly mean of (1pl / four-cell sum) for Ukrainian vs Russian poems.

    Years with fewer than ``MIN_POEMS_PER_YEAR_STRICT`` poems are drawn with a
    faded marker so the literary reader is not misled by single-year noise.
    """
    MIN_POEMS_PER_YEAR_STRICT = 10  # below this, fade the dot

    d = df.copy()
    d["share_1pl"] = _quartet_share(d, "1pl")
    d = d[d["language_clean"].isin(["Ukrainian", "Russian"])]
    d = d[d["share_1pl"].notna()]

    yearly = (
        d.groupby(["language_clean", "year"], as_index=False)
        .agg(
            mean_1pl_share=("share_1pl", "mean"),
            n_poems=("share_1pl", "size"),
        )
    )
    rng = np.random.default_rng(42)
    ci_low: list[float] = []
    ci_high: list[float] = []
    for _, row in yearly.iterrows():
        sub = d[
            (d["language_clean"] == row["language_clean"]) & (d["year"] == row["year"])
        ]["share_1pl"].to_numpy()
        n = len(sub)
        if n < 3:
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            continue
        boots = np.array(
            [np.mean(rng.choice(sub, size=n, replace=True)) for _ in range(500)]
        )
        ci_low.append(float(np.quantile(boots, 0.025)))
        ci_high.append(float(np.quantile(boots, 0.975)))
    yearly["ci_low"] = ci_low
    yearly["ci_high"] = ci_high

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_facecolor("white")

    y_top = 0.55  # fixed y-axis so the invasion label has clear room

    # Shade post-2022 region
    ax.axvspan(2022, yearly["year"].max() + 0.5, color=COLOR_INVASION,
               alpha=0.06, zorder=0)
    ax.axvline(2022, color=COLOR_INVASION, linewidth=1.0,
               linestyle="-", alpha=0.85, zorder=1)
    # Invasion annotation at the TOP, horizontal, with a leader
    ax.annotate(
        "24 Feb 2022 — full-scale invasion",
        xy=(2022, y_top * 0.97), xytext=(2022.4, y_top * 0.96),
        fontsize=10, color=COLOR_INVASION, fontstyle="italic",
        ha="left", va="top",
        arrowprops=dict(arrowstyle="-", color=COLOR_INVASION,
                        linewidth=0.8, alpha=0.7),
        zorder=4,
    )

    for lang, color in [("Ukrainian", COLOR_UA), ("Russian", COLOR_RU)]:
        sub = yearly[yearly["language_clean"] == lang].sort_values("year")
        if sub.empty:
            continue
        ax.fill_between(
            sub["year"], sub["ci_low"], sub["ci_high"],
            color=color, alpha=0.13, linewidth=0, zorder=2,
        )
        ax.plot(
            sub["year"], sub["mean_1pl_share"],
            color=color, linewidth=1.6, alpha=0.85, zorder=3,
        )
        # Two-tier markers: faded for sparse years
        strict = sub[sub["n_poems"] >= MIN_POEMS_PER_YEAR_STRICT]
        sparse = sub[sub["n_poems"] < MIN_POEMS_PER_YEAR_STRICT]
        ax.scatter(strict["year"], strict["mean_1pl_share"],
                   color=color, s=55, zorder=4, edgecolor="white", linewidth=0.8,
                   label=f"{lang} (n = {int(sub['n_poems'].sum())} poems)")
        ax.scatter(sparse["year"], sparse["mean_1pl_share"],
                   color=color, s=22, zorder=4, edgecolor="white", linewidth=0.4,
                   alpha=0.45)

    ax.set_xlabel("Composition year")
    ax.set_ylabel(r"Share of 1pl «we» within {1sg, 1pl, 2sg, 2pl}")
    ax.set_title(
        "The wartime rise of «we»: 1pl share in the first/second-person attention quartet",
        fontweight="regular", pad=14,
    )
    ax.set_xlim(yearly["year"].min() - 0.4, yearly["year"].max() + 0.5)
    ax.set_ylim(0, y_top)
    years_present = sorted(yearly["year"].unique())
    ax.set_xticks(years_present)
    ax.legend(loc="upper left", title="Language stratum", title_fontsize=9.5,
              bbox_to_anchor=(0.0, 1.0))

    fig.text(
        0.5, -0.02,
        f"Yearly mean share · 95 % bootstrap CI ribbons · large dots = ≥ {MIN_POEMS_PER_YEAR_STRICT} poems / year, "
        "small faded dots = sparse year (interpret cautiously)."
        f"  N = {int(yearly['n_poems'].sum())} poems · {len(years_present)} years.",
        ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic",
    )

    _save(fig, out_dir, "fig_narr_1_time_series_we_rises")
    log.info("Wrote fig_narr_1_time_series_we_rises")


# --- Figure 2: Author trajectories -----------------------------------------

def fig2_author_trajectories(
    df: pd.DataFrame, roster: pd.DataFrame, cov: pd.DataFrame, out_dir: Path
) -> None:
    """For each roster author, draw an arrow from pre-period to post-period position
    in the (1sg-share, 1pl-share) plane of the four-cell quartet.
    """
    roster_set = set(roster.loc[roster["included"] == True, "author"].astype(str))
    d = df[df["author"].isin(roster_set)].copy()
    d = d[d["period3"].isin([PRE_PERIOD, POST_PERIOD])]
    d["share_1sg"] = _quartet_share(d, "1sg")
    d["share_1pl"] = _quartet_share(d, "1pl")

    # Per-author per-period mean share, weighted by poem count (simple mean of shares)
    agg = (
        d.groupby(["author", "period3"])
        .agg(
            share_1sg=("share_1sg", lambda s: float(np.nanmean(s))),
            share_1pl=("share_1pl", lambda s: float(np.nanmean(s))),
            n_poems=("share_1sg", "size"),
        )
        .reset_index()
    )
    wide = agg.pivot(index="author", columns="period3",
                     values=["share_1sg", "share_1pl", "n_poems"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(
        subset=[f"share_1sg_{PRE_PERIOD}", f"share_1sg_{POST_PERIOD}",
                f"share_1pl_{PRE_PERIOD}", f"share_1pl_{POST_PERIOD}"]
    ).reset_index()

    wide = wide.merge(cov[["author", "generation_cohort"]], on="author", how="left")
    wide["generation_cohort"] = wide["generation_cohort"].fillna("(unknown)")

    fig, ax = plt.subplots(figsize=(11.0, 8.2))
    ax.set_facecolor("white")

    # Diagonal: constant total 1st-person share lines
    for total in (0.2, 0.4, 0.6, 0.8):
        ax.plot([0, total], [total, 0], color="#C9BFA8", linewidth=0.6,
                linestyle="--", alpha=0.7, zorder=0)
        ax.text(total + 0.008, 0.012, f"{int(total*100)}%",
                color="#9F947E", fontsize=8, fontstyle="italic", alpha=0.9)

    annotate_set = {
        "Ihor Mitrov", "Iya Kiva", "Yaryna Chornohuz", "Serhiy Zhadan",
        "Boris Khersonsky", "Hryhoryi Falkovych", "Halyna Kruk", "Elizaveta Zharikova",
    }

    # Two-tier rendering: faded background for all, prominent for highlighted
    for _, r in wide.iterrows():
        x0 = r[f"share_1sg_{PRE_PERIOD}"]
        y0 = r[f"share_1pl_{PRE_PERIOD}"]
        x1 = r[f"share_1sg_{POST_PERIOD}"]
        y1 = r[f"share_1pl_{POST_PERIOD}"]
        is_hi = r["author"] in annotate_set
        if is_hi:
            continue  # draw highlights on top
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#B0A78F", lw=0.8, alpha=0.50,
                            shrinkA=0, shrinkB=0, mutation_scale=10),
            zorder=2,
        )
        ax.scatter([x0], [y0], color="#B0A78F", s=14, alpha=0.55, zorder=3,
                   edgecolor="white", linewidth=0.4)

    # Highlighted poets — colored by gender as a literary-friendly grouping
    cov_subset = cov.set_index("author")["gender"].to_dict() if "gender" in cov.columns else {}
    color_by_gender = {"F": "#9A1750", "M": "#2E5266", "NB": "#6B2C5C"}

    hi_rows = wide[wide["author"].isin(annotate_set)]
    label_offsets = {
        "Ihor Mitrov":         (12, -18),
        "Iya Kiva":            (10, 14),
        "Yaryna Chornohuz":    (12, -22),
        "Serhiy Zhadan":       (-20, 16),
        "Boris Khersonsky":    (-110, -20),
        "Hryhoryi Falkovych":  (-110, 14),
        "Halyna Kruk":         (-90, -22),
        "Elizaveta Zharikova": (12, 14),
    }
    for _, r in hi_rows.iterrows():
        x0 = r[f"share_1sg_{PRE_PERIOD}"]
        y0 = r[f"share_1pl_{PRE_PERIOD}"]
        x1 = r[f"share_1sg_{POST_PERIOD}"]
        y1 = r[f"share_1pl_{POST_PERIOD}"]
        c = color_by_gender.get(str(cov_subset.get(r["author"], "")).strip(), "#444444")
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, alpha=0.95,
                            shrinkA=0, shrinkB=0, mutation_scale=14),
            zorder=5,
        )
        ax.scatter([x0], [y0], color=c, s=42, zorder=6,
                   edgecolor="white", linewidth=0.9)
        ax.scatter([x1], [y1], color=c, s=10, marker="s", zorder=6,
                   edgecolor="white", linewidth=0.4)
        dx, dy = label_offsets.get(r["author"], (10, 10))
        ax.annotate(
            r["author"],
            xy=(x1, y1), xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5, color=COLOR_ANNOT, fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFFCF5",
                      ec="#9F947E", lw=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#9F947E", lw=0.5, alpha=0.7,
                            connectionstyle="arc3,rad=0.05"),
            zorder=7,
        )

    ax.set_xlabel(r"Mean share of 1sg «I» within the quartet  →")
    ax.set_ylabel(r"Mean share of 1pl «we» within the quartet  →")
    ax.set_title(
        "Author trajectories: pre-war «I» / «we» footprint and its wartime drift",
        pad=12,
    )

    # Legend: highlighted vs background + gender colors
    handles = [
        Line2D([], [], marker="o", linestyle="None", color=color_by_gender["F"],
               label="Highlighted female poet", markersize=8),
        Line2D([], [], marker="o", linestyle="None", color=color_by_gender["M"],
               label="Highlighted male poet", markersize=8),
        Line2D([], [], marker="o", linestyle="None", color="#B0A78F",
               label="Other roster author (faded)", markersize=6, alpha=0.6),
        Line2D([], [], marker=">", linestyle="-", color=COLOR_ANNOT,
               label="dot = pre-2022, arrow = wartime drift", markersize=8,
               markeredgewidth=0.0),
    ]
    ax.legend(handles=handles, loc="upper right",
              title_fontsize=9.5, framealpha=0.95, edgecolor="#B0A78F",
              fancybox=True)

    ax.set_xlim(0, max(0.85, wide[[f"share_1sg_{PRE_PERIOD}", f"share_1sg_{POST_PERIOD}"]].max().max() * 1.08))
    ax.set_ylim(0, max(0.85, wide[[f"share_1pl_{PRE_PERIOD}", f"share_1pl_{POST_PERIOD}"]].max().max() * 1.10))

    fig.text(
        0.5, -0.015,
        f"N = {len(wide)} roster authors with poems in both periods. "
        "Dashed diagonals mark constant total 1st-person share (20–80 %).  "
        "Eight narratively notable poets highlighted by gender.",
        ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic",
    )

    _save(fig, out_dir, "fig_narr_2_author_trajectories")
    log.info("Wrote fig_narr_2_author_trajectories (n=%d authors)", len(wide))


# --- Figure 3: Ukraine birth-region tile map -------------------------------

def fig3_ukraine_birthplace_map(
    df: pd.DataFrame, roster: pd.DataFrame, cov: pd.DataFrame, out_dir: Path
) -> None:
    roster_set = set(roster.loc[roster["included"] == True, "author"].astype(str))
    d = df[df["author"].isin(roster_set)].copy()
    d = d[d["period3"].isin([PRE_PERIOD, POST_PERIOD])]
    d["share_1pl"] = _quartet_share(d, "1pl")

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
        .agg(mean_delta=("delta", "mean"), n_authors=("delta", "size"),
             author_list=("author", lambda s: sorted(s.tolist())))
        .reset_index()
    )

    fig = plt.figure(figsize=(11.5, 8.0))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 2, height_ratios=[10, 1], width_ratios=[3.0, 1.0],
                            hspace=0.20, wspace=0.18)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_side = fig.add_subplot(grid[0, 1])
    ax_cbar = fig.add_subplot(grid[1, 0])
    ax_map.set_facecolor("white")
    ax_side.set_facecolor("white")
    ax_cbar.set_facecolor("white")

    abs_max = max(0.05, float(np.nanmax(np.abs(region_stats["mean_delta"]))))
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list(
        "lit_diverging",
        [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS],
    )
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    # Draw tile map — slightly smaller tiles, more vertical gap
    tile_w = 1.0
    tile_h = 0.80
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
        x = col * (tile_w + 0.06)
        y = -row * (tile_h + 0.18)
        rect = mpatches.FancyBboxPatch(
            (x, y), tile_w, tile_h, boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=color, edgecolor="#6F665B", linewidth=0.8,
        )
        ax_map.add_patch(rect)
        # Region label inside top portion of tile
        ax_map.text(
            x + tile_w / 2, y + tile_h - 0.12, REGION_LABEL.get(region, region),
            ha="center", va="top", fontsize=11.5, fontweight="semibold", color="#252525",
        )
        if not np.isnan(mean_d):
            ax_map.text(
                x + tile_w / 2, y + tile_h / 2 - 0.04,
                f"Δ 1pl = {mean_d*100:+.1f} pp",
                ha="center", va="center", fontsize=10.5,
                color="#1f1f1f" if abs(mean_d) < abs_max*0.55 else "#fdf7ec",
                fontweight="semibold",
            )
            ax_map.text(
                x + tile_w / 2, y + 0.10,
                f"n = {n} author{'s' if n != 1 else ''}",
                ha="center", va="bottom", fontsize=8.5,
                color="#3f3f3f" if abs(mean_d) < abs_max*0.55 else "#fdf7ec",
                fontstyle="italic",
            )
        else:
            ax_map.text(
                x + tile_w / 2, y + tile_h / 2,
                "no authors\nin roster",
                ha="center", va="center", fontsize=9, color="#6F665B",
                fontstyle="italic",
            )
    ax_map.set_xlim(-0.15, 3 * (tile_w + 0.06))
    ax_map.set_ylim(-2 * (tile_h + 0.18) - 0.20, tile_h + 0.25)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    ax_map.set_title(
        "Wartime change in 1pl share by birth region",
        fontsize=13.5, pad=18, y=1.02,
    )

    # Side panel: off-map cohorts (only those present)
    ax_side.axis("off")
    ax_side.set_title("Off-map cohorts", fontsize=11, pad=10,
                      color="#3F3F3F")
    off_map = [r for r in ("born_abroad", "diaspora", "(missing)")
               if (region_stats["region_of_birth"] == r).any()]
    yb = 0.95
    for special in off_map:
        s = region_stats[region_stats["region_of_birth"] == special]
        mean_d = float(s["mean_delta"].iloc[0])
        n = int(s["n_authors"].iloc[0])
        ax_side.text(0.02, yb, REGION_LABEL.get(special, special.replace("_", " ").title()),
                     fontsize=11, fontweight="semibold", color="#1F1F1F")
        ax_side.text(0.02, yb - 0.06,
                     f"Δ 1pl = {mean_d*100:+.1f} pp   ·   n = {n}",
                     fontsize=9.5, color="#3F3F3F", fontstyle="italic")
        stripe = mpatches.Rectangle(
            (0.02, yb - 0.105), 0.92, 0.014,
            color=cmap(norm(mean_d)), transform=ax_side.transAxes, clip_on=False,
        )
        ax_side.add_patch(stripe)
        yb -= 0.24

    # Colorbar as its own axes row (no overlap)
    ax_cbar.imshow(np.linspace(-1, 1, 256).reshape(1, -1), aspect="auto",
                   cmap=cmap, extent=[-abs_max, abs_max, 0, 1])
    ax_cbar.set_yticks([])
    ax_cbar.set_xticks([-abs_max, 0, abs_max])
    ax_cbar.set_xticklabels(
        [f"{-abs_max*100:+.0f} pp", "0", f"{abs_max*100:+.0f} pp"],
        fontsize=9.5,
    )
    ax_cbar.set_xlabel(
        "mean change in 1pl share among authors born in this region (post − pre)",
        fontsize=9.5, color="#3F3F3F", fontstyle="italic", labelpad=4,
    )
    for s in ax_cbar.spines.values():
        s.set_visible(False)

    fig.text(0.5, -0.015,
             "Diverging palette: cool = wartime decline in 1pl share, warm = wartime growth. "
             f"N = {int(region_stats['n_authors'].sum())} roster authors with both-period poems.",
             ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic")

    _save(fig, out_dir, "fig_narr_3_ukraine_birthplace_map")
    log.info("Wrote fig_narr_3_ukraine_birthplace_map")


# --- Figure 4: Generation cohort composition pre vs post ------------------

def fig4_generation_cohort_composition(
    df: pd.DataFrame, roster: pd.DataFrame, cov: pd.DataFrame, out_dir: Path
) -> None:
    roster_set = set(roster.loc[roster["included"] == True, "author"].astype(str))
    d = df[df["author"].isin(roster_set)].copy()
    d = d[d["period3"].isin([PRE_PERIOD, POST_PERIOD])]
    d = d.merge(cov[["author", "generation_cohort"]], on="author", how="left")

    rows = []
    for (coh, per), sub in d.groupby(["generation_cohort", "period3"]):
        denom = sub[list(QUARTET_CELLS)].to_numpy().sum()
        if denom <= 0:
            continue
        sums = sub[list(QUARTET_CELLS)].sum()
        rows.append({
            "generation_cohort": coh,
            "period": per,
            "n_authors": sub["author"].nunique(),
            "n_poems": len(sub),
            **{f"share_{c}": float(sums[c]) / denom for c in QUARTET_CELLS},
        })
    comp = pd.DataFrame(rows)
    comp = comp[comp["generation_cohort"].isin(GEN_COHORT_ORDER)]
    if comp.empty:
        log.warning("Skipping fig4: no cohort composition data.")
        return

    cohorts = [c for c in GEN_COHORT_ORDER if c in comp["generation_cohort"].unique()]
    n_coh = len(cohorts)

    cell_colors = {
        "1sg":                 "#7A8FA6",   # cool slate
        "1pl":                 "#A33B2A",   # warm rust (the wartime star)
        "2sg":                 "#C9A961",   # muted gold
        "2pl_vy_true_plural":  "#4A6741",   # forest
    }

    BAR_H = 0.34
    GAP_WITHIN = 0.10   # gap between pre/post bars of a cohort
    GAP_BETWEEN = 0.62  # gap between cohorts

    # Compute y positions
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

    total_height = abs(y_cursor)
    fig, ax = plt.subplots(figsize=(11.5, max(5.6, 0.55 * total_height + 1.4)))
    ax.set_facecolor("white")

    # Title bar BELOW the figure title — extra top padding to host the legend
    legend_room = 0.10  # in figure-fraction units, reserved at the top via rect

    LEFT_LABEL_X = -0.21  # negative axes coords → left margin
    LEFT_N_X = -0.018      # near 0 → just inside the axes

    for coh in cohorts:
        for period in (PRE_PERIOD, POST_PERIOD):
            row = comp[(comp["generation_cohort"] == coh) & (comp["period"] == period)]
            if row.empty:
                continue
            y = y_positions[(coh, period)]
            left = 0.0
            for cell in QUARTET_CELLS:
                w = float(row[f"share_{cell}"].iloc[0])
                ax.barh(y, w, height=BAR_H, left=left,
                        color=cell_colors[cell], edgecolor="white", linewidth=0.7)
                if w >= 0.04:
                    ax.text(left + w / 2, y, f"{w*100:.0f}%",
                            ha="center", va="center", fontsize=9,
                            color="white" if cell in ("1pl", "2pl_vy_true_plural") else "#1f1f1f")
                left += w
            label_short = "pre-2022" if period == PRE_PERIOD else "post-2022"
            n_a = int(row["n_authors"].iloc[0])
            n_p = int(row["n_poems"].iloc[0])
            ax.text(LEFT_N_X, y,
                    f"{label_short}  ·  {n_a} authors · {n_p} poems",
                    ha="right", va="center", fontsize=9, color="#3f3f3f")
        # Cohort label far left, centered between the pair
        ax.text(LEFT_LABEL_X, cohort_centers[coh],
                GEN_COHORT_LABEL[coh],
                ha="left", va="center", fontsize=12.5, fontweight="semibold",
                color="#1f1f1f", transform=ax.get_yaxis_transform())

    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of the four-cell first/second-person attention quartet")
    ax.set_ylim(y_cursor + GAP_BETWEEN - 0.30, BAR_H + 0.30)

    # Bigger left margin so the cohort labels have room
    ax.spines["left"].set_visible(False)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.85, bottom=0.16)

    fig.suptitle(
        "Generation cohorts: first/second-person composition before vs after the invasion",
        fontsize=13.5, y=0.965, x=0.555,
    )

    handles = [
        mpatches.Patch(color=cell_colors[c], label=CELL_HUMAN[c]) for c in QUARTET_CELLS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.555, 0.925),
               ncol=4, frameon=False, fontsize=10)

    fig.text(0.555, 0.025,
             "Each pair compares pre-2022 (top) vs post-2022 (bottom) corpus composition. "
             "Cohort labels in the left margin; sample sizes inset on each bar.",
             ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic")

    _save(fig, out_dir, "fig_narr_4_generation_cohort_composition")
    log.info("Wrote fig_narr_4_generation_cohort_composition")


# --- Figure 5: Case-study poets --------------------------------------------

def fig5_case_study_poets(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.copy()
    d["share_1pl"] = _quartet_share(d, "1pl")

    ncols = 2
    nrows = int(np.ceil(len(CASE_STUDY_POETS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.1 * nrows + 1.0),
                             sharey=True)
    axes_flat = np.asarray(axes).ravel()
    fig.patch.set_facecolor("white")

    for ax in axes_flat:
        ax.set_facecolor("white")

    full_year_range = list(range(2013, int(d["year"].max()) + 1))

    for i, (poet, tagline) in enumerate(CASE_STUDY_POETS):
        ax = axes_flat[i]
        sub = d[d["author"] == poet].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        per_year = (
            sub.groupby("year")
            .agg(mean_share=("share_1pl", lambda s: float(np.nanmean(s))),
                 n_poems=("share_1pl", "size"))
            .reset_index()
            .sort_values("year")
        )
        ax.axvspan(2022, full_year_range[-1] + 0.5,
                   color=COLOR_INVASION, alpha=0.07, zorder=0)
        ax.axvline(2022, color=COLOR_INVASION, linewidth=1.0,
                   linestyle="-", alpha=0.85, zorder=1)
        ax.plot(per_year["year"], per_year["mean_share"],
                marker="o", color=COLOR_UA, linewidth=1.9, markersize=6,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        for _, r in per_year.iterrows():
            ax.text(r["year"], r["mean_share"] + 0.06,
                    f"{int(r['n_poems'])}", ha="center", va="bottom",
                    fontsize=8, color="#888")
        # Poet title with bio tagline as subtitle inline (no in-plot box)
        ax.set_title(
            f"{poet}\n", fontsize=12.0, fontweight="semibold",
            color="#1f1f1f", pad=22, loc="left",
        )
        ax.text(
            0.0, 1.022, tagline, transform=ax.transAxes,
            fontsize=8.8, color="#5a5a5a", fontstyle="italic", va="bottom", ha="left",
        )

        ax.set_ylim(0, 1.0)
        ax.set_xlim(full_year_range[0] - 0.5, full_year_range[-1] + 0.5)
        ax.set_xticks(full_year_range)
        ax.set_xticklabels([str(y) for y in full_year_range], rotation=45, fontsize=8.5)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    for j in range(len(CASE_STUDY_POETS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    for row_idx in range(nrows):
        axes_flat[row_idx * ncols].set_ylabel("1pl share within quartet")
    for col_idx in range(ncols):
        bottom_idx = (nrows - 1) * ncols + col_idx
        if bottom_idx < len(axes_flat) and axes_flat[bottom_idx].get_visible():
            axes_flat[bottom_idx].set_xlabel("year")

    fig.suptitle(
        "Six poets in close-up: per-year share of «we» in the first/second-person attention quartet",
        fontsize=13.5, y=0.995,
    )
    fig.text(0.5, 0.005,
             "Vertical wine line = 24 Feb 2022 invasion.  Small gray numbers above each dot = poems in that year. "
             "Common x-axis: 2013 → present.",
             ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic")
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.965], h_pad=2.4)

    _save(fig, out_dir, "fig_narr_5_case_study_poets")
    log.info("Wrote fig_narr_5_case_study_poets")


# --- Figure 7: Q1 person×number redistribution ---------------------------

def fig7_person_number_redistribution(df: pd.DataFrame, roster: pd.DataFrame, out_dir: Path) -> None:
    """Pre vs wartime 2x2 share matrices of the closed first/second-person quartet.

    Two clean tile matrices side by side; small per-cell Δ annotations
    (in percentage points) make the period delta legible inside the figure
    itself. All interpretive prose --- including the within-plural conditional
    share and the person×number odds ratio --- now lives in the body text
    (\\S 4.1.4), not in an in-figure callout panel.
    """
    roster_set = set(roster.loc[roster["included"] == True, "author"].astype(str))
    d = df[df["author"].isin(roster_set)].copy()
    d = d[d["period3"].isin([PRE_PERIOD, POST_PERIOD])]
    d = d[d["language_clean"].isin(["Ukrainian", "Russian"])]
    d["denom"] = d[list(QUARTET_CELLS)].sum(axis=1)
    d = d[d["denom"] > 0]

    def _matrix(period: str) -> tuple[pd.DataFrame, int]:
        sub = d[d["period3"] == period]
        sums = sub[list(QUARTET_CELLS)].sum()
        denom = int(sub["denom"].sum())
        mat = pd.DataFrame(
            {
                "singular": [float(sums["1sg"]) / denom, float(sums["2sg"]) / denom],
                "plural":   [float(sums["1pl"]) / denom, float(sums["2pl_vy_true_plural"]) / denom],
            },
            index=["1st", "2nd"],
        )
        return mat, denom

    pre_mat, pre_n = _matrix(PRE_PERIOD)
    post_mat, post_n = _matrix(POST_PERIOD)
    delta_mat = post_mat - pre_mat  # percentage-point change

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "lit_seq", ["#FFFFFF", "#9A1750"]
    )
    vmax = float(max(pre_mat.values.max(), post_mat.values.max()))

    fig = plt.figure(figsize=(9.6, 4.8))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, wspace=0.18)
    ax_pre = fig.add_subplot(grid[0, 0])
    ax_post = fig.add_subplot(grid[0, 1])
    for ax in (ax_pre, ax_post):
        ax.set_facecolor("white")

    def _draw_matrix(ax, mat: pd.DataFrame, period_label: str, n: int,
                      delta: pd.DataFrame | None = None) -> None:
        for i, row_label in enumerate(mat.index):
            for j, col_label in enumerate(mat.columns):
                v = float(mat.iloc[i, j])
                color = cmap(v / vmax)
                rect = mpatches.Rectangle(
                    (j, -i - 1.0), 0.96, 0.96,
                    facecolor=color, edgecolor="#444444", linewidth=0.7,
                )
                ax.add_patch(rect)
                text_color = "white" if v / vmax > 0.55 else "#1f1f1f"
                ax.text(j + 0.48, -i - 0.42, f"{v*100:.1f}%",
                        ha="center", va="center",
                        fontsize=16, fontweight="semibold", color=text_color)
                cell_id = {(0, 0): "1sg", (0, 1): "1pl",
                           (1, 0): "2sg", (1, 1): "2pl"}.get((i, j), "")
                ax.text(j + 0.48, -i - 0.72, cell_id,
                        ha="center", va="center",
                        fontsize=10, color=text_color, fontstyle="italic")
                if delta is not None:
                    dv = float(delta.iloc[i, j]) * 100
                    sign = "+" if dv >= 0 else ""
                    ax.text(j + 0.48, -i - 0.90, f"{sign}{dv:.1f} pp",
                            ha="center", va="center",
                            fontsize=9.5, color=text_color, fontstyle="italic")
        ax.set_xlim(-0.05, 2.0)
        ax.set_ylim(-2.15, 0.30)
        ax.set_xticks([0.48, 1.48])
        ax.set_xticklabels(["singular", "plural"], fontsize=11)
        ax.set_yticks([-0.52, -1.52])
        ax.set_yticklabels(["1st", "2nd"], fontsize=11)
        ax.tick_params(left=False, bottom=False, pad=4)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(
            f"{period_label}\n(n = {n:,} pronoun tokens)",
            fontsize=11.5, fontweight="regular", pad=6,
        )

    _draw_matrix(ax_pre, pre_mat, "Pre-war (2014–2021)", pre_n, delta=None)
    _draw_matrix(ax_post, post_mat, "Wartime (2022–present)", post_n, delta=delta_mat)

    fig.suptitle(
        "Token-weighted share of the closed first/second-person quartet, pre vs wartime",
        fontsize=12.5, y=1.04,
    )
    fig.text(
        0.5, -0.02,
        "Roster authors only ($n = 33$). Wartime tiles annotate the period delta in percentage points (pp).",
        ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic",
    )

    _save(fig, out_dir, "fig_narr_7_person_number_redistribution")
    log.info("Wrote fig_narr_7_person_number_redistribution (clean 2x2, no callout)")


# --- Figure 8: focused single-cell (1pl) author caterpillar ---------------

def fig8_caterpillar_1pl_focused(authors: pd.DataFrame, out_dir: Path,
                                  stratum: str = "pooled_Ukrainian_Russian",
                                  highlight_top_n: int = 5,
                                  highlight_bottom_n: int = 5) -> None:
    """Single-cell (1pl) author caterpillar designed to fit one page with a focal point.

    Replaces the 5-cell `fig_q2_author_random_slope_caterpillar_pooled_*` produced
    by the modeling script. Authors are sorted by posterior mean total period-shift
    on the 1pl cell; the top `highlight_top_n` and bottom `highlight_bottom_n` are
    drawn with literary-palette colors and named in-figure labels, while the
    middle band is greyed out as context. Designed to fit a single 0.95-width
    LaTeX figure on one printed page.
    """
    if authors.empty:
        log.warning("Skipping fig8: authors table empty.")
        return
    d = authors[(authors["language_stratum"] == stratum) & (authors["cell"] == "1pl")].copy()
    if d.empty:
        log.warning("Skipping fig8: no rows for (stratum=%s, cell=1pl).", stratum)
        return

    # Sort by total period-shift posterior mean (descending = strongest positive at top)
    d = d.sort_values("author_total_period_shift_mean_log_mu", ascending=True).reset_index(drop=True)
    n = len(d)

    # Define highlight masks
    top_idx = set(range(n - highlight_top_n, n))
    bot_idx = set(range(highlight_bottom_n))
    hi_idx = top_idx | bot_idx

    # Colors
    COLOR_POS = "#1F4E5F"     # deep teal for positive shifts (top)
    COLOR_NEG = "#9A1750"     # burgundy for negative shifts (bottom)
    COLOR_MID = "#C8C2B6"     # warm muted grey for unhighlighted middle

    fig_h = max(7.5, 0.20 * n + 2.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    ax.set_facecolor("white")

    x_mean = d["author_total_period_shift_mean_log_mu"].to_numpy(float)
    x_lo = d["author_total_period_shift_hdi95_low"].to_numpy(float)
    x_hi = d["author_total_period_shift_hdi95_high"].to_numpy(float)
    y = np.arange(n)

    # Reference lines
    ax.axvline(0.0, color="#444444", linestyle="-", linewidth=0.9, alpha=0.85, zorder=0)
    # Population-mean line (dashed) — average of the per-author totals
    pop_mean = float(np.nanmean(x_mean))
    ax.axvline(pop_mean, color="#7A7A7A", linestyle="--", linewidth=0.8,
               alpha=0.8, zorder=0)
    ax.text(pop_mean, n - 0.4, f"  pop. mean = {pop_mean:+.2f}",
            color="#5A5A5A", fontsize=8.5, fontstyle="italic",
            va="center", ha="left")

    # Background middle: greyed
    middle = [i for i in range(n) if i not in hi_idx]
    if middle:
        m_arr = np.array(middle)
        ax.hlines(m_arr, x_lo[m_arr], x_hi[m_arr],
                  color=COLOR_MID, linewidth=1.4, alpha=0.7, zorder=1)
        ax.plot(x_mean[m_arr], m_arr, "o",
                color=COLOR_MID, markersize=3.4, alpha=0.85, zorder=2,
                markeredgecolor="white", markeredgewidth=0.4)

    # Top (positive)
    if top_idx:
        t_arr = np.array(sorted(top_idx))
        ax.hlines(t_arr, x_lo[t_arr], x_hi[t_arr],
                  color=COLOR_POS, linewidth=2.0, alpha=0.95, zorder=3)
        ax.plot(x_mean[t_arr], t_arr, "o",
                color=COLOR_POS, markersize=5.6, zorder=4,
                markeredgecolor="white", markeredgewidth=0.7)

    # Bottom (negative)
    if bot_idx:
        b_arr = np.array(sorted(bot_idx))
        ax.hlines(b_arr, x_lo[b_arr], x_hi[b_arr],
                  color=COLOR_NEG, linewidth=2.0, alpha=0.95, zorder=3)
        ax.plot(x_mean[b_arr], b_arr, "o",
                color=COLOR_NEG, markersize=5.6, zorder=4,
                markeredgecolor="white", markeredgewidth=0.7)

    # Author labels — show all author names on the y-axis
    ax.set_yticks(y)
    labels = []
    label_colors = []
    for i, row in d.iterrows():
        rr = float(row["author_total_period_shift_rate_ratio_mean"])
        if i in top_idx:
            labels.append(f"{row['author']}   (RR={rr:.2f})")
            label_colors.append(COLOR_POS)
        elif i in bot_idx:
            labels.append(f"{row['author']}   (RR={rr:.2f})")
            label_colors.append(COLOR_NEG)
        else:
            labels.append(row["author"])
            label_colors.append("#666666")
    ax.set_yticklabels(labels, fontsize=8.5)
    for tick_label, c in zip(ax.get_yticklabels(), label_colors):
        tick_label.set_color(c)
    ax.tick_params(left=False)

    # Layout
    span_lo = float(np.nanmin(x_lo))
    span_hi = float(np.nanmax(x_hi))
    pad = max(0.05 * (span_hi - span_lo), 0.15)
    ax.set_xlim(span_lo - pad, span_hi + pad)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(
        r"Per-author total period-shift posterior on the 1pl cell  (log rate, P2 $-$ P1)",
        fontsize=10.5,
    )

    # Top axis with rate-ratio scale
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda x: np.exp(x), lambda x: np.log(np.clip(x, 1e-6, None))),
    )
    sec.set_xlabel("Rate ratio (post / pre)", fontsize=10)
    sec.tick_params(axis="x", labelsize=9)

    # Title and legend
    ax.set_title(
        "Per-author 1pl period-shift posterior, pooled Ukrainian + Russian stratum",
        fontsize=12, pad=20,
    )

    handles = [
        Line2D([], [], marker="o", color=COLOR_POS, linewidth=2,
               label=f"top-{highlight_top_n} positive shifts", markersize=7),
        Line2D([], [], marker="o", color=COLOR_NEG, linewidth=2,
               label=f"bottom-{highlight_bottom_n} negative shifts", markersize=7),
        Line2D([], [], marker="o", color=COLOR_MID, linewidth=1.4,
               label="middle band (context)", markersize=5, alpha=0.7),
    ]
    ax.legend(handles=handles, loc="lower right",
              fontsize=9, framealpha=0.95, edgecolor="#B0A78F")

    fig.text(
        0.5, -0.005,
        f"N = {n} roster authors with poems in both periods. "
        "Thick segments are 95\\% highest-density intervals; thin vertical at zero, dashed at population mean.",
        ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic",
    )
    fig.tight_layout(rect=[0.0, 0.015, 1.0, 1.0])

    _save(fig, out_dir, "fig_narr_8_caterpillar_1pl_focused")
    log.info("Wrote fig_narr_8_caterpillar_1pl_focused (n=%d)", n)


# --- Figure 6 (redo): covariate coverage with literary palette ------------

def fig_redo_covariate_coverage(
    cov: pd.DataFrame, roster: pd.DataFrame, out_dir: Path
) -> None:
    roster_set = set(roster.loc[roster["included"] == True, "author"].astype(str))
    d = cov[cov["author"].astype(str).isin(roster_set)].copy()
    predictors = ["gender", "generation_cohort", "region_of_birth", "language_corpus_p1"]
    title_map = {
        "gender": "Gender",
        "generation_cohort": "Generation cohort",
        "region_of_birth": "Region of birth",
        "language_corpus_p1": "Empirical P1 language",
    }
    fig, axes = plt.subplots(1, len(predictors), figsize=(13.2, 3.6))
    for ax, pred in zip(axes, predictors):
        ax.set_facecolor("white")
        levels = d[pred].astype(str).str.strip().replace("", "(missing)")
        counts = levels.value_counts()
        counts = counts.reindex(sorted(counts.index, key=lambda x: (x == "(missing)", x)))
        colors = []
        for lbl in counts.index:
            if lbl == "(missing)":
                colors.append("#C9BFA8")
            elif pred == "gender":
                colors.append({"F": COLOR_UA, "M": COLOR_RU, "NB": "#6B2C5C"}.get(lbl, "#7A8FA6"))
            elif pred == "language_corpus_p1":
                colors.append({"Ukrainian": COLOR_UA, "Russian": COLOR_RU,
                               "bilingual": "#7A6A4F", "other": "#999"}.get(lbl, "#7A8FA6"))
            else:
                colors.append("#7A8FA6")
        bars = ax.bar(range(len(counts)), counts.values, color=colors,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.18,
                    f"{int(v)}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.tolist(), rotation=30, ha="right", fontsize=9)
        ax.set_title(title_map[pred], fontsize=11.5, pad=4)
        ax.set_ylim(0, counts.max() * 1.20 if counts.max() else 1)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", labelleft=False, left=False)
    fig.suptitle(
        f"Roster sample composition (n_authors = {int(d['author'].nunique())})",
        fontsize=12.5, y=1.02,
    )
    fig.subplots_adjust(bottom=0.30, top=0.86)
    fig.text(0.5, 0.04,
             "Counts of authors in the modeled roster by each time-invariant predictor used in Q2 covariate adjustment.",
             ha="center", fontsize=8.5, color=COLOR_RULE, fontstyle="italic")
    _save(fig, out_dir, "fig_narr_6_roster_sample_composition")
    log.info("Wrote fig_narr_6_roster_sample_composition")


# --- main ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poem-cell", type=Path, default=DEFAULT_POEM_CELL)
    parser.add_argument("--authors", type=Path, default=DEFAULT_AUTHORS)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    _setup_style()
    df, roster, authors, cov = _read_inputs(
        args.poem_cell.resolve(), args.roster.resolve(), args.authors.resolve()
    )
    out_dir = args.output.resolve()

    fig1_time_series_we_rises(df, out_dir)
    fig2_author_trajectories(df, roster, cov, out_dir)
    fig3_ukraine_birthplace_map(df, roster, cov, out_dir)
    fig4_generation_cohort_composition(df, roster, cov, out_dir)
    fig5_case_study_poets(df, out_dir)
    fig7_person_number_redistribution(df, roster, out_dir)
    fig8_caterpillar_1pl_focused(authors, out_dir)
    fig_redo_covariate_coverage(cov, roster, out_dir)

    log.info("Done. Narrative figures in %s", out_dir)


if __name__ == "__main__":
    main()
