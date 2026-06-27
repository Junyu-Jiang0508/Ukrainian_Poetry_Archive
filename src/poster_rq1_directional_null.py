"""
Poster figure: RQ1 result — directional null at the corpus mean.
Two panels:
  A) Specification curve for 1pl IRR across language strata (Ukrainian, Russian, pooled).
     30 specs · all CIs span 1.0 · UA and pooled 100% positive.
  B) Hierarchical posterior for pooled 1pl δ.
     P(δ>0) = 76 % · HDI95 still spans 1.0.

Usage : python src/poster_rq1_directional_null.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq1_directional_null.{pdf,png,svg}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

BG     = "#FFD500"
NAV    = "#1C3557"
BLU    = "#005BBB"
HL     = "#F4A200"
DOWN_C = "#6F7A8A"   # cool grey-blue for "below 1.0" half
UP_C   = "#1C3557"   # warm navy for "above 1.0" half (the story)
RULE_C = "#9A7B00"
DARK   = "#2C2000"
WHITE  = "#FFFFFF"

# Stratum-specific colours (sequential, visually distinct)
COL_POOL = "#1C3557"
COL_UA   = "#005BBB"
COL_RU   = "#7AA9D6"

FONT = "Times New Roman"
ROOT = Path(__file__).parent.parent
OUT  = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"

FIG_W, FIG_H = 11.5, 5.4


def setup_axis(ax, bg="none"):
    ax.set_facecolor(bg)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=DARK, length=0)


def panel_spec_curve(ax, spec_df):
    """
    Specification curve for 1pl IRR.
    Three strata stacked vertically (pooled top, UA middle, RU bottom).
    Each spec is a dot + 95% CI, sorted by IRR within stratum.
    """
    setup_axis(ax)
    ax.set_xscale("log")

    strata_order = [
        ("pooled_Ukrainian_Russian", "Pooled",   COL_POOL, "All UA + RU"),
        ("Ukrainian",                "Ukrainian", COL_UA,   "Primary stratum"),
        ("Russian",                  "Russian",   COL_RU,   "Secondary stratum"),
    ]
    n_strata = len(strata_order)
    spec_count_per_stratum = 10
    band_h = 1.0                 # vertical room per stratum band
    band_gap = 0.4
    total_h = n_strata * band_h + (n_strata - 1) * band_gap

    band_centers = []
    for i, _ in enumerate(strata_order):
        # top stratum at high y
        center = total_h - (i * (band_h + band_gap) + band_h / 2)
        band_centers.append(center)

    # x-limits on log scale
    x_min, x_max = 0.55, 3.6
    ax.axvline(1.0, color=RULE_C, lw=1.2, ls=(0, (4, 3)), zorder=1, alpha=0.9)
    ax.text(1.0, total_h + 0.08, "no shift", ha="center", va="bottom",
            color=RULE_C, fontsize=8.8, fontstyle="italic",
            fontfamily=FONT)

    for (stratum_key, stratum_label, col, sublabel), cy in zip(strata_order, band_centers):
        d = spec_df[(spec_df["cell"] == "1pl") & (spec_df["language_stratum"] == stratum_key)].copy()
        d = d.sort_values("rate_ratio_post_vs_pre").reset_index(drop=True)

        # spread vertically across the band
        n = len(d)
        y_positions = cy + np.linspace(-band_h * 0.42, band_h * 0.42, n)

        # CI segments
        for (_, row), y in zip(d.iterrows(), y_positions):
            ax.plot([row["rate_ratio_ci95_low"], row["rate_ratio_ci95_high"]],
                    [y, y], color=col, lw=1.6, alpha=0.55, zorder=2,
                    solid_capstyle="round")
            ax.scatter([row["rate_ratio_post_vs_pre"]], [y],
                       s=46, color=col, zorder=3,
                       edgecolor="white", linewidth=0.9)

        # stratum label on the left
        ax.text(x_min * 0.92, cy + 0.06, stratum_label,
                ha="right", va="center", fontsize=12, fontweight="bold",
                fontfamily=FONT, color=col)
        ax.text(x_min * 0.92, cy - 0.20, sublabel,
                ha="right", va="center", fontsize=8.2,
                fontstyle="italic", fontfamily=FONT, color=DARK, alpha=0.75)

        # within-stratum direction tally
        n_pos = (d["rate_ratio_post_vs_pre"] > 1.0).sum()
        tally = f"{n_pos}/{n} specs > 1.0"
        ax.text(x_max * 1.03, cy + 0.06, tally,
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                fontfamily=FONT, color=col)
        ax.text(x_max * 1.03, cy - 0.20,
                "0 survive FDR",
                ha="left", va="center", fontsize=8.0,
                fontstyle="italic", fontfamily=FONT, color=DARK, alpha=0.75)

    # axis labelling
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.25, total_h + 0.70)
    ax.set_yticks([])
    ticks = [0.6, 1.0, 1.5, 2.0, 3.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=10, fontfamily=FONT, color=DARK)
    ax.set_xlabel("wartime / prewar rate ratio  (1pl  «we»)",
                  fontsize=10.5, fontfamily=FONT, color=DARK, labelpad=8)

    # interpretation labels at top of plot — clearer placement
    ax.text(0.78, total_h + 0.42, "← fewer ‹we› in wartime",
            ha="center", va="center", fontsize=8.5, fontstyle="italic",
            fontfamily=FONT, color=DOWN_C)
    ax.text(1.95, total_h + 0.42, "more ‹we› in wartime →",
            ha="center", va="center", fontsize=8.5, fontstyle="italic",
            fontfamily=FONT, color=UP_C)

    # panel title
    ax.text(0.50, 1.10,
            "A.  All specifications drift right — none of them clear FDR",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", fontfamily=FONT, color=NAV)
    ax.text(0.50, 1.04,
            "frequentist GLM (Poisson + NB) · author-clustered SE · wild-cluster bootstrap · 30 specs",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=8.5, fontstyle="italic", fontfamily=FONT, color=DARK)


def panel_posterior(ax, post_row):
    """
    Posterior density panel: pooled 1pl δ.
    Shade above RR=1.0 in amber, below in muted grey.
    """
    setup_axis(ax)
    mean = float(post_row["population_shift_mean_log_mu"])
    hdi_lo = float(post_row["population_shift_hdi95_low"])
    hdi_hi = float(post_row["population_shift_hdi95_high"])
    p_dir  = float(post_row["population_shift_p_direction_gt0"])

    # Normal approximation: SD from HDI95
    sd = (hdi_hi - hdi_lo) / (2 * 1.96)

    # Build density on log scale, plot on RR scale via x = exp(log_mu)
    x_log = np.linspace(-1.0, 1.2, 600)
    y_dens = stats.norm.pdf(x_log, loc=mean, scale=sd)
    x_rr = np.exp(x_log)

    ax.set_xscale("log")
    # Shading: above 1.0 = HL, below = mute
    above_mask = x_log > 0
    below_mask = ~above_mask
    ax.fill_between(x_rr[below_mask], 0, y_dens[below_mask],
                    color=DOWN_C, alpha=0.40, zorder=2)
    ax.fill_between(x_rr[above_mask], 0, y_dens[above_mask],
                    color=HL, alpha=0.75, zorder=3)

    # density outline
    ax.plot(x_rr, y_dens, color=NAV, lw=2.0, zorder=4)

    # reference line at RR = 1.0
    ax.axvline(1.0, color=RULE_C, lw=1.4, ls=(0, (4, 3)), zorder=2)

    # HDI95 horizontal bar at low y position
    y_hdi = -0.18
    ax.plot([np.exp(hdi_lo), np.exp(hdi_hi)], [y_hdi, y_hdi],
            color=NAV, lw=4.0, solid_capstyle="round", zorder=3)
    ax.plot([np.exp(mean)], [y_hdi], "o", color=HL,
            markersize=11, markeredgecolor=NAV, markeredgewidth=1.3, zorder=4)

    ax.text(np.exp(hdi_lo), y_hdi - 0.16, f"{np.exp(hdi_lo):.2f}",
            ha="center", va="top", fontsize=8.5, fontfamily=FONT, color=DARK)
    ax.text(np.exp(hdi_hi), y_hdi - 0.16, f"{np.exp(hdi_hi):.2f}",
            ha="center", va="top", fontsize=8.5, fontfamily=FONT, color=DARK)
    ax.text(np.exp(mean), y_hdi + 0.18, "posterior mean",
            ha="center", va="bottom", fontsize=8.5, fontstyle="italic",
            fontfamily=FONT, color=NAV)
    ax.text(1.0, -0.55, "← 95 % HDI →",
            ha="center", va="center", fontsize=9.0, fontstyle="italic",
            fontfamily=FONT, color=DARK)

    # Big number callouts for the headline statistics
    y_max = y_dens.max()
    ax.text(1.55, y_max * 0.84,
            f"P(δ > 0) = {p_dir*100:.0f}%",
            ha="left", va="center", fontsize=20, fontweight="bold",
            fontfamily=FONT, color=HL, zorder=5)
    ax.text(1.55, y_max * 0.62,
            "posterior mass above\n‘no shift’",
            ha="left", va="center", fontsize=9.5, fontstyle="italic",
            fontfamily=FONT, color=DARK, linespacing=1.3)

    # Annotation on the below-1.0 mass
    ax.text(0.78, y_max * 0.30,
            f"{(1-p_dir)*100:.0f}% below",
            ha="center", va="center", fontsize=10,
            fontweight="bold", fontfamily=FONT, color=DOWN_C, alpha=0.85)

    # axis
    ax.set_xlim(0.55, 3.6)
    ax.set_ylim(-0.75, y_max * 1.20)
    ticks = [0.6, 1.0, 1.5, 2.0, 3.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=10, fontfamily=FONT, color=DARK)
    ax.set_yticks([])
    ax.set_xlabel("wartime / prewar rate ratio  (1pl  «we»)",
                  fontsize=10.5, fontfamily=FONT, color=DARK, labelpad=8)

    # panel title
    ax.text(0.50, 1.10,
            "B.  Bayesian replication: same shape, same indeterminacy",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", fontfamily=FONT, color=NAV)
    ax.text(0.50, 1.04,
            "hierarchical NB with author random slopes · pooled 1pl posterior · normal approx. from HDI",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=8.5, fontstyle="italic", fontfamily=FONT, color=DARK)


def main():
    spec_df = pd.read_csv(ROOT / "outputs" / "02_modeling_specification_curve"
                          / "specification_curve_rr.csv")
    post_df = pd.read_csv(ROOT / "outputs" / "02_modeling_q2_hierarchical"
                          / "q2_population_shifts_by_cell.csv")
    pooled_1pl = post_df[(post_df["language_stratum"] == "pooled_Ukrainian_Russian")
                         & (post_df["cell"] == "1pl")].iloc[0]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_alpha(0.0)     # transparent — blends into the yellow board

    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1.15, 1.0],
        left=0.07, right=0.93, top=0.90, bottom=0.10,
        wspace=0.30,
    )

    ax_spec = fig.add_subplot(gs[0, 0]); ax_spec.set_facecolor("none")
    ax_post = fig.add_subplot(gs[0, 1]); ax_post.set_facecolor("none")

    panel_spec_curve(ax_spec, spec_df)
    panel_posterior(ax_post,  pooled_1pl)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_rq1_directional_null.{ext}",
                    dpi=300, bbox_inches="tight",
                    transparent=True, format=ext)
        print(f"saved → {OUT}/poster_rq1_directional_null.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
