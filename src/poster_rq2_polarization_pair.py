"""
Poster figure: RQ2 bidirectional polarization — Mitrov vs Falkovych.
Two panels side by side, per-year 1pl share with 2022 cutline.
Usage : python src/poster_rq2_polarization_pair.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq2_polarization_pair.{pdf,png,svg}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG    = "#FFD500"
NAV   = "#1C3557"
BLU   = "#005BBB"
HL    = "#F4A200"
MOB_C = "#9A1750"   # frontline crimson (Mitrov: mobilized)
RET_C = "#2E5266"   # in-Ukraine teal (Falkovych: not mobilized, evacuated within Ukraine)
RULE  = "#9A7B00"
DARK  = "#2C2000"
WHITE = "#FFFFFF"

FONT  = "Times New Roman"
ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"

FIG_W, FIG_H = 9.0, 4.8

POETS = {
    "Ihor Mitrov": {
        "color": MOB_C,
        "rr": 4.70, "p_dir": 1.00,
        "role": "95th Air Assault Brigade · frontline since Mar 2022",
        "label": "On the frontline",
    },
    "Hryhoryi Falkovych": {
        "color": RET_C,
        "rr": 0.43, "p_dir": 0.08,
        "role": "b. 1940 · evacuated Kyiv → Kolomyia post-2022",
        "label": "In Ukraine",
    },
}


def load_data():
    df = pd.read_csv(ROOT / "outputs" / "02_modeling_q2_hierarchical"
                     / "q2_poem_cell_counts_12.csv")
    d = df[df["author"].isin(POETS)].copy()
    quartet = d["1sg"] + d["1pl"] + d["2sg"] + d["2pl"]
    d = d.assign(share_1pl=d["1pl"] / quartet.replace(0, np.nan))
    d["year"] = pd.to_numeric(d["year_int"], errors="coerce")
    d = d.dropna(subset=["year"])
    d["year"] = d["year"].astype(int)
    agg = (d.groupby(["author", "year"])
            .agg(share_1pl=("share_1pl", "mean"),
                 n=("share_1pl", "size"))
            .reset_index())
    return agg


def draw_panel(ax, data, poet, info, show_ylabel):
    col  = info["color"]
    rr   = info["rr"]
    p    = info["p_dir"]

    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=DARK, length=3, width=0.8)

    years = data["year"].values
    share = data["share_1pl"].values
    n_arr = data["n"].values

    # bar plot — width proportional to poem count
    max_n = n_arr.max()
    bar_w = np.clip(n_arr / max_n * 0.65, 0.12, 0.72)

    for yr, sh, bw, n in zip(years, share, bar_w, n_arr):
        if np.isnan(sh):
            continue
        bar_col = col if yr >= 2022 else (col if yr >= 2019 else "#B0A78F")
        ax.bar(yr, sh, width=bw, color=bar_col,
               alpha=(0.90 if yr >= 2022 else 0.55),
               zorder=3)
        # tiny n label above each bar
        if n >= 3:
            ax.text(yr, sh + 0.012, str(n),
                    ha="center", va="bottom", fontsize=7,
                    fontfamily=FONT, color=DARK, alpha=0.70)

    # 2022 cutline
    ax.axvline(2021.5, color=RULE, lw=1.6, ls=(0, (5, 3)), zorder=4)
    ax.text(2021.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.85,
            "Feb 2022", ha="center", va="bottom",
            fontsize=8, fontstyle="italic",
            fontfamily=FONT, color=RULE)

    # pre/post mean lines
    pre_mask  = (years <  2022) & ~np.isnan(share)
    post_mask = (years >= 2022) & ~np.isnan(share)
    if pre_mask.any():
        pre_mean = share[pre_mask].mean()
        ax.hlines(pre_mean, years[pre_mask].min() - 0.4,
                  2021.3, color=col, lw=1.2, ls="--", alpha=0.55, zorder=4)
    if post_mask.any():
        post_mean = share[post_mask].mean()
        ax.hlines(post_mean, 2021.7, years[post_mask].max() + 0.4,
                  color=col, lw=1.6, ls="--", alpha=0.85, zorder=4)

    # y-axis
    ax.set_ylim(0, 0.90)
    ax.set_yticks([0, 0.25, 0.50, 0.75])
    ax.set_yticklabels(["0", "25%", "50%", "75%"],
                       fontsize=9, fontfamily=FONT, color=DARK)
    if show_ylabel:
        ax.set_ylabel("share of 1pl «we»\nwithin person-number quartet",
                      fontsize=9, fontfamily=FONT, color=DARK, labelpad=6)

    # x-axis
    all_years = list(range(2014, 2026))
    ax.set_xlim(2013.3, 2025.7)
    ax.set_xticks([2014, 2016, 2018, 2020, 2022, 2024])
    ax.set_xticklabels(["2014","2016","2018","2020","2022","2024"],
                       fontsize=9, fontfamily=FONT, color=DARK)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color=DARK, lw=0.6, alpha=0.3)

    # ── headline stats box ────────────────────────────────────────────────
    p_str = f"P(δ>0) = {p*100:.0f}%"
    rr_str = f"RR = {rr:.2f}"
    ax.text(0.97, 0.97, rr_str,
            ha="right", va="top", transform=ax.transAxes,
            fontsize=18, fontweight="bold", fontfamily=FONT,
            color=col, zorder=5)
    ax.text(0.97, 0.78, p_str,
            ha="right", va="top", transform=ax.transAxes,
            fontsize=11, fontfamily=FONT, color=col, zorder=5)

    # ── title block ───────────────────────────────────────────────────────
    label_bg = col
    rect = mpatches.FancyBboxPatch(
        (0.0, 1.02), 1.0, 0.18,
        boxstyle="round,pad=0.01",
        facecolor=label_bg, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=5)
    ax.add_patch(rect)
    ax.text(0.50, 1.13, poet,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", fontfamily=FONT,
            color=WHITE, zorder=6)

    # role tag below panel
    ax.text(0.50, -0.16, info["role"],
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8.5, fontfamily=FONT, fontstyle="italic",
            color=DARK, zorder=5)

    # label chip (Mobilized / Withdrawal)
    chip_col = col
    chip = mpatches.FancyBboxPatch(
        (0.0, -0.28), 1.0, 0.10,
        boxstyle="round,pad=0.008",
        facecolor=chip_col, edgecolor="none", alpha=0.18,
        transform=ax.transAxes, clip_on=False, zorder=4)
    ax.add_patch(chip)
    ax.text(0.50, -0.23, info["label"].upper(),
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", fontfamily=FONT,
            color=col, zorder=5, alpha=0.9)


def main():
    data = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.22,
                        wspace=0.28)

    for ax, (poet, info) in zip(axes, POETS.items()):
        d = data[data["author"] == poet].copy()
        draw_panel(ax, d, poet, info, show_ylabel=(ax is axes[0]))

    # cutline label needs y-lim set — restate after draw
    for ax in axes:
        ax.texts[0].set_y(0.87)   # "Feb 2022" label

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_rq2_polarization_pair.{ext}",
                    dpi=300, bbox_inches="tight",
                    facecolor=BG, format=ext)
        print(f"saved → {OUT}/poster_rq2_polarization_pair.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
