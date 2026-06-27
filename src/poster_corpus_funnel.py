"""
Poster figure: corpus construction funnel — compact square version.
Usage: python src/poster_corpus_funnel.py
Output: docs/Ukranian_Poetr_Report/figures/poster_corpus_funnel.{pdf,png}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG       = "#FFD500"
BOX_MAIN = "#1C3557"
BOX_BLUE = "#005BBB"
TXT      = "#FFFFFF"
ARROW_C  = "#2C2C2C"
EXCL_C   = "#6B3A00"   # dark amber — exclusion text colour
ITALIC_C = "#4A3000"
FONT     = "Times New Roman"

OUT_DIR = Path(__file__).parent.parent / "docs" / "Ukranian_Poetr_Report" / "figures"

FIG_W = FIG_H = 5.6   # square

CX = 0.50
BW, BH   = 0.68, 0.14   # main box
SBW, SBH = 0.30, 0.14   # stratum sub-box
LEFT_CX  = CX - 0.18
RIGHT_CX = CX + 0.18

Y_TOP    = 0.82
Y_STRAT  = 0.47
Y_FINAL  = 0.14


def rbox(ax, cx, cy, w, h, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.014",
        facecolor=color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3))


def label(ax, cx, cy, title, sub=None, color=TXT, tsz=11, ssz=9.2):
    dy = 0.010 if sub else 0
    ax.text(cx, cy + dy, title, ha="center", va="center",
            transform=ax.transAxes, fontsize=tsz, fontweight="bold",
            fontfamily=FONT, color=color, zorder=4)
    if sub:
        ax.text(cx, cy - 0.030, sub, ha="center", va="center",
                transform=ax.transAxes, fontsize=ssz,
                fontfamily=FONT, color=color, alpha=0.92,
                zorder=4, linespacing=1.3)


def arrow(ax, x, y0, y1, lw=1.5):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                lw=lw, mutation_scale=13,
                                shrinkA=0, shrinkB=0), zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    # title
    ax.text(CX, 0.97, "Corpus Construction", ha="center", va="top",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            fontfamily=FONT, color=BOX_MAIN)

    # ── top box ───────────────────────────────────────────────────────────
    rbox(ax, CX, Y_TOP, BW, BH, BOX_MAIN)
    label(ax, CX, Y_TOP,
          "Contemporary Ukrainian Poetry Archive",
          "3,196 poems · 131 authors · 2013–2025")

    # ── long arrow top → stratum ──────────────────────────────────────────
    y_arr_top = Y_TOP - BH/2 - 0.01
    y_arr_bot = Y_STRAT + SBH/2 + 0.01
    arrow(ax, CX, y_arr_top, y_arr_bot)

    # exclusion labels to the right of the arrow
    excl_x = CX + 0.06
    mid_y   = (y_arr_top + y_arr_bot) / 2
    gap     = 0.050
    excl_lines = [
        "− translations & reposts",
        "− Qirimli & other non-Slavic",
        "roster filter  ≥ 5 poems / period",
    ]
    for i, line in enumerate(excl_lines):
        y = mid_y + (1 - i) * gap
        style = "italic" if i == 2 else "normal"
        col   = ITALIC_C if i == 2 else EXCL_C
        ax.text(excl_x, y, line, ha="left", va="center",
                transform=ax.transAxes, fontsize=8.5,
                fontfamily=FONT, fontstyle=style, color=col, zorder=4)

    # tick marks linking exclusion text to arrow
    for i in range(len(excl_lines)):
        y = mid_y + (1 - i) * gap
        ax.plot([CX, excl_x - 0.01], [y, y],
                color=EXCL_C, lw=0.6, ls="--",
                transform=ax.transAxes, zorder=2, alpha=0.6)

    # ── split to stratum boxes ────────────────────────────────────────────
    for tcx in (LEFT_CX, RIGHT_CX):
        ax.annotate("", xy=(tcx, Y_STRAT + SBH/2 + 0.01),
                    xytext=(CX, y_arr_bot - 0.01),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                    lw=1.3, mutation_scale=11,
                                    shrinkA=0, shrinkB=0), zorder=2)

    rbox(ax, LEFT_CX,  Y_STRAT, SBW, SBH, BOX_BLUE)
    label(ax, LEFT_CX,  Y_STRAT, "Ukrainian", "1,383 poems · 31 authors")

    rbox(ax, RIGHT_CX, Y_STRAT, SBW, SBH, BOX_BLUE)
    label(ax, RIGHT_CX, Y_STRAT, "Russian",   "439 poems · 14 authors")

    # ── merge → final ─────────────────────────────────────────────────────
    for scx in (LEFT_CX, RIGHT_CX):
        ax.annotate("", xy=(CX, Y_FINAL + BH/2 + 0.01),
                    xytext=(scx, Y_STRAT - SBH/2 - 0.01),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                    lw=1.3, mutation_scale=11,
                                    shrinkA=0, shrinkB=0), zorder=2)

    ax.text(CX, (Y_STRAT - SBH/2 + Y_FINAL + BH/2) / 2,
            "stanza explosion · LLM annotation",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8.2, fontfamily=FONT, fontstyle="italic",
            color=ITALIC_C, zorder=4)

    rbox(ax, CX, Y_FINAL, BW, BH, BOX_BLUE)
    label(ax, CX, Y_FINAL,
          "Inferential Stanza Dataset",
          "22,635 stanza records")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"poster_corpus_funnel.{ext}",
                    dpi=300, bbox_inches="tight",
                    facecolor=BG, format=ext)
        print(f"saved → {OUT_DIR}/poster_corpus_funnel.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
