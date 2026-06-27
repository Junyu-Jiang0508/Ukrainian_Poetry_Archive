"""
Poster figure: three analytic layers.
Usage : python src/poster_three_layers.py
Output: docs/Ukranian_Poetr_Report/figures/poster_three_layers.{pdf,png,svg}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG   = "#FFD500"
NAV  = "#1C3557"
BLU  = "#005BBB"
HL   = "#F4A200"
TXT  = "#FFFFFF"
DARK = "#2C2000"

FONT = "Times New Roman"
OUT  = Path(__file__).parent.parent / "docs" / "Ukranian_Poetr_Report" / "figures"

FIG_W, FIG_H = 10.5, 4.0
CW, CH = 0.27, 0.80
CY     = 0.50
CXS    = [0.175, 0.500, 0.825]


def rbox(ax, cx, cy, w, h, color, r=0.018):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={r}",
        facecolor=color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3))


def card(ax, cx, color, layer, question, method, finding_lines):
    """
    finding_lines: list of (text, highlight_bool) per display line.
    Each element is one rendered line.
    """
    rbox(ax, cx, CY, CW, CH, color)

    # layer label
    ax.text(cx, CY + CH/2 - 0.065, layer.upper(),
            ha="center", va="center", transform=ax.transAxes,
            fontsize=7.5, fontfamily=FONT, fontweight="bold",
            color=TXT, alpha=0.65, zorder=4)

    # question
    ax.text(cx, CY + 0.13, question,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=13, fontfamily=FONT, fontweight="bold",
            color=TXT, zorder=4, linespacing=1.25)

    # thin rule
    ax.plot([cx - CW/2 + 0.022, cx + CW/2 - 0.022], [CY + 0.005]*2,
            color=TXT, lw=0.5, alpha=0.30,
            transform=ax.transAxes, zorder=4)

    # method
    ax.text(cx, CY - 0.080, method,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=7.8, fontfamily=FONT, fontstyle="italic",
            color=TXT, alpha=0.62, zorder=4, linespacing=1.35)

    # finding lines — each line is (str, highlight) or str
    n = len(finding_lines)
    line_gap = 0.075
    y_start = CY - 0.245 + (n - 1) * line_gap / 2

    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()

    for i, line in enumerate(finding_lines):
        fy = y_start - i * line_gap
        if isinstance(line, str):
            ax.text(cx, fy, line,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10.5, fontfamily=FONT, fontweight="bold",
                    color=TXT, zorder=5)
        else:
            # list of (segment, is_hl) — render inline left→right, centred
            full = "".join(s for s, _ in line)
            # estimate total width to find start x
            t_probe = ax.text(cx, fy, full,
                              ha="center", va="center",
                              transform=ax.transAxes,
                              fontsize=10.5, fontfamily=FONT,
                              color=(0, 0, 0, 0), zorder=1)
            fig.canvas.draw()
            bb_full = t_probe.get_window_extent(renderer=renderer)
            t_probe.remove()
            inv = ax.transAxes.inverted()
            x0_disp = bb_full.x0
            x = inv.transform((x0_disp, bb_full.y0))[0]

            for seg, is_hl in line:
                col = HL  if is_hl else TXT
                wt  = "bold"
                t = ax.text(x, fy, seg,
                            ha="left", va="center",
                            transform=ax.transAxes,
                            fontsize=10.5, fontfamily=FONT,
                            fontweight=wt, color=col, zorder=5)
                fig.canvas.draw()
                bb = t.get_window_extent(renderer=renderer)
                x  = inv.transform((bb.x1, bb.y0))[0]


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── Card 1 ────────────────────────────────────────────────────────────
    card(ax, CXS[0], NAV,
         layer="Corpus level",
         question="Does the rate\nshift?",
         method="Poisson & NB GLM · clustered SE\nwild-cluster bootstrap · BH-FDR",
         finding_lines=[
             "No cell survives FDR",
             "at corpus mean",
         ])

    # ── Card 2 ────────────────────────────────────────────────────────────
    card(ax, CXS[1], NAV,
         layer="Author level",
         question="Who shifts —\nand how much?",
         method="Bayesian NB · author random slopes\nBambi / PyMC · P(δ > 0) · BH q-direction",
         finding_lines=[
             "σ_δ ≈ 0.5  ·  frontline ↑1pl",
             "evacuees ↓1pl",
         ])

    # ── Card 3 ────────────────────────────────────────────────────────────
    card(ax, CXS[2], BLU,
         layer="Word level",
         question="What does\nthe shift mean?",
         method="Dependency parse (Stanza UD)\nG²  ·  BH within (language × cell)",
         finding_lines=[
             [("мусити", True), (" (must)  absent pre-2022", False)],
             "emerges only post-invasion",
         ])

    # ── arrows ────────────────────────────────────────────────────────────
    for i in range(2):
        x0 = CXS[i] + CW/2 + 0.005
        x1 = CXS[i+1] - CW/2 - 0.005
        ax.annotate("",
            xy=(x1, CY), xytext=(x0, CY),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.6,
                            mutation_scale=14, shrinkA=0, shrinkB=0))

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_three_layers.{ext}",
                    dpi=300, bbox_inches="tight",
                    facecolor=BG, format=ext)
        print(f"saved → {OUT}/poster_three_layers.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
