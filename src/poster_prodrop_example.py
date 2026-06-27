"""
Poster figure: pro-drop recovery via Shakespearean translation.
Ihor Mitrov stanza — 4 lines, zero explicit Ukrainian pronouns,
4 × 1sg recovered in English.

Usage : python src/poster_prodrop_example.py
Output: docs/Ukranian_Poetr_Report/figures/poster_prodrop_example.{pdf,png}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

BG     = "#FFD500"
NAV    = "#1C3557"
BLU    = "#005BBB"
HL     = "#F4A200"
TXT    = "#FFFFFF"
DARK   = "#2C2000"
FONT   = "Times New Roman"
OUT    = Path(__file__).parent.parent / "docs" / "Ukranian_Poetr_Report" / "figures"

FIG_W, FIG_H = 9.0, 3.8


def rbox(ax, cx, cy, w, h, color, r=0.015):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={r}",
        facecolor=color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3))


def inline_text(ax, x, y, tokens, fsz=11.5):
    """
    Render tokens = [(str, highlight_bool), ...] left-to-right,
    using bounding-box measurement after each piece to advance x.
    Returns final x position.
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()

    for text, is_hl in tokens:
        color  = HL  if is_hl else TXT
        weight = "bold" if is_hl else "normal"
        t = ax.text(x, y, text,
                    ha="left", va="center",
                    transform=ax.transAxes,
                    fontsize=fsz, fontfamily=FONT,
                    fontweight=weight, color=color, zorder=5)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        # convert right edge from display coords to axes fraction
        inv = ax.transAxes.inverted()
        x = inv.transform((bb.x1, bb.y0))[0]
    return x


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── panels ────────────────────────────────────────────────────────────
    PW, PH  = 0.37, 0.74
    PY      = 0.50
    LCX, RCX = 0.22, 0.76
    LINE_H  = 0.135
    Y0      = PY + LINE_H * 1.5

    rbox(ax, LCX, PY, PW, PH, NAV)
    rbox(ax, RCX, PY, PW, PH, BLU)

    # ── panel subtitle tags ───────────────────────────────────────────────
    for cx, tag in (
        (LCX, "Ukrainian  ·  Ihor Mitrov  ·  2022"),
        (RCX, "Shakespearean English  ·  GPT-4o-mini"),
    ):
        ax.text(cx, PY + PH/2 + 0.055, tag,
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=8.5, fontfamily=FONT, fontstyle="italic",
                color=DARK, zorder=4)

    # ── stanza lines ──────────────────────────────────────────────────────
    FSZ = 11.5
    MARGIN = 0.032  # left margin inside panel

    ukr_lines = [
        [("[я] ",        True),  ("хотів",  True), (" нарешті знову",    False)],
        [("викупатися в рідному морі", False)],
        [("перш ніж нерозумну росію", False)],
        [("[я] ",        True),  ("приректи", True), (" на мечі й пожежі", False)],
    ]
    en_lines = [
        [("I",           True),  (" desired at last again",       False)],
        [("to bathe ",   False), ("me",    True), (" in ", False),
         ("my",          True),  (" native sea,",                 False)],
        [("ere ",        False), ("I",     True), (" do doom unwise Russia", False)],
        [("to the sword and to the burnings.", False)],
    ]

    for li, parts in enumerate(ukr_lines):
        inline_text(ax, LCX - PW/2 + MARGIN, Y0 - li * LINE_H, parts, FSZ)

    for li, parts in enumerate(en_lines):
        inline_text(ax, RCX - PW/2 + MARGIN, Y0 - li * LINE_H, parts, FSZ)

    # ── centre arrow ──────────────────────────────────────────────────────
    ax.annotate("",
        xy  =(RCX - PW/2 - 0.015, PY),
        xytext=(LCX + PW/2 + 0.015, PY),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.0,
                        mutation_scale=16, shrinkA=0, shrinkB=0))


    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_prodrop_example.{ext}",
                    dpi=300, bbox_inches="tight",
                    facecolor=BG, format=ext)
        print(f"saved → {OUT}/poster_prodrop_example.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
