"""
Poster figure: RQ3 "Two Voices of We" — actual poem excerpts, ultra-compact.

48×36 poster, 1/6 panel ≈ 16×9 inches.
Usage : python src/poster_rq3_two_voices.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq3_two_voices.{pdf,png,svg}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"

NAV    = "#1C3557"
BLU    = "#005BBB"
GOLD   = "#FFD500"
MOB_C  = "#9A1750"
DARK   = "#2C2000"
WHITE  = "#FFFFFF"
FONT   = "Times New Roman"

FIG_W, FIG_H = 16, 7.0   # wide & short — no wasted space


# ── poem data ────────────────────────────────────────────────────────────
LEFT_POEMS = [
    {
        "author": "Eva Tur",
        "role": "ZSU since 2014 · paramedic · drone pilot",
        "ukr": (
            "ми мусимо пам\u02bcятати.\n"
            "ми мусимо не забувати.\n"
            "ми мусимо створити таку колективну пам\u02bcять\n"
            "про все, що відбувається з нами\n"
            "від віку віків і до нині"
        ),
        "en": (
            "We must remember.  We must not forget.\n"
            "We must create such a collective memory\n"
            "of all that befalleth us, from age to age"
        ),
    },
    {
        "author": "Ostap Slyvynsky",
        "role": "Lviv · VP PEN Ukraine · professor",
        "ukr": (
            "Ми мусимо розповісти цю війну.\n"
            "Інакше це зроблять за нас інші,\n"
            "не про те і не так, як потрібно."
        ),
        "en": (
            "We must tell this war.  Otherwise,\n"
            "others shall do it for us, not as it ought."
        ),
    },
]

RIGHT_POEMS = [
    {
        "author": "Yaryna Chornohuz",
        "role": "140th Marine Recon · combat medic · Shevchenko Prize 2024",
        "ukr": (
            "якщо ми з тобою лишимося живі\n"
            "я спробую вперше в житті\n"
            "посадити рослину у ще\n"
            "незаміноване поле"
        ),
        "en": (
            "If we remain alive, I shall try\n"
            "for the first time to plant a seed\n"
            "in yet unmined field"
        ),
    },
    {
        "author": "Maksym Kryvtsov",
        "role": "KIA 7 Jan 2024 · Hero of Ukraine (posthumous)",
        "ukr": (
            "дороги розмокли та стали\n"
            "не прохідними для техніки\n"
            "тому ми мусили йти\n"
            "іноді перечікували у посадці\n"
            "встеленій пожовклим листям"
        ),
        "en": (
            "The roads became impassable,\n"
            "therefore we must needs go.\n"
            "At times we waited in the thicket\n"
            "strewed with yellowed leaves"
        ),
    },
]


def _rr(ax, x, y, w, h, color, alpha=1.0, r=0.008, zorder=1):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transAxes, clip_on=False, zorder=zorder))


def _draw_card(ax, poem, x, y_top, w, accent):
    """Draw poem card, return y_bottom."""
    pad = 0.018
    cx = x + pad
    cy = y_top

    # accent bar
    _rr(ax, x + 0.002, 0, 0.005, 1.0, accent, alpha=0.0)  # placeholder

    # author
    ax.text(cx, cy, poem["author"], ha="left", va="top",
            fontsize=13.5, fontweight="bold", fontfamily=FONT, color=accent, zorder=5)
    cy -= 0.048

    # role
    ax.text(cx, cy, poem["role"], ha="left", va="top",
            fontsize=8.5, fontstyle="italic", fontfamily=FONT, color="#6F665B", zorder=5)
    cy -= 0.038

    # Ukrainian poem — larger font
    n_ukr = poem["ukr"].count("\n") + 1
    ax.text(cx, cy, poem["ukr"], ha="left", va="top",
            fontsize=12.5, fontfamily=FONT, color=NAV, zorder=5, linespacing=1.30)
    cy -= n_ukr * 0.052 + 0.004

    # English — compact
    n_en = poem["en"].count("\n") + 1
    ax.text(cx, cy, poem["en"], ha="left", va="top",
            fontsize=9.5, fontstyle="italic", fontfamily=FONT, color="#8A8A8A", zorder=5,
            linespacing=1.25)
    cy -= n_en * 0.042

    return cy


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_alpha(0.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── title row ────────────────────────────────────────────────────────
    ax.text(0.50, 0.99, "Two Voices of  «ми»  (we)",
            ha="center", va="top", fontsize=22, fontweight="bold",
            fontfamily=FONT, color=NAV, zorder=10)

    # ── column geometry ──────────────────────────────────────────────────
    col_w = 0.470
    left_x = 0.008
    right_x = 0.522
    mid_x = 0.506

    # ── column header pills ──────────────────────────────────────────────
    pill_y = 0.880
    pill_h = 0.045

    _rr(ax, left_x, pill_y, col_w, pill_h, BLU, zorder=4)
    ax.text(left_x + col_w * 0.55, pill_y + pill_h / 2,
            'THE  DEONTIC  «WE  MUST»',
            ha="center", va="center", fontsize=12, fontweight="bold",
            fontfamily=FONT, color=WHITE, zorder=5)
    # stat badge inside pill, right side
    ax.text(left_x + col_w - 0.010, pill_y + pill_h / 2,
            '0→13  q=.007',
            ha="right", va="center", fontsize=9, fontfamily=FONT,
            color=GOLD, fontweight="bold", zorder=5)

    _rr(ax, right_x, pill_y, col_w, pill_h, MOB_C, zorder=4)
    ax.text(right_x + col_w * 0.55, pill_y + pill_h / 2,
            'THE  EXISTENTIAL  «IF  WE  REMAIN»',
            ha="center", va="center", fontsize=12, fontweight="bold",
            fontfamily=FONT, color=WHITE, zorder=5)
    ax.text(right_x + col_w - 0.010, pill_y + pill_h / 2,
            '198 tokens',
            ha="right", va="center", fontsize=9, fontfamily=FONT,
            color="#FFB0C0", fontweight="bold", zorder=5)

    # ── poem cards ───────────────────────────────────────────────────────
    cards_top = pill_y - 0.012
    card_sep = 0.015

    # Left column
    y_l = cards_top
    for poem in LEFT_POEMS:
        card_top = y_l
        card_bot = _draw_card(ax, poem, left_x, y_l, col_w, BLU)
        # card background + accent bar
        ch = card_top - card_bot + 0.010
        _rr(ax, left_x, card_bot - 0.005, col_w, ch, BLU, alpha=0.05, zorder=0)
        _rr(ax, left_x + 0.002, card_bot - 0.002, 0.005, ch - 0.006,
            BLU, alpha=0.60, r=0.002, zorder=2)
        y_l = card_bot - card_sep

    # Right column
    y_r = cards_top
    for poem in RIGHT_POEMS:
        card_top = y_r
        card_bot = _draw_card(ax, poem, right_x, y_r, col_w, MOB_C)
        ch = card_top - card_bot + 0.010
        _rr(ax, right_x, card_bot - 0.005, col_w, ch, MOB_C, alpha=0.05, zorder=0)
        _rr(ax, right_x + 0.002, card_bot - 0.002, 0.005, ch - 0.006,
            MOB_C, alpha=0.60, r=0.002, zorder=2)
        y_r = card_bot - card_sep

    # ── vertical divider ─────────────────────────────────────────────────
    ax.plot([mid_x, mid_x], [min(y_l, y_r) + card_sep, pill_y],
            color=NAV, lw=0.8, alpha=0.15, zorder=0,
            transform=ax.transAxes, clip_on=False)

    # ── tagline bar at bottom ────────────────────────────────────────────
    tag_y = min(y_l, y_r) - 0.005
    _rr(ax, 0.10, tag_y, 0.80, 0.048, NAV, zorder=4)
    ax.text(0.50, tag_y + 0.024,
            "Same pronoun.   Two registers.   Two wars.",
            ha="center", va="center", fontsize=15, fontweight="bold",
            fontstyle="italic", fontfamily=FONT, color=GOLD, zorder=5)

    # ── save ─────────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_rq3_two_voices.{ext}",
                    dpi=300, bbox_inches="tight",
                    facecolor="none", edgecolor="none",
                    transparent=True, format=ext)
        print(f"saved → {OUT}/poster_rq3_two_voices.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
