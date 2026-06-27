"""
Poster figure: RQ3 three-row narrative panel — collocation shifts.

Three rows (UKR 1pl, UKR 2sg, RUS 1sg) + coda (cluster split).
Each row shows the BH-surviving head lemma(s) with pre→post counts
rendered as horizontal bars.

Usage : python src/poster_rq3_collocation_panel.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq3_collocation_panel.{pdf,png,svg}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette (shared with other poster scripts) ────────────────────
BG    = "#FFD500"
NAV   = "#1C3557"
BLU   = "#005BBB"
HL    = "#F4A200"
MOB_C = "#9A1750"
RULE  = "#9A7B00"
DARK  = "#2C2000"
WHITE = "#FFFFFF"
TEAL  = "#1f4d4a"

FONT  = "Times New Roman"
ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"

FIG_W, FIG_H = 8.0, 10.5


# ── data (hardcoded from BH-survivor table in the paper) ─────────────────
ROWS = [
    {
        "tag": "Ukrainian · 1pl  (we)",
        "entries": [
            {"lemma": "мусити", "gloss": "must",
             "pre": 0, "post": 13, "q": 0.007, "direction": "up"},
        ],
        "caption": (
            "Pre-war «we» was descriptive;\n"
            "wartime «we» is exhorted into duty."
        ),
    },
    {
        "tag": "Ukrainian · 2sg  (ти / thou)",
        "entries": [
            {"lemma": "хотіти", "gloss": "want",
             "pre": 7, "post": 28, "q": 0.013, "direction": "up"},
        ],
        "caption": (
            "The intimate addressee shifts from agent\n"
            "(think, write, seek) to beloved (want, love, wait)."
        ),
    },
    {
        "tag": "Russian · 1sg  (я / I)",
        "entries": [
            {"lemma": "називати", "gloss": "name/call",
             "pre": 0, "post": 30, "q": 0.00001, "direction": "up"},
            {"lemma": "любити", "gloss": "love",
             "pre": 27, "post": 1, "q": 0.082, "direction": "down"},
        ],
        "caption": (
            "The Russian lyric «I» stops loving\n"
            "and starts naming."
        ),
    },
]

CODA_LINES = [
    ("мусити", "concentrates in", "non-frontline", "poets."),
    ("Frontline poets govern", "боятися", "(fear),", "лишитися", "(remain)"),
]
CODA_TAGLINE = "Same pronoun.  Two registers.  Two wars."


def _rounded_rect(ax, x, y, w, h, color, radius=0.012, alpha=1.0, zorder=1):
    """Draw a rounded rectangle in axes coordinates."""
    rr = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transAxes, clip_on=False, zorder=zorder,
    )
    ax.add_patch(rr)
    return rr


def _bar(ax, x, y, length, height, color, alpha=1.0, zorder=3):
    """Draw a horizontal bar in axes coordinates."""
    rect = mpatches.FancyBboxPatch(
        (x, y), length, height,
        boxstyle="round,pad=0.003",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transAxes, clip_on=False, zorder=zorder,
    )
    ax.add_patch(rect)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── title ────────────────────────────────────────────────────────────
    ax.text(
        0.50, 0.97,
        "Lexical-Syntactic Anchoring",
        ha="center", va="top", fontsize=18, fontweight="bold",
        fontfamily=FONT, color=NAV, zorder=10,
    )
    ax.text(
        0.50, 0.938,
        "What verbs does each pronoun govern — and how did that change?",
        ha="center", va="top", fontsize=11, fontstyle="italic",
        fontfamily=FONT, color=DARK, zorder=10,
    )

    # ── layout geometry ──────────────────────────────────────────────────
    margin_x = 0.06
    panel_w = 1.0 - 2 * margin_x
    row_gap = 0.020
    top_start = 0.895

    # Row heights differ: single-entry rows vs the double-entry Row 3
    row_heights = [0.175, 0.175, 0.245]

    max_count = 30  # for bar scaling
    bar_region_x = 0.46  # left edge of the bar area (axes frac)
    bar_max_w = 0.42     # max bar width (axes frac)
    bar_h = 0.022
    entry_spacing = 0.065  # vertical gap between entries within a row

    y_cursor = top_start
    for i, row in enumerate(ROWS):
        rh = row_heights[i]
        y_top = y_cursor
        y_bot = y_top - rh
        y_cursor = y_bot - row_gap

        # row background card
        _rounded_rect(ax, margin_x, y_bot, panel_w, rh,
                       color=NAV, alpha=0.07, radius=0.008, zorder=1)

        # tag / cell label — pill badge
        tag_w = 0.38
        tag_h = 0.030
        tag_x = margin_x + 0.01
        tag_y = y_top - 0.042
        _rounded_rect(ax, tag_x, tag_y, tag_w, tag_h,
                       color=NAV, alpha=1.0, radius=0.006, zorder=4)
        ax.text(
            tag_x + tag_w / 2, tag_y + tag_h / 2,
            row["tag"],
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            fontfamily=FONT, color=WHITE, zorder=5,
        )

        # entries (bars + labels)
        entry_top = tag_y - 0.022

        for j, ent in enumerate(row["entries"]):
            ey = entry_top - j * entry_spacing

            # lemma + gloss on the left
            lemma_str = f"{ent['lemma']}  ({ent['gloss']})"
            entry_color = BLU if ent["direction"] == "up" else MOB_C
            ax.text(
                margin_x + 0.02, ey,
                lemma_str,
                ha="left", va="center", fontsize=11.5, fontweight="bold",
                fontfamily=FONT, color=entry_color, zorder=5,
            )

            # pre → post count, right-aligned before bar
            count_str = f"{ent['pre']}  →  {ent['post']}"
            ax.text(
                bar_region_x - 0.015, ey,
                count_str,
                ha="right", va="center", fontsize=12.5, fontweight="bold",
                fontfamily=FONT, color=DARK, zorder=5,
            )

            # horizontal bar
            if ent["direction"] == "up":
                bar_len = (ent["post"] / max_count) * bar_max_w
                bar_color = BLU
            else:
                bar_len = (ent["pre"] / max_count) * bar_max_w
                bar_color = MOB_C

            # ghost bar (the "from" value, faded)
            if ent["direction"] == "up" and ent["pre"] > 0:
                ghost_len = (ent["pre"] / max_count) * bar_max_w
                _bar(ax, bar_region_x, ey - bar_h / 2,
                     ghost_len, bar_h, bar_color, alpha=0.18, zorder=2)
            if ent["direction"] == "down" and ent["post"] > 0:
                ghost_len = (ent["post"] / max_count) * bar_max_w
                _bar(ax, bar_region_x, ey - bar_h / 2,
                     ghost_len, bar_h, bar_color, alpha=0.18, zorder=2)

            _bar(ax, bar_region_x, ey - bar_h / 2,
                 bar_len, bar_h, bar_color, alpha=0.85, zorder=3)

            # q-value label at bar end
            q_val = ent["q"]
            q_str = f"q = {q_val:.3f}" if q_val >= 0.001 else "q < 10⁻⁵"
            ax.text(
                bar_region_x + bar_len + 0.012, ey,
                q_str,
                ha="left", va="center", fontsize=8.5, fontstyle="italic",
                fontfamily=FONT, color=DARK, alpha=0.70, zorder=5,
            )

        # caption text — pinned above row bottom
        caption_y = y_bot + 0.012
        ax.text(
            margin_x + 0.02, caption_y,
            row["caption"],
            ha="left", va="bottom", fontsize=9.5, fontstyle="italic",
            fontfamily=FONT, color=DARK, alpha=0.85, zorder=5,
            linespacing=1.30,
        )

    # ── coda panel ───────────────────────────────────────────────────────
    coda_top = y_cursor + row_gap   # sits just below last row
    coda_h = 0.195
    coda_bot = coda_top - coda_h

    _rounded_rect(ax, margin_x, coda_bot, panel_w, coda_h,
                   color=NAV, alpha=1.0, radius=0.008, zorder=4)

    # coda header
    ax.text(
        0.50, coda_top - 0.018,
        "Who speaks the deontic «we must»?",
        ha="center", va="top", fontsize=12, fontweight="bold",
        fontfamily=FONT, color=WHITE, zorder=5,
    )

    # coda body — two clusters side by side
    col_left_x = margin_x + 0.04
    col_right_x = 0.52
    body_y = coda_top - 0.055

    # Left: non-frontline
    card_h = 0.075
    _rounded_rect(ax, col_left_x - 0.01, body_y - card_h,
                   0.42, card_h, color=BLU, alpha=0.25, radius=0.005, zorder=4)
    ax.text(
        col_left_x, body_y - 0.005,
        "NON-FRONTLINE  (751 tokens)",
        ha="left", va="top", fontsize=9.5, fontweight="bold",
        fontfamily=FONT, color=HL, zorder=6,
    )
    ax.text(
        col_left_x, body_y - 0.032,
        "мусити (must) · воювати (wage war)\n"
        "битися (fight) · сміятися (laugh)",
        ha="left", va="top", fontsize=9, fontfamily=FONT,
        color=WHITE, zorder=6, linespacing=1.3,
    )

    # Right: frontline
    _rounded_rect(ax, col_right_x - 0.01, body_y - card_h,
                   0.42, card_h, color=MOB_C, alpha=0.25, radius=0.005, zorder=4)
    ax.text(
        col_right_x, body_y - 0.005,
        "FRONTLINE  (198 tokens)",
        ha="left", va="top", fontsize=9.5, fontweight="bold",
        fontfamily=FONT, color="#FFB0C0", zorder=6,
    )
    ax.text(
        col_right_x, body_y - 0.032,
        "боятися (fear) · лишитися (remain)\n"
        "строїти (build) · писати (write)",
        ha="left", va="top", fontsize=9, fontfamily=FONT,
        color=WHITE, zorder=6, linespacing=1.3,
    )

    # tagline — pinned near coda bottom with breathing room
    ax.text(
        0.50, coda_bot + 0.020,
        CODA_TAGLINE,
        ha="center", va="bottom", fontsize=13, fontweight="bold",
        fontfamily=FONT, fontstyle="italic", color=HL, zorder=6,
    )

    # ── save ─────────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(
            OUT / f"poster_rq3_collocation_panel.{ext}",
            dpi=300, bbox_inches="tight", facecolor=BG, format=ext,
        )
        print(f"saved → {OUT}/poster_rq3_collocation_panel.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
