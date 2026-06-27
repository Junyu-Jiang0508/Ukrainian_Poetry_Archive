"""
Poster figure: RQ3 — what verbs each pronoun governs, before vs after 2022.

Three rows, one per (pronoun, distributionally-robust survivor):
  * 1pl «ми» — «мусити» (we must), emergent in wartime;
  * 2sg «ти» — «хотіти» (you want), shifting from agentive desire to longing;
  * 1sg «я»  — Russian «любить» (I love), a distributed wartime *decline*.

LEFT of each row: the quantitative collocate shift (bar pre→post, q, companion
verbs with glosses, semantic register). RIGHT: a *before 2022 vs after 2022* poem
pair under the same pronoun, so the verb/noun that the pronoun leads can be
compared directly. Pronouns are highlighted blue, governed verbs magenta.

The Russian 1sg «любить» row is the only Russian survivor shown: it is distributed
across 9 poets / 22 poems (pre 27 → post 1), unlike the other Russian-1sg
"survivors" (называть, благодарный) that are a single-poet (Solomko) artifact.

Usage : python src/poster_rq3_verbs_and_poems.py
Output: docs/Ukranian_Poetr_Report/figures/poster_rq3_verbs_and_poems.{pdf,png,svg}
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
HL     = "#F4A200"
MOB_C  = "#9A1750"
RU_C   = "#6A3D9A"
DARK   = "#2C2000"
GREY   = "#6F665B"
WHITE  = "#FFFFFF"
PRON_C = "#005BBB"
VERB_C = "#9A1750"
FONT   = "Times New Roman"

FIG_W, FIG_H = 16, 8.87   # three rows; per-row height preserved via band scaling
MAX_COUNT = 30.0
REF_H = 0.403             # reference band height the offsets were tuned at

ROWS = [
    {
        "cell": "1pl  «ми»  (we)",
        "lemma": "мусити", "gloss": "must",
        "pre": 0, "post": 13, "q": 0.007,
        "pre_co": [("могти", "can", 13, 1), ("бігти", "run", 6, 0), ("світити", "shine", 6, 1)],
        "post_co": [("думати", "think", 1, 6), ("вийти", "go out", 1, 5), ("лишитися", "remain", 0, 4)],
        "pre_lab": "mundane action", "post_lab": "duty · reflection",
        "poets": 5, "poems": 6,
        "before": {
            "text": "ми не можемо захистити кримчан,\nми не можемо їх захистити",
            "en": "we cannot protect them",
            "byline": "Halyna Kruk", "date": "Mar 2014",
            "prons": {"ми"}, "verbs": {"можемо"},
        },
        "after": {
            "text": "ми мусимо памʼятати.\nми мусимо не забувати.",
            "en": "we must remember, we must not forget",
            "byline": "Eva Tur", "date": "Jun 2024",
            "prons": {"ми"}, "verbs": {"мусимо"},
        },
    },
    {
        "cell": "2sg  «ти»  (thou)",
        "lemma": "хотіти", "gloss": "want",
        "pre": 7, "post": 28, "q": 0.013,
        "pre_co": [("говорити", "speak", 11, 2), ("повертатися", "return", 10, 0), ("тримати", "hold", 8, 2)],
        "post_co": [("чути", "hear", 4, 9), ("бачити", "see", 8, 12), ("забути", "forget", 0, 4)],
        "pre_lab": "agentive speech", "post_lab": "longing · memory",
        "poets": 10, "poems": 19,
        "before": {
            "text": "Я хотів, щоби все було так,\nяк хотіла ти.",
            "en": "I wanted it all to be as you wanted",
            "byline": "Serhiy Zhadan", "date": "Sep 2014",
            "prons": {"ти"}, "verbs": {"хотіла"},
        },
        "after": {
            "text": "куди ти хотіла потрапити\nпісля весни в якій ми втратили памʼять",
            "en": "where didst thou wish to go, after the spring we lost",
            "byline": "Anatoliy Dnistrovyi", "date": "Oct 2022",
            "prons": {"ти", "ми"}, "verbs": {"хотіла", "втратили"},
        },
    },
    {
        "cell": "1sg  «я»  (I)  ·  Russian",
        "lemma": "любить", "gloss": "love",
        "pre": 27, "post": 1, "q": 0.082,
        "pre_co": [("смотреть", "look", 15, 1), ("знать", "know", 18, 6), ("говорить", "speak", 11, 2)],
        "post_co": [("мочь", "can", 9, 13), ("идти", "go", 3, 7), ("думать", "think", 2, 3)],
        "pre_lab": "affection · perception", "post_lab": "endurance · motion",
        "poets": 9, "poems": 22,
        "before": {
            "text": "я очень люблю свой дом,\nсвой город, утопающий в розах",
            "en": "I love my home, my city drowning in roses",
            "byline": "Iya Kiva", "date": "Jul 2014",
            "prons": {"я"}, "verbs": {"люблю"},
        },
        "after": {
            "text": "я не могу обещать тебе этого,\nтак же, как не могу уехать",
            "en": "I cannot promise you this, nor can I leave",
            "byline": "Danik Zadorozhnyi", "date": "Apr 2022",
            "prons": {"я"}, "verbs": {"могу"},
        },
    },
]


def _rr(ax, x, y, w, h, color, alpha=1.0, r=0.008, zorder=1):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transAxes, clip_on=False, zorder=zorder))


def _co_str(items):
    return "   ".join(f"{lm} ({gl}) {pre}→{post}" for lm, gl, pre, post in items)


def _draw_poem(ax, fig, lines, x, y_top, prons, verbs, line_h, fontsize):
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    space = 0.0050
    y = y_top
    for line in lines.split("\n"):
        cx = x
        for word in line.split(" "):
            token = word.strip(".,;:!?»«").lower()
            if token in verbs:
                color, weight = VERB_C, "bold"
            elif token in prons:
                color, weight = PRON_C, "bold"
            else:
                color, weight = NAV, "normal"
            t = ax.text(cx, y, word, ha="left", va="top", fontsize=fontsize,
                        fontweight=weight, fontfamily=FONT, color=color, zorder=5)
            bb = t.get_window_extent(renderer=renderer)
            x0, _ = inv.transform((bb.x0, 0))
            x1, _ = inv.transform((bb.x1, 0))
            cx += (x1 - x0) + space
        y -= line_h
    return y


def _draw_minipoem(ax, fig, x, y, block, tag, tagcolor, accent, s):
    ax.text(x, y, tag, ha="left", va="top", fontsize=9.5, fontstyle="italic",
            fontweight="bold", fontfamily=FONT, color=tagcolor, zorder=5)
    y -= 0.030 * s
    y = _draw_poem(ax, fig, block["text"], x, y, block["prons"], block["verbs"],
                   line_h=0.040 * s, fontsize=12.5)
    y -= 0.004 * s
    ax.text(x, y, block["en"], ha="left", va="top", fontsize=9,
            fontstyle="italic", fontfamily=FONT, color=GREY, zorder=5)
    y -= 0.028 * s
    ax.text(x, y, f"— {block['byline']} · {block['date']}", ha="left", va="top",
            fontsize=9.5, fontweight="bold", fontfamily=FONT, color=accent, zorder=5)
    y -= 0.030 * s
    return y


def _draw_row(ax, fig, row, y_top, y_bot, accent):
    s = (y_top - y_bot) / REF_H
    left_x, div_x, right_x = 0.030, 0.520, 0.560

    _rr(ax, 0.012, y_bot, 0.976, y_top - y_bot, NAV, alpha=0.05, r=0.006, zorder=0)

    # ── LEFT: verb panel ────────────────────────────────────────────────
    cy = y_top - 0.030 * s
    _rr(ax, left_x, cy - 0.030 * s, 0.235, 0.038 * s, accent, r=0.005, zorder=3)
    ax.text(left_x + 0.118, cy - 0.011 * s, row["cell"], ha="center", va="center",
            fontsize=12.5, fontweight="bold", fontfamily=FONT, color=WHITE, zorder=5)

    cy -= 0.075 * s
    ax.text(left_x, cy, row["lemma"], ha="left", va="center",
            fontsize=20, fontweight="bold", fontfamily=FONT, color=accent, zorder=5)
    lemma_w = 0.008 * len(row["lemma"]) + 0.060
    ax.text(left_x + lemma_w, cy + 0.002 * s, f"({row['gloss']})", ha="left", va="center",
            fontsize=12, fontstyle="italic", fontfamily=FONT, color=GREY, zorder=5)

    bar_x0, bar_max = left_x + 0.205, 0.190
    bar_h = 0.030 * s
    by = cy - bar_h / 2
    ax.text(bar_x0 - 0.012, cy, str(row["pre"]), ha="right", va="center",
            fontsize=13, fontweight="bold", fontfamily=FONT, color=GREY, zorder=5)
    if row["pre"] > 0:
        _rr(ax, bar_x0, by, bar_max * row["pre"] / MAX_COUNT, bar_h,
            accent, alpha=0.22, r=0.003, zorder=2)
    post_len = bar_max * row["post"] / MAX_COUNT
    _rr(ax, bar_x0, by, post_len, bar_h, accent, alpha=0.90, r=0.003, zorder=3)
    ax.text(bar_x0 + post_len + 0.010, cy, str(row["post"]), ha="left", va="center",
            fontsize=13, fontweight="bold", fontfamily=FONT, color=accent, zorder=5)
    ax.text(bar_x0 + post_len + 0.040, cy, f"q = {row['q']:.3f}", ha="left", va="center",
            fontsize=9.5, fontstyle="italic", fontfamily=FONT, color=DARK, alpha=0.7, zorder=5)

    cy -= 0.066 * s
    ax.text(left_x, cy, "before 2022", ha="left", va="center",
            fontsize=9, fontstyle="italic", fontfamily=FONT, color=GREY, zorder=5)
    ax.text(left_x + 0.115, cy, _co_str(row["pre_co"]), ha="left", va="center",
            fontsize=10.5, fontfamily=FONT, color=DARK, alpha=0.60, zorder=5)
    cy -= 0.046 * s
    ax.text(left_x, cy, "after 2022", ha="left", va="center",
            fontsize=9, fontstyle="italic", fontfamily=FONT, color=accent, zorder=5)
    ax.text(left_x + 0.115, cy, _co_str(row["post_co"]), ha="left", va="center",
            fontsize=10.5, fontweight="bold", fontfamily=FONT, color=NAV, zorder=5)

    cy -= 0.050 * s
    ax.text(left_x, cy, f"{row['pre_lab']}   →   {row['post_lab']}",
            ha="left", va="center", fontsize=11, fontstyle="italic",
            fontfamily=FONT, color=accent, zorder=5)

    cy -= 0.050 * s
    ax.text(left_x, cy, "●" * row["poets"], ha="left", va="center",
            fontsize=9, fontfamily=FONT, color=accent, alpha=0.85, zorder=5)
    ax.text(left_x + 0.011 * row["poets"] + 0.008, cy,
            f"{row['poets']} poets · {row['poems']} poems",
            ha="left", va="center", fontsize=9.5, fontfamily=FONT, color=GREY, zorder=5)

    # ── divider ──────────────────────────────────────────────────────────
    ax.plot([div_x, div_x], [y_bot + 0.018 * s, y_top - 0.018 * s],
            color=NAV, lw=0.8, alpha=0.18, zorder=1,
            transform=ax.transAxes, clip_on=False)

    # ── RIGHT: before-vs-after poem pair (same pronoun) ───────────────────
    py = y_top - 0.032 * s
    py = _draw_minipoem(ax, fig, right_x, py, row["before"], "before 2022", GREY, GREY, s)
    py -= 0.004 * s
    ax.text(right_x, py, "↓  what «" + row["cell"].split("«")[1].split("»")[0] + "» comes to govern",
            ha="left", va="top", fontsize=8.5, fontstyle="italic",
            fontfamily=FONT, color=accent, alpha=0.8, zorder=5)
    py -= 0.034 * s
    _draw_minipoem(ax, fig, right_x, py, row["after"], "after 2022", accent, accent, s)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_alpha(0.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.canvas.draw()

    ax.text(0.988, 0.998, "pronoun", ha="right", va="top", fontsize=9.5,
            fontweight="bold", fontfamily=FONT, color=PRON_C, zorder=6)
    ax.text(0.918, 0.998, "verb", ha="right", va="top", fontsize=9.5,
            fontweight="bold", fontfamily=FONT, color=VERB_C, zorder=6)
    ax.text(0.864, 0.998, "highlighted:", ha="right", va="top", fontsize=9.5,
            fontstyle="italic", fontfamily=FONT, color=GREY, zorder=6)

    row_specs = [(0.965, 0.665, BLU), (0.645, 0.345, MOB_C), (0.325, 0.025, RU_C)]
    for row, (yt, yb, accent) in zip(ROWS, row_specs):
        _draw_row(ax, fig, row, yt, yb, accent)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"poster_rq3_verbs_and_poems.{ext}",
                    dpi=300, bbox_inches="tight",
                    facecolor="none", edgecolor="none",
                    transparent=True, format=ext)
        print(f"saved → {OUT}/poster_rq3_verbs_and_poems.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
