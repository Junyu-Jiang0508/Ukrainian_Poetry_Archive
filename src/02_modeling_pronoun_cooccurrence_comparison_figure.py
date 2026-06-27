"""Render the period comparison figure for content-filtered ego neighbours.

Produces a side-by-side bar chart per (language, focal) showing the top-N
content-word neighbours' log-Dice in 2014–2021 vs post-2022. Words appearing
only in one period are shown at log_dice = 0 in the missing period.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = (
    ROOT / "outputs" / "02_modeling_pronoun_cooccurrence_content_filtered" / "ego_edges_content_lemmas.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_cooccurrence_content_filtered"

FOCALS_BY_LANG = {
    "Ukrainian": ("ми", "я", "ти", "ви"),
    "Russian": ("мы", "я", "ты", "вы"),
}

# Bilingual gloss for the most-likely-to-appear lemmas; we leave Ukrainian /
# Russian forms as-is for any lemma not in this table.
GLOSS_TABLE = {
    # Ukrainian
    "війна": "war",
    "око": "eye",
    "земля": "land/earth",
    "життя": "life",
    "бути": "be",
    "час": "time",
    "людина": "person",
    "слово": "word",
    "серце": "heart",
    "дім": "home",
    "хата": "house",
    "ніч": "night",
    "день": "day",
    "ранок": "morning",
    "син": "son",
    "донька": "daughter",
    "син": "son",
    "брат": "brother",
    "мати": "mother",
    "батько": "father",
    "країна": "country",
    "україна": "Ukraine",
    "вода": "water",
    "хліб": "bread",
    "море": "sea",
    "дорога": "road",
    "пам'ять": "memory",
    "смерть": "death",
    "сила": "strength",
    "дорога": "road",
    "сон": "dream",
    "тиша": "silence",
    "крик": "scream",
    # Russian
    "война": "war",
    "глаз": "eye",
    "земля": "land/earth",
    "жизнь": "life",
    "быть": "be",
    "время": "time",
    "ребенок": "child",
    "море": "sea",
    "любовь": "love",
    "говорить": "speak",
    "знать": "know",
    "нет": "is-not",
    "слово": "word",
    "значить": "mean",
    "вина": "guilt",
    "вра": "enemy(trunc)",
    "общий": "shared",
    "дело": "deed/business",
    "раз": "once/time",
    "лисичка": "little-fox",
    "давать": "give",
    "казаться": "seem",
}


def _gloss(lemma: str) -> str:
    g = GLOSS_TABLE.get(lemma.lower())
    return f"{lemma} ({g})" if g else lemma


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    df = pd.read_csv(args.input)
    log.info("Loaded %d content-lemma rows", len(df))

    for lang, focals in FOCALS_BY_LANG.items():
        fig, axes = plt.subplots(
            len(focals), 1, figsize=(9, 2.4 * len(focals)), constrained_layout=True
        )
        for ax, focal in zip(axes, focals):
            sub = df.loc[df["language"].eq(lang) & df["focal"].eq(focal)].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            # Union of top-k lemmas across the two periods.
            top_lemmas: set[str] = set()
            for period in ("2014_2021", "post_2022"):
                top_lemmas.update(
                    sub.loc[sub["period"].eq(period)]
                    .nlargest(args.top_k, "log_dice")["neighbour_lemma"]
                    .tolist()
                )
            if not top_lemmas:
                ax.set_visible(False)
                continue
            # Build a (lemma, period) → log_dice frame; fill missing with 0.
            keep = sub.loc[sub["neighbour_lemma"].isin(top_lemmas)].copy()
            wide = keep.pivot_table(
                index="neighbour_lemma",
                columns="period",
                values="log_dice",
                aggfunc="max",
            ).fillna(0.0)
            for p in ("2014_2021", "post_2022"):
                if p not in wide.columns:
                    wide[p] = 0.0
            wide = wide[["2014_2021", "post_2022"]]
            wide["max"] = wide.max(axis=1)
            wide = wide.sort_values("max", ascending=True).drop(columns="max")
            ypos = np.arange(len(wide))
            bw = 0.4
            ax.barh(
                ypos - bw / 2,
                wide["2014_2021"].values,
                height=bw,
                color="#9e9e9e",
                label="2014–2021",
            )
            ax.barh(
                ypos + bw / 2,
                wide["post_2022"].values,
                height=bw,
                color="#7f2a32",
                label="post-2022",
            )
            ax.set_yticks(ypos)
            ax.set_yticklabels([_gloss(l) for l in wide.index], fontsize=8)
            ax.set_title(f"{lang} · focal = {focal}", fontsize=10, loc="left")
            ax.tick_params(axis="x", labelsize=8)
            ax.axvline(0, color="black", lw=0.4)
            ax.set_xlabel("log-Dice (content-word neighbours)")
            if ax is axes[0]:
                ax.legend(fontsize=8, loc="lower right")

        fig.suptitle(
            f"{lang}: top content-word neighbours of focal pronouns, 2014–2021 vs post-2022",
            fontsize=11,
        )
        out_pdf = out_dir / f"period_comparison_{lang}.pdf"
        out_png = out_dir / f"period_comparison_{lang}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        log.info("Wrote %s", out_pdf)


if __name__ == "__main__":
    main()
