"""Diverging-bar collocation figures (replaces the ΔPMI ``barbell`` panels).

Each (language, cell) is drawn as a *single* signed-G² axis: bars to the left
are pre-2022 affinity, bars to the right are post-2022 affinity, sorted
monotonically so positive and negative never interleave. Two motivations:

1. **Strict pos/neg separation, no crossing.** The previous two-panel barbell
   took a "top-15" list per side, which leaked opposite-sign bars across the
   panel boundary. A single sorted diverging axis cannot do that.

2. **The BH survivors actually appear.** The barbell ranked by ΔPMI, which is
   undefined for a head lemma that emerges from a zero pre-war count
   (PMI(pre) = log 0). Exactly the headline survivors do this (мусити 0→13,
   называть 0→30, благодарный 0→13, казаться 0→11), so they were silently
   dropped from the figure even though they dominate the confirmatory table.
   Signed G² is defined for every row, so the figure and the survivor table
   finally agree. Survivors (q_BH < 0.10) are bold-labelled and marked ``*``.

Outputs (written directly into the paper figures dir):
  fig_narr_9_collocations_ukrainian.{pdf,png}   1pl panel          (body)
  fig_narr_11_collocations_russian.{pdf,png}    1sg panel          (appendix)
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

DEFAULT_INPUT = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "differential_collocations.csv"
DEFAULT_OUTDIR = ROOT / "docs" / "Ukranian_Poetr_Report" / "figures"

MIN_COOCC_TOTAL = 5
TOP_N = 14
DEPRELS = ("nsubj", "obj", "obl", "nmod")

# English glosses for the head lemmas (shown in parentheses instead of the
# dependency relation). Note: ``спишати`` is a Stanza mis-lemmatisation of the
# 2sg forms спиш / спишь ("ти спиш" = you sleep), so it is glossed "sleep".
GLOSS = {
    # Ukrainian
    "могти": "can", "залишитися": "remain", "бігти": "run", "говорити": "speak",
    "світити": "shine", "прокинутися": "wake up", "брати": "take", "кожний": "every",
    "втратити": "lose", "треба": "need", "сміятися": "laugh", "вийти": "go out",
    "думати": "think", "мусити": "must", "повертатися": "return", "відкрити": "open",
    "вигадати": "invent", "розуміти": "understand", "бути": "be", "чекати": "wait",
    "ми": "we", "чути": "hear", "спишати": "sleep", "хотіти": "want",
    "носити": "carry", "ховатися": "hide", "ставати": "become", "дивитися": "look",
    "бачити": "see", "знати": "know", "прийти": "come", "любити": "love",
    "взяти": "take", "шукати": "seek", "написати": "write", "казати": "say", "іти": "go",
    # Russian
    "любить": "love", "смотреть": "look", "жить": "live", "ощущать": "feel",
    "кто": "who", "умереть": "die", "простить": "forgive", "писать": "write",
    "идти": "go", "мочь": "can", "делать": "do", "казаться": "seem",
    "благодарный": "grateful", "называть": "name", "верить": "believe",
    "немати": "lack", "лежать": "lie", "видеть": "see", "хотеть": "want",
}
Q_STAR = 0.10
COLOR_POST = "#1F4E5F"  # teal: post-2022 affinity
COLOR_PRE = "#7F2A32"   # burgundy: pre-2022 affinity


def _prep(df: pd.DataFrame, language: str, cell: str) -> pd.DataFrame:
    sub = df[(df.language == language) & (df.cell == cell) & (df.deprel.isin(DEPRELS))].copy()
    sub["total_coocc"] = sub["cooc_2014_2021"] + sub["cooc_post_2022"]
    sub = sub[sub["total_coocc"] >= MIN_COOCC_TOTAL].copy()
    if sub.empty:
        return sub
    # Direction from the per-period rate (defined even when one period count is 0).
    rate_pre = sub["cooc_2014_2021"] / sub["N_p1"].replace(0, np.nan)
    rate_post = sub["cooc_post_2022"] / sub["N_p2"].replace(0, np.nan)
    sign = np.where(rate_post.fillna(0).values >= rate_pre.fillna(0).values, 1.0, -1.0)
    sub["signed_g2"] = sign * sub["g2_period_contrast"].astype(float)
    sub["q"] = sub["q_value_bh"].astype(float)
    # Keep the strongest movers in either direction, then sort monotonically.
    sub = sub.reindex(sub["signed_g2"].abs().sort_values(ascending=False).index).head(TOP_N)
    sub = sub.sort_values("signed_g2").reset_index(drop=True)
    return sub


def _panel(ax, sub: pd.DataFrame, title: str) -> None:
    span = float(sub["signed_g2"].abs().max()) if not sub.empty else 1.0
    for i, r in sub.iterrows():
        val = float(r["signed_g2"])
        color = COLOR_POST if val >= 0 else COLOR_PRE
        q = r["q"] if pd.notna(r["q"]) else 1.0
        alpha = 0.30 + 0.70 * (1.0 - min(max(float(q), 0.0), 1.0))
        ax.barh(i, val, color=color, alpha=alpha, edgecolor="none")
        # raw count transition at the bar tip (0→N makes emergence-from-zero visible)
        pre, post = int(r["cooc_2014_2021"]), int(r["cooc_post_2022"])
        off = 0.012 * span
        ax.text(val + (off if val >= 0 else -off), i, f"{pre}→{post}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=6.3, color="#6b6b6b")
    labels = []
    for _, r in sub.iterrows():
        surv = pd.notna(r["q"]) and float(r["q"]) < Q_STAR
        lemma = str(r["head_lemma"])
        gloss = GLOSS.get(lemma)
        base = f"{lemma} ({gloss})" if gloss else lemma
        labels.append(("* " if surv else "") + base)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=8.5)
    for tl, (_, r) in zip(ax.get_yticklabels(), sub.iterrows()):
        if pd.notna(r["q"]) and float(r["q"]) < Q_STAR:
            tl.set_fontweight("bold")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlim(-1.18 * span, 1.18 * span)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


_XLABEL = ("signed $G^2$ (log-likelihood ratio):  "
           "negative = pre-2022 affinity,  positive = post-2022 affinity")


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300, facecolor="white")
        log.info("wrote %s", out_dir / f"{stem}.{ext}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300,
                         "axes.facecolor": "white", "figure.facecolor": "white"})
    df = pd.read_csv(args.input)
    out_dir = args.outdir.resolve()

    # --- Ukrainian: 1pl (body figure) ---
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _panel(ax, _prep(df, "Ukrainian", "1pl"), "1pl  «we»")
    ax.set_xlabel(_XLABEL, fontsize=8.5)
    fig.suptitle("Period-differential head-lemma collocates (Ukrainian 1pl)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "fig_narr_9_collocations_ukrainian")
    plt.close(fig)

    # --- Russian: 1sg (appendix figure; the pseudo-replication artifact) ---
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _panel(ax, _prep(df, "Russian", "1sg"), "Russian 1sg  «я»")
    ax.set_xlabel(_XLABEL, fontsize=8.5)
    fig.suptitle("Period-differential head-lemma collocates (Russian 1sg)", fontsize=12)
    fig.text(0.5, -0.03,
             "Bold + ``*`` = survives BH at $q<0.10$; opacity $\\propto 1-q$; bar-tip counts are pre→post. "
             "The post-2022 survivors (называть, благодарный, казаться) all emerge from zero "
             "and a token-level audit traces them to a single poet (pseudo-replication).",
             ha="center", fontsize=7.6, fontstyle="italic", color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _save(fig, out_dir, "fig_narr_11_collocations_russian")
    plt.close(fig)


if __name__ == "__main__":
    main()
