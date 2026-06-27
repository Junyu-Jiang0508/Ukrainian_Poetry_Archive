"""Summarise the dependency-parsed pronoun collocation outputs (P1-A/B follow-up).

Reads ``outputs/02_modeling_pronoun_collocations/differential_collocations.csv``
and produces three things:

1. A wide publication table per ``(language, cell)`` showing the top-N positive-
   ΔPMI head lemmas (post-2022 affinity) and the top-N negative-ΔPMI lemmas
   (pre-2022 affinity), restricted to deprel = ``nsubj`` (the cell's
   *subject-of* collocates, i.e. the verbs the pronoun governs) and
   deprel = ``obl``/``nmod`` (oblique modifiers, the entities the
   pronoun is *placed alongside*).

2. A "barbell" figure for the 1pl cell in each language stratum: left side =
   pre-2022-affinity head lemmas (ranked by |ΔPMI|), right side = post-2022-
   affinity head lemmas. Bar length encodes |ΔPMI|; bar opacity encodes
   BH-corrected G² q-value (darker = more confident).

3. A short text report (.md) noting effect sizes, BH-survivor counts, and
   sanity checks (stopword pollution, low-count survivors).

Run after ``02_modeling_pronoun_collocations.py``.
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

DEFAULT_INPUT_DIR = ROOT / "outputs" / "02_modeling_pronoun_collocations"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_collocation_summary"

PRIMARY_CELLS = ("1sg", "1pl", "2sg", "2pl_vy_true_plural")
PRIMARY_DEPRELS = ("nsubj", "obj", "obl", "nmod")
TOP_N = 15
MIN_COOCC_TOTAL = 5  # exclude head lemmas with < 5 total co-occurrences across periods


def _load(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "differential_collocations.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run 02_modeling_pronoun_collocations.py first; missing {path}"
        )
    df = pd.read_csv(path)
    df["total_coocc"] = df["cooc_2014_2021"] + df["cooc_post_2022"]
    df = df.loc[df["total_coocc"] >= MIN_COOCC_TOTAL].copy()
    return df


def _top_movers(
    df: pd.DataFrame, language: str, cell: str, deprel: str, n: int = TOP_N
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df.loc[
        df["language"].eq(language) & df["cell"].eq(cell) & df["deprel"].eq(deprel)
    ].copy()
    sub = sub.dropna(subset=["delta_pmi"])
    pos = sub.sort_values("delta_pmi", ascending=False).head(n).copy()
    neg = sub.sort_values("delta_pmi", ascending=True).head(n).copy()
    return pos, neg


def _wide_table(
    df: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lang in sorted(df["language"].unique()):
        for cell in PRIMARY_CELLS:
            for deprel in PRIMARY_DEPRELS:
                pos, neg = _top_movers(df, lang, cell, deprel, n=TOP_N)
                for rank, (_, r) in enumerate(pos.iterrows(), start=1):
                    rows.append(
                        {
                            "language": lang,
                            "cell": cell,
                            "deprel": deprel,
                            "direction": "post_2022_affinity",
                            "rank": rank,
                            "head_lemma": r["head_lemma"],
                            "delta_pmi": float(r["delta_pmi"]),
                            "cooc_p1": int(r["cooc_2014_2021"]),
                            "cooc_p2": int(r["cooc_post_2022"]),
                            "g2_period_contrast": float(r.get("g2_period_contrast", np.nan)),
                            "q_value_bh": float(r.get("q_value_bh", np.nan)),
                        }
                    )
                for rank, (_, r) in enumerate(neg.iterrows(), start=1):
                    rows.append(
                        {
                            "language": lang,
                            "cell": cell,
                            "deprel": deprel,
                            "direction": "pre_2022_affinity",
                            "rank": rank,
                            "head_lemma": r["head_lemma"],
                            "delta_pmi": float(r["delta_pmi"]),
                            "cooc_p1": int(r["cooc_2014_2021"]),
                            "cooc_p2": int(r["cooc_post_2022"]),
                            "g2_period_contrast": float(r.get("g2_period_contrast", np.nan)),
                            "q_value_bh": float(r.get("q_value_bh", np.nan)),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "top_movers_per_cell_deprel.csv", index=False)
    return out


def _barbell_figure(df: pd.DataFrame, language: str, cell: str, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pos_all: list[pd.DataFrame] = []
    neg_all: list[pd.DataFrame] = []
    for deprel in ("nsubj", "obj", "obl", "nmod"):
        pos, neg = _top_movers(df, language, cell, deprel, n=8)
        pos = pos.assign(deprel=deprel)
        neg = neg.assign(deprel=deprel)
        pos_all.append(pos)
        neg_all.append(neg)
    pos = pd.concat(pos_all, ignore_index=True)
    neg = pd.concat(neg_all, ignore_index=True)
    if pos.empty and neg.empty:
        return

    pos = pos.sort_values("delta_pmi", ascending=False).head(15)
    neg = neg.sort_values("delta_pmi", ascending=True).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=False)
    for ax, side, color in (
        (axes[0], neg, "#7f2a32"),  # burgundy: pre-2022 affinity
        (axes[1], pos, "#1f4d4a"),  # teal: post-2022 affinity
    ):
        if side.empty:
            ax.set_visible(False)
            continue
        side = side.iloc[::-1].reset_index(drop=True)
        q = side["q_value_bh"].fillna(1.0).clip(0, 1)
        alpha = 0.25 + 0.75 * (1.0 - q.values)
        labels = side["head_lemma"].astype(str) + " (" + side["deprel"].astype(str) + ")"
        bars = ax.barh(
            np.arange(len(side)),
            side["delta_pmi"].astype(float),
            color=color,
            alpha=0.85,
            edgecolor="none",
        )
        for b, a in zip(bars, alpha):
            b.set_alpha(float(a))
        ax.set_yticks(np.arange(len(side)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", lw=0.5)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_title("pre-2022 affinity (ΔPMI < 0)", fontsize=11)
    axes[1].set_title("post-2022 affinity (ΔPMI > 0)", fontsize=11)
    axes[0].set_xlabel("ΔPMI = PMI(post-2022) − PMI(2014–2021)")
    axes[1].set_xlabel("ΔPMI = PMI(post-2022) − PMI(2014–2021)")
    fig.suptitle(
        f"Period-differential head-lemma collocates of {cell} ({language})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"barbell_{language}_{cell}.pdf")
    fig.savefig(out_dir / f"barbell_{language}_{cell}.png", dpi=200)
    plt.close(fig)


def _bh_survivors(df: pd.DataFrame) -> pd.DataFrame:
    """Per (language, cell), count head-lemmas surviving q < 0.05 / 0.10."""
    out = []
    for (lang, cell), grp in df.groupby(["language", "cell"]):
        out.append(
            {
                "language": lang,
                "cell": cell,
                "n_head_lemmas": int(len(grp)),
                "q_lt_005": int((grp["q_value_bh"] < 0.05).sum()),
                "q_lt_010": int((grp["q_value_bh"] < 0.10).sum()),
                "q_lt_005_post_aff": int(
                    ((grp["q_value_bh"] < 0.05) & (grp["delta_pmi"] > 0)).sum()
                ),
                "q_lt_005_pre_aff": int(
                    ((grp["q_value_bh"] < 0.05) & (grp["delta_pmi"] < 0)).sum()
                ),
            }
        )
    return pd.DataFrame(out)


def _summary_markdown(
    df: pd.DataFrame,
    bh_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Pronoun collocation summary\n")
    lines.append(
        "Auto-generated by `02_modeling_pronoun_collocation_summary.py`. "
        "Source: dependency-parsed head-lemma counts per (language, cell, period, deprel) "
        "from `02_modeling_pronoun_collocations.py`. BH-FDR is applied within each "
        "(language, cell) family across head lemmas on the G² period-contrast p-value.\n\n"
    )

    lines.append("## BH-FDR survivor counts\n\n")
    lines.append(bh_summary.to_markdown(index=False))
    lines.append("\n\n")

    for lang in sorted(df["language"].unique()):
        for cell in PRIMARY_CELLS:
            sub = df.loc[df["language"].eq(lang) & df["cell"].eq(cell)]
            if sub.empty:
                continue
            lines.append(f"## {lang} · {cell}\n\n")
            for deprel in ("nsubj", "obj", "obl", "nmod"):
                pos, neg = _top_movers(sub, lang, cell, deprel, n=10)
                if pos.empty and neg.empty:
                    continue
                lines.append(f"### deprel = `{deprel}`\n\n")
                lines.append("**post-2022 affinity (top ΔPMI > 0)**\n\n")
                if not pos.empty:
                    pos_show = pos[
                        ["head_lemma", "delta_pmi", "cooc_2014_2021", "cooc_post_2022", "q_value_bh"]
                    ].copy()
                    pos_show.columns = ["head_lemma", "ΔPMI", "n_p1", "n_p2", "q_BH"]
                    lines.append(pos_show.to_markdown(index=False, floatfmt=".3f"))
                    lines.append("\n\n")
                lines.append("**pre-2022 affinity (top ΔPMI < 0)**\n\n")
                if not neg.empty:
                    neg_show = neg[
                        ["head_lemma", "delta_pmi", "cooc_2014_2021", "cooc_post_2022", "q_value_bh"]
                    ].copy()
                    neg_show.columns = ["head_lemma", "ΔPMI", "n_p1", "n_p2", "q_BH"]
                    lines.append(neg_show.to_markdown(index=False, floatfmt=".3f"))
                    lines.append("\n\n")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(args.input_dir)
    log.info("Loaded %d head-lemma records (after MIN_COOCC_TOTAL filter)", len(df))

    wide = _wide_table(df, out_dir)
    log.info("Wrote top_movers_per_cell_deprel.csv with %d rows", len(wide))

    bh_summary = _bh_survivors(df)
    bh_summary.to_csv(out_dir / "bh_survivor_counts.csv", index=False)

    for lang in sorted(df["language"].unique()):
        for cell in PRIMARY_CELLS:
            _barbell_figure(df, lang, cell, out_dir)

    _summary_markdown(df, bh_summary, out_dir)
    log.info("Wrote summary outputs to %s", out_dir)


if __name__ == "__main__":
    main()
