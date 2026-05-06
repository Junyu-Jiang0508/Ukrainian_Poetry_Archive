"""Publication-style figures for outputs/14_significance_models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
DEFAULT_INPUT = ROOT / "outputs" / "14_significance_models"
DEFAULT_OUTPUT = ROOT / "outputs" / "14_significance_models" / "figures"

FEATURE_LABELS = {
    "prop_1st": "1st person share",
    "prop_2nd": "2nd person share",
    "prop_3rd": "3rd person share",
    "prop_plural": "Plural share",
    "prop_pro_drop": "Pro-drop share",
}


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_poem_overall(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    d = df.copy()
    d["feature_label"] = d["feature"].map(FEATURE_LABELS).fillna(d["feature"])
    d = d.sort_values("coef_post_2022_logit")
    y = np.arange(len(d))
    x = d["coef_post_2022_logit"].to_numpy()
    lo = d["ci95_low"].to_numpy()
    hi = d["ci95_high"].to_numpy()
    sig = d["q_value_bh"].fillna(1.0).to_numpy() < 0.05

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.8)
    for i in range(len(d)):
        color = "#c0392b" if sig[i] else "#2c3e50"
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=color, linewidth=2)
        ax.scatter(x[i], y[i], s=42, color=color, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature_label"].tolist())
    ax.set_xlabel("Post-2022 effect (logit coefficient, 95% CI)")
    ax.set_title("Poem-level effects (overall)")
    _save(fig, out_dir, "fig1_poem_overall_forest")


def plot_poem_by_language(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    langs = [l for l in ["Ukrainian", "Russian", "Qirimli"] if l in set(df["language"])]
    if not langs:
        return
    fig, axes = plt.subplots(1, len(langs), figsize=(4.8 * len(langs), 5.2), sharex=True, sharey=True)
    if len(langs) == 1:
        axes = [axes]
    for ax, lang in zip(axes, langs):
        d = df[df["language"] == lang].copy()
        d["feature_label"] = d["feature"].map(FEATURE_LABELS).fillna(d["feature"])
        d = d.sort_values("coef_post_2022_logit")
        y = np.arange(len(d))
        x = d["coef_post_2022_logit"].to_numpy()
        lo = d["ci95_low"].to_numpy()
        hi = d["ci95_high"].to_numpy()
        sig = d["q_value_bh_within_language"].fillna(1.0).to_numpy() < 0.05
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.8)
        for i in range(len(d)):
            color = "#c0392b" if sig[i] else "#2c3e50"
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=color, linewidth=2)
            ax.scatter(x[i], y[i], s=36, color=color, zorder=3)
        ax.set_title(lang)
        ax.set_yticks(y)
        ax.set_yticklabels(d["feature_label"].tolist())
        ax.set_xlabel("Post-2022 effect (logit coef)")
    fig.suptitle("Poem-level effects by language", y=1.02)
    _save(fig, out_dir, "fig2_poem_by_language_forest")


def plot_stanza_models(df_plural: pd.DataFrame, df_pn: pd.DataFrame, out_dir: Path) -> None:
    if df_plural.empty and df_pn.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

    if not df_plural.empty:
        d = df_plural.copy()
        y = np.arange(len(d))
        x = d["odds_ratio_post_2022"].to_numpy()
        lo = d["or_ci95_low"].to_numpy()
        hi = d["or_ci95_high"].to_numpy()
        sig = d["q_value_bh"].fillna(1.0).to_numpy() < 0.05
        axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
        for i in range(len(d)):
            color = "#c0392b" if sig[i] else "#2c3e50"
            axes[0].plot([lo[i], hi[i]], [y[i], y[i]], color=color, linewidth=2)
            axes[0].scatter(x[i], y[i], color=color, s=40, zorder=3)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(d["outcome"].tolist())
        axes[0].set_xlabel("Odds ratio (post vs pre)")
        axes[0].set_title("Stanza number model")

    if not df_pn.empty:
        d = df_pn.copy().sort_values("odds_ratio_post_2022")
        y = np.arange(len(d))
        x = d["odds_ratio_post_2022"].to_numpy()
        lo = d["or_ci95_low"].to_numpy()
        hi = d["or_ci95_high"].to_numpy()
        sig = d["q_value_bh"].fillna(1.0).to_numpy() < 0.05
        axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
        for i in range(len(d)):
            color = "#c0392b" if sig[i] else "#2c3e50"
            axes[1].plot([lo[i], hi[i]], [y[i], y[i]], color=color, linewidth=2)
            axes[1].scatter(x[i], y[i], color=color, s=40, zorder=3)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(d["category"].tolist())
        axes[1].set_xlabel("Odds ratio (post vs pre)")
        axes[1].set_title("Stanza PN one-vs-rest")

    _save(fig, out_dir, "fig3_stanza_or_forest")


def plot_segmented_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    terms = ["post_2014", "t_after_2014", "post_2022", "t_after_2022"]
    feats = ["prop_1st", "prop_2nd", "prop_3rd", "prop_plural", "prop_pro_drop"]
    d = df[df["term"].isin(terms) & df["feature"].isin(feats)].copy()
    if d.empty:
        return
    mat = (
        d.pivot(index="feature", columns="term", values="coef")
        .reindex(index=feats, columns=terms)
        .to_numpy(dtype=float)
    )
    q = d.pivot(index="feature", columns="term", values="q_value_bh_within_term").reindex(index=feats, columns=terms)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(terms)))
    ax.set_xticklabels(terms, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in feats])
    for i in range(len(feats)):
        for j in range(len(terms)):
            v = mat[i, j]
            if not np.isfinite(v):
                txt = "NA"
            else:
                star = "*" if (pd.notna(q.iloc[i, j]) and float(q.iloc[i, j]) < 0.05) else ""
                txt = f"{v:.2f}{star}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Segmented time effects (logit scale)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coefficient")
    _save(fig, out_dir, "fig4_segmented_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-style figures for 14_significance_models.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    poem_overall = pd.read_csv(in_dir / "poem_level_logit_ols.csv") if (in_dir / "poem_level_logit_ols.csv").is_file() else pd.DataFrame()
    poem_lang = pd.read_csv(in_dir / "poem_level_logit_ols_by_language.csv") if (in_dir / "poem_level_logit_ols_by_language.csv").is_file() else pd.DataFrame()
    st_plural = pd.read_csv(in_dir / "stanza_plural_glm.csv") if (in_dir / "stanza_plural_glm.csv").is_file() else pd.DataFrame()
    st_pn = pd.read_csv(in_dir / "stanza_pn_one_vs_rest_glm.csv") if (in_dir / "stanza_pn_one_vs_rest_glm.csv").is_file() else pd.DataFrame()
    segmented = pd.read_csv(in_dir / "poem_level_segmented_2014_2022.csv") if (in_dir / "poem_level_segmented_2014_2022.csv").is_file() else pd.DataFrame()

    plot_poem_overall(poem_overall, out_dir)
    plot_poem_by_language(poem_lang, out_dir)
    plot_stanza_models(st_plural, st_pn, out_dir)
    plot_segmented_heatmap(segmented, out_dir)

    print(f"Wrote publication figures to: {out_dir}")


if __name__ == "__main__":
    main()

