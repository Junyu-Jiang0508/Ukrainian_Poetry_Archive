"""Publication-style figures for outputs/02_modeling_significance_models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.pronoun_encoding import PRIMARY_GLM_CELLS
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
DEFAULT_INPUT = ROOT / "outputs" / "02_modeling_significance_models"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_significance_models" / "figures"

DEFAULT_Q1_GLM_CSV = ROOT / "outputs" / "02_modeling_q1_per_cell_glm" / "q1_poem_per_cell_glm_by_language.csv"
DEFAULT_Q1B_FE_CSV = ROOT / "outputs" / "02_modeling_q1b_within_author_fe" / "q1b_within_author_fe_interactions.csv"

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


def plot_q1_per_cell_poisson_rr(df_glm: pd.DataFrame, out_dir: Path) -> None:
    """PRIMARY_GLM_CELLS × (Ukrainian, Russian); drops pooled stratum rows.

    Number of cells follows the inferential set (currently 4 after polite-singular
    was excluded for sparsity).
    """
    if df_glm.empty:
        return
    d = df_glm.copy()
    d = d.loc[~d["language_stratum"].eq("pooled_Ukrainian_Russian")].copy()
    if d.empty:
        return
    cells = [c for c in PRIMARY_GLM_CELLS if c in set(d["cell"])]
    if not cells:
        return
    strata_l = ("Ukrainian", "Russian")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), sharey=True, squeeze=False)
    for ax, st in zip(axes.ravel(), strata_l):
        sub = d.loc[d["language_stratum"].eq(st)]
        y = np.arange(len(cells))
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        for i, cell in enumerate(cells):
            r = sub.loc[sub["cell"].eq(cell)]
            if r.empty:
                continue
            row = r.iloc[0]
            xm = float(row["rate_ratio_post_vs_pre"])
            lo = float(row["rate_ratio_ci95_low"])
            hi = float(row["rate_ratio_ci95_high"])
            sig = False
            if "q_value_bh_within_stratum" in row.index and pd.notna(row["q_value_bh_within_stratum"]):
                sig = float(row["q_value_bh_within_stratum"]) < 0.05
            color = "#c0392b" if sig else "#2c3e50"
            ax.plot([lo, hi], [i, i], color=color, linewidth=2)
            ax.scatter(xm, i, s=38, color=color, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(cells)
        ax.set_xlabel("Rate ratio (post vs pre, 95% CI)")
        ax.set_title(st)
    fig.suptitle("Q1 per-cell Poisson GLM (offset exposure_n_stanzas)", fontsize=12.25, y=1.02)
    _save(fig, out_dir, "fig5_q1_per_cell_poisson_rr")


def plot_q1b_within_author_fe(
    df_fe: pd.DataFrame,
    out_dir: Path,
    *,
    sort_cell: str = "1pl",
) -> None:
    """Forest of author interaction coefficients; red if 95% CI excludes 0 on log scale."""
    if df_fe.empty:
        return
    need = {"language_stratum", "cell", "author", "interaction_coef_log_mu", "ci95_low", "ci95_high"}
    if need - set(df_fe.columns):
        return
    strata_l = ("Ukrainian", "Russian")
    for st in strata_l:
        d = df_fe.loc[df_fe["language_stratum"].eq(st) & df_fe["cell"].eq(sort_cell)].copy()
        if d.empty:
            continue
        d = d.sort_values("interaction_coef_log_mu", ascending=True)
        y = np.arange(len(d))
        xm = d["interaction_coef_log_mu"].to_numpy(dtype=float)
        lo = d["ci95_low"].to_numpy(dtype=float)
        hi = d["ci95_high"].to_numpy(dtype=float)
        exclude_zero = (lo > 0) | (hi < 0)
        fig, ax = plt.subplots(figsize=(7.6, max(4.2, 0.28 * len(d) + 1.2)))
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        for i in range(len(d)):
            color = "#c0392b" if exclude_zero[i] else "#2c3e50"
            ax.plot([lo[i], hi[i]], [i, i], color=color, linewidth=2)
            ax.scatter(xm[i], i, s=32, color=color, zorder=3)
        ax.set_yticks(y)
        fontsize = max(7.0, min(9.0, 11.0 - 0.035 * len(d)))
        ax.set_yticklabels(d["author"].astype(str).tolist(), fontsize=fontsize)
        ax.set_xlabel(f"Interaction P2 vs reference (log μ scale), sorted by {sort_cell}")
        ax.set_title(f"Q1b author × period FE interactions — {st} — cell={sort_cell}")
        _save(fig, out_dir, f"fig6_q1b_within_author_fe_{st}_{sort_cell}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-style figures for 02_modeling_significance_models.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--q1-glm-csv",
        type=Path,
        default=None,
        help=f"Optional Q1 glm CSV (default tries {DEFAULT_Q1_GLM_CSV})",
    )
    parser.add_argument(
        "--q1b-fe-csv",
        type=Path,
        default=None,
        help=f"Optional Q1b FE interactions CSV (default tries {DEFAULT_Q1B_FE_CSV})",
    )
    parser.add_argument("--forest-sort-cell", type=str, default="1pl", help="Which cell determines Q1b sort / filter")
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

    q1_path = args.q1_glm_csv
    if q1_path is None and DEFAULT_Q1_GLM_CSV.is_file():
        q1_path = DEFAULT_Q1_GLM_CSV
    if q1_path is not None and Path(q1_path).is_file():
        plot_q1_per_cell_poisson_rr(pd.read_csv(q1_path, low_memory=False), out_dir)

    q1b_path = args.q1b_fe_csv
    if q1b_path is None and DEFAULT_Q1B_FE_CSV.is_file():
        q1b_path = DEFAULT_Q1B_FE_CSV
    if q1b_path is not None and Path(q1b_path).is_file():
        plot_q1b_within_author_fe(
            pd.read_csv(q1b_path, low_memory=False),
            out_dir,
            sort_cell=str(args.forest_sort_cell),
        )

    print(f"Wrote publication figures to: {out_dir}")


if __name__ == "__main__":
    main()

