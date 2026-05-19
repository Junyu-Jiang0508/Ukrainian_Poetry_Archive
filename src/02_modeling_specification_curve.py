"""Aggregate Q1/Q1c robustness outputs into a specification-curve table and plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_specification_curve"
DEFAULT_Q1_DIR = ROOT / "outputs" / "02_modeling_q1_per_cell_glm"
DEFAULT_PERIOD_DIR = ROOT / "outputs" / "02_modeling_robustness_period"
DEFAULT_RATIO_DIR = ROOT / "outputs" / "02_modeling_ratio_q1_binomial"


def _read(path: Path, spec_label: str, source: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    df["spec_label"] = spec_label
    df["spec_source"] = source
    return df


def _collect_inputs(q1_dir: Path, period_dir: Path, ratio_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    # Post-P0-3: the unsuffixed Q1 file now carries the token-offset (primary)
    # estimates; the stanza offset has moved to a suffixed sensitivity file.
    files = [
        (q1_dir / "q1_poem_per_cell_glm_by_language_coprimary.csv", "q1_primary_tokens_coprimary", "q1"),
        (q1_dir / "q1_poem_per_cell_glm_by_language_offset_n_stanzas_coprimary.csv", "q1_offset_stanzas", "q1"),
        (q1_dir / "q1_poem_per_cell_glm_by_language_offset_n_finite_verbs_coprimary.csv", "q1_offset_fv", "q1"),
        (
            q1_dir / "q1_poem_per_cell_glm_by_language_offset_n_finite_verbs_excl_imperative_coprimary.csv",
            "q1_offset_fv_excl_imp",
            "q1",
        ),
        (period_dir / "q1_poem_per_cell_glm_robust_period_primary_calendar.csv", "period_primary_calendar", "period"),
        (period_dir / "q1_poem_per_cell_glm_robust_period_triple_drop_pre2014.csv", "period_drop_pre2014", "period"),
        (period_dir / "q1_poem_per_cell_glm_robust_period_invasion_20220224.csv", "period_posted_invasion", "period"),
        (period_dir / "q1_poem_per_cell_glm_robust_period_author_onset_le2014.csv", "period_author_onset_le2014", "period"),
        (ratio_dir / "ratio_q1_binomial_by_language.csv", "ratio_primary", "ratio"),
    ]
    for path, spec_label, source in files:
        d = _read(path, spec_label, source)
        if not d.empty:
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prepare_curve(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in ("language_stratum", "cell", "model_variant", "cohort_definition", "spec_label", "spec_source", "rate_ratio_post_vs_pre", "rate_ratio_ci95_low", "rate_ratio_ci95_high", "q_value_bh_within_stratum", "q_value_bh_wild_within_stratum", "n_poems", "n_authors") if c in df.columns]
    out = df[keep_cols].copy()
    if "model_variant" not in out.columns:
        out["model_variant"] = "poisson_cluster"
    if "cohort_definition" not in out.columns:
        out["cohort_definition"] = "none"
    if "rate_ratio_post_vs_pre" not in out.columns and "OR_post_vs_pre" in out.columns:
        out["rate_ratio_post_vs_pre"] = out["OR_post_vs_pre"]
    if "rate_ratio_ci95_low" not in out.columns and "ci95_low" in out.columns:
        out["rate_ratio_ci95_low"] = np.exp(out["ci95_low"])
    if "rate_ratio_ci95_high" not in out.columns and "ci95_high" in out.columns:
        out["rate_ratio_ci95_high"] = np.exp(out["ci95_high"])
    if "cell" not in out.columns and "ratio_index" in out.columns:
        out["cell"] = out["ratio_index"]
    if "q_value_bh_within_stratum" not in out.columns and "q_value_bh" in out.columns:
        out["q_value_bh_within_stratum"] = out["q_value_bh"]
    out["direction_positive"] = out["rate_ratio_post_vs_pre"].gt(1.0)
    out["spec_id"] = (
        out["spec_source"].astype(str)
        + "|"
        + out["spec_label"].astype(str)
        + "|"
        + out["model_variant"].astype(str)
        + "|"
        + out["cohort_definition"].astype(str)
    )
    out = out.sort_values(["language_stratum", "cell", "rate_ratio_post_vs_pre"], ascending=[True, True, True]).reset_index(drop=True)
    out["curve_index"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _plot_curve(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = df["curve_index"].to_numpy(dtype=float)
    y = df["rate_ratio_post_vs_pre"].to_numpy(dtype=float)
    lo = df["rate_ratio_ci95_low"].to_numpy(dtype=float) if "rate_ratio_ci95_low" in df.columns else y
    hi = df["rate_ratio_ci95_high"].to_numpy(dtype=float) if "rate_ratio_ci95_high" in df.columns else y
    xerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])
    ratio_mask = df["spec_source"].astype(str).eq("ratio")
    if (~ratio_mask).any():
        ax.errorbar(
            x[~ratio_mask.to_numpy()],
            y[~ratio_mask.to_numpy()],
            yerr=xerr[:, ~ratio_mask.to_numpy()],
            fmt="o",
            ms=3,
            alpha=0.7,
            label="count-model specs",
        )
    if ratio_mask.any():
        ax.errorbar(
            x[ratio_mask.to_numpy()],
            y[ratio_mask.to_numpy()],
            yerr=xerr[:, ratio_mask.to_numpy()],
            fmt="D",
            ms=4,
            alpha=0.8,
            label="ratio specs",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Specification index (sorted by RR)")
    ax.set_ylabel("Rate ratio post vs pre")
    ax.set_title("Specification curve for Q1-family rate-ratio estimates")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _holm_step_down(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni step-down adjusted p-values (FWER, no dependence model)."""
    p = pd.to_numeric(p_values, errors="coerce")
    n_valid = p.notna().sum()
    if n_valid == 0:
        return pd.Series(np.full(len(p), np.nan), index=p.index)
    ranks = p.rank(method="first").astype(float)
    adj = (n_valid - ranks + 1.0) * p
    # Monotone non-decreasing along the sorted p-values.
    order = p.sort_values().index
    cum_max = -np.inf
    out = pd.Series(np.nan, index=p.index, dtype=float)
    for idx in order:
        cum_max = max(cum_max, float(adj.loc[idx]))
        out.loc[idx] = min(cum_max, 1.0)
    return out


def _romano_wolf_step_down(
    p_values: pd.Series,
    *,
    correlation: float = 0.5,
    n_bootstrap: int = 5000,
    seed: int = 20260519,
) -> pd.Series:
    """Parametric Romano-Wolf FWER-adjusted p-values under exchangeable correlation.

    Draw ``n_bootstrap`` Gaussian samples of length ``K`` with exchangeable
    correlation ``correlation`` between specs (no constraint that each spec
    points to the same underlying null), compute the bootstrap distribution
    of the *max* |z| over the remaining specs at each step-down stage, and
    use it to adjust each spec's p-value. Returns adjusted p-values aligned
    to the input index.

    The exchangeable-correlation assumption is conservative for spec families
    that share a denominator and aggressive for those that do not; we report
    Holm (no dependence model) alongside as the discipline.
    """
    p = pd.to_numeric(p_values, errors="coerce")
    valid_mask = p.notna() & p.gt(0.0)
    valid = p[valid_mask].copy()
    if len(valid) <= 1 or correlation < 0.0 or correlation >= 1.0:
        return _holm_step_down(p_values)

    from scipy.stats import norm

    # Convert each two-sided p-value to absolute z-score under H0.
    z_obs = norm.isf(valid.to_numpy() / 2.0)
    n = len(valid)
    rng = np.random.default_rng(seed)
    # Exchangeable AR(0) covariance: Sigma = (1 - rho) * I + rho * 11^T.
    common = rng.standard_normal(size=n_bootstrap) * np.sqrt(correlation)
    idio = rng.standard_normal(size=(n_bootstrap, n)) * np.sqrt(max(1.0 - correlation, 0.0))
    z_boot = np.abs(common[:, None] + idio)

    # Step-down: order specs by descending |z_obs|.
    order = np.argsort(-z_obs)
    adj = np.zeros(n)
    cum_max_threshold = 0.0
    for step, idx in enumerate(order):
        remaining = order[step:]
        max_over_remaining = z_boot[:, remaining].max(axis=1)
        p_step = float(np.mean(max_over_remaining >= z_obs[idx]))
        # Monotone: subsequent specs cannot have a smaller adjusted p than the prior step.
        cum_max_threshold = max(cum_max_threshold, p_step)
        adj[idx] = min(cum_max_threshold, 1.0)

    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    out.loc[valid.index] = adj
    return out


def _joint_inference_table(curve: pd.DataFrame, **rw_kwargs) -> pd.DataFrame:
    """Per (language_stratum, cell) joint-inference table across specs.

    Reports Holm-adjusted p-values and Romano-Wolf-adjusted p-values, plus the
    fraction of specs that remain significant at $\alpha = 0.05$ after each
    correction. The naïve per-spec ``p_value_clustered_author`` is also
    retained for transparency.
    """
    df = curve.copy()
    # Pick the most defensible per-spec p-value: prefer the wild-cluster
    # bootstrap p when available, otherwise the clustered-SE p.
    if "q_value_bh_wild_within_stratum" in df.columns and "q_value_bh_within_stratum" in df.columns:
        pass
    if "p_value_clustered_author" not in df.columns:
        # ratio-binomial path uses different column names; fall back.
        for cand in ("p_value", "p_clustered"):
            if cand in df.columns:
                df["p_value_clustered_author"] = df[cand]
                break
    if "p_value_clustered_author" not in df.columns:
        return pd.DataFrame()

    df["p_value_for_joint"] = df["p_value_clustered_author"].astype(float)

    rows: list[dict[str, object]] = []
    for (lang, cell), grp in df.groupby(["language_stratum", "cell"], sort=False):
        if grp.empty:
            continue
        holm = _holm_step_down(grp["p_value_for_joint"])
        rw = _romano_wolf_step_down(grp["p_value_for_joint"], **rw_kwargs)
        rows.append(
            {
                "language_stratum": lang,
                "cell": cell,
                "n_specs": int(len(grp)),
                "n_specs_p05_raw": int((grp["p_value_for_joint"] < 0.05).sum()),
                "n_specs_p05_holm": int((holm < 0.05).sum()),
                "n_specs_p05_romano_wolf": int((rw < 0.05).sum()),
                "min_p_raw": float(grp["p_value_for_joint"].min()),
                "min_p_holm": float(holm.min()),
                "min_p_romano_wolf": float(rw.min()),
                "median_rate_ratio": float(grp["rate_ratio_post_vs_pre"].median()),
                "share_positive_direction": float(grp["direction_positive"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _per_spec_adjusted_pvals(curve: pd.DataFrame, **rw_kwargs) -> pd.DataFrame:
    """Per-spec long table with adjusted p-values appended."""
    df = curve.copy()
    if "p_value_clustered_author" not in df.columns:
        return df
    df["holm_adj_p_within_cell_stratum"] = np.nan
    df["romano_wolf_adj_p_within_cell_stratum"] = np.nan
    for (lang, cell), grp in df.groupby(["language_stratum", "cell"], sort=False):
        holm = _holm_step_down(grp["p_value_clustered_author"])
        rw = _romano_wolf_step_down(grp["p_value_clustered_author"], **rw_kwargs)
        df.loc[holm.index, "holm_adj_p_within_cell_stratum"] = holm.values
        df.loc[rw.index, "romano_wolf_adj_p_within_cell_stratum"] = rw.values
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build specification-curve outputs from Q1-family robustness runs.")
    ap.add_argument("--q1-dir", type=Path, default=DEFAULT_Q1_DIR)
    ap.add_argument("--period-dir", type=Path, default=DEFAULT_PERIOD_DIR)
    ap.add_argument("--ratio-dir", type=Path, default=DEFAULT_RATIO_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--rw-correlation",
        type=float,
        default=0.5,
        help="Exchangeable correlation assumption for Romano-Wolf (0=independent specs).",
    )
    ap.add_argument("--rw-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _collect_inputs(args.q1_dir.resolve(), args.period_dir.resolve(), args.ratio_dir.resolve())
    if raw.empty:
        raise SystemExit("No input model outputs found. Run Q1/Q1c/robustness scripts first.")
    curve = _prepare_curve(raw)
    curve.to_csv(out_dir / "specification_curve_rr.csv", index=False)
    summary = (
        curve.groupby(["language_stratum", "cell"], dropna=False)["direction_positive"]
        .mean()
        .rename("share_positive_direction")
        .reset_index()
    )
    summary.to_csv(out_dir / "specification_curve_direction_summary.csv", index=False)

    # P2-4: Holm + Romano-Wolf joint inference across specs.
    rw_kwargs = {"correlation": args.rw_correlation, "n_bootstrap": args.rw_bootstrap}
    joint = _joint_inference_table(curve, **rw_kwargs)
    if not joint.empty:
        joint.to_csv(out_dir / "spec_curve_joint_inference.csv", index=False)
    per_spec_adj = _per_spec_adjusted_pvals(curve, **rw_kwargs)
    per_spec_adj.to_csv(out_dir / "spec_curve_per_spec_adjusted.csv", index=False)

    _plot_curve(curve, out_dir / "specification_curve_rr.png")
    print(f"Wrote specification curve outputs (incl. Romano-Wolf joint inference) to: {out_dir}")


if __name__ == "__main__":
    main()
