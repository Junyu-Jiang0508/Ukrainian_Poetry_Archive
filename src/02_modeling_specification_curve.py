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


def _read(path: Path, spec_label: str, source: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    df = df.copy()
    df["spec_label"] = spec_label
    df["spec_source"] = source
    return df


def _collect_inputs(q1_dir: Path, period_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    files = [
        (q1_dir / "q1_poem_per_cell_glm_by_language_coprimary.csv", "q1_primary_coprimary", "q1"),
        (q1_dir / "q1_poem_per_cell_glm_by_language_offset_n_tokens_coprimary.csv", "q1_offset_tokens", "q1"),
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
    ax.errorbar(x, y, yerr=xerr, fmt="o", ms=3, alpha=0.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Specification index (sorted by RR)")
    ax.set_ylabel("Rate ratio post vs pre")
    ax.set_title("Specification curve for Q1-family rate-ratio estimates")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build specification-curve outputs from Q1-family robustness runs.")
    ap.add_argument("--q1-dir", type=Path, default=DEFAULT_Q1_DIR)
    ap.add_argument("--period-dir", type=Path, default=DEFAULT_PERIOD_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _collect_inputs(args.q1_dir.resolve(), args.period_dir.resolve())
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
    _plot_curve(curve, out_dir / "specification_curve_rr.png")
    print(f"Wrote specification curve outputs to: {out_dir}")


if __name__ == "__main__":
    main()
