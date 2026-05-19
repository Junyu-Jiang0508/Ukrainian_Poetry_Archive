"""Smooth-year random-effect model and PELT-based breakpoint CI (P2-2).

The binary period contrast (P1: 2014--2021, P2: post-2022) used in Q1/Q2a
imposes a step at 24 February 2022. That assumption is substantively
motivated --- the full-scale invasion is the substantive intervention ---
but it is *not* a property the data are required to confirm. This stage:

1. Fits a **B-spline smooth in year** Poisson GLM per cell × language with the
   primary token-exposure offset, returning the per-year predicted rate
   trajectory and an in-data $\chi^2$ test of smooth-vs-binary.
2. Runs **PELT change-point detection** \citep{killick2012optimal} on the
   per-year corpus-level rate trajectory per cell, returning the detected
   change point(s) with a bootstrap 95\% CI.
3. Reports whether the bootstrap CI of the detected change point covers
   2022.0 (the data-driven check on the substantive cutpoint).

Inputs come from the same poem-level cell-count table that 02b uses (built
by :func:`utils.poem_cell_counts.build_poem_cell_table_with_exposure`).

Outputs (``outputs/02_modeling_breakpoint_smooth_year/``)
---------------------------------------------------------
* ``smooth_year_fits.csv``        — per (language, cell), spline coefficients,
  smooth-vs-binary $\chi^2$, AIC, BIC.
* ``rate_by_year.csv``            — per-year aggregated rate (k/exposure) per cell.
* ``breakpoint_ci.csv``           — PELT-detected breakpoints with bootstrap CI.
* ``rate_trajectory_<cell>.png``  — annotated time-series figure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.finite_verb_exposure import resolve_finite_verb_counts_for_modeling
from utils.language_strata import (
    LANGUAGE_STRATA,
    filter_annotation_for_inference_language,
    filter_poems_by_language_stratum,
)
from utils.poem_cell_counts import build_poem_cell_table_with_exposure
from utils.pronoun_encoding import PRIMARY_GLM_CELLS
from utils.stats_common import normalize_bool_flag
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_breakpoint_smooth_year"


def _bspline_basis(year: pd.Series, *, df: int = 5) -> pd.DataFrame:
    """Cubic B-spline basis on year via patsy. Knot placement is quantile-based."""
    from patsy import bs

    y = pd.to_numeric(year, errors="coerce").to_numpy(dtype=float)
    basis = bs(y, df=df, degree=3, include_intercept=False)
    return pd.DataFrame(basis, columns=[f"bs_{i}" for i in range(basis.shape[1])])


def _fit_smooth_year_for_cell(
    poem_df: pd.DataFrame,
    cell: str,
    language: str,
    *,
    df_spline: int = 5,
) -> dict[str, object] | None:
    """Fit `k ~ bs(year, df=5) + offset(log_tokens)` Poisson, clustered SE."""
    sub = poem_df.copy()
    sub = sub.loc[sub["exposure_n_tokens"].astype(float).gt(0)].copy()
    sub["k"] = sub[cell].astype(int)
    sub["log_exposure"] = np.log(sub["exposure_n_tokens"].astype(float))
    sub["year_int"] = pd.to_numeric(sub["year"], errors="coerce")
    sub = sub.dropna(subset=["year_int"]).copy()
    if sub.empty or int(sub["k"].sum()) == 0 or sub["year_int"].nunique() < df_spline + 1:
        return None

    basis = _bspline_basis(sub["year_int"], df=df_spline)
    basis.index = sub.index
    fit_df = pd.concat([sub[["k", "log_exposure", "author"]], basis], axis=1)
    bs_terms = " + ".join(basis.columns)
    formula = f"k ~ {bs_terms}"
    try:
        fit_smooth = smf.glm(
            formula, data=fit_df, family=sm.families.Poisson(), offset=fit_df["log_exposure"]
        ).fit(cov_type="cluster", cov_kwds={"groups": fit_df["author"].astype(str)})
    except Exception as exc:
        log.warning("Smooth fit failed for %s/%s: %s", language, cell, exc)
        return None

    try:
        fit_null = smf.glm(
            "k ~ 1", data=fit_df, family=sm.families.Poisson(), offset=fit_df["log_exposure"]
        ).fit()
    except Exception:
        fit_null = None

    if fit_null is not None:
        from scipy.stats import chi2

        deviance_drop = float(fit_null.deviance - fit_smooth.deviance)
        df_diff = int(fit_smooth.df_model - fit_null.df_model)
        p_value = float(1.0 - chi2.cdf(deviance_drop, df=max(df_diff, 1))) if df_diff > 0 else np.nan
    else:
        deviance_drop = np.nan
        df_diff = np.nan
        p_value = np.nan

    # Predicted rate trajectory: predict at each integer year.
    yrs = np.arange(int(sub["year_int"].min()), int(sub["year_int"].max()) + 1)
    basis_pred = _bspline_basis(pd.Series(yrs), df=df_spline)
    pred_df = basis_pred.copy()
    pred_df["log_exposure"] = 0.0  # rate at zero offset = predicted rate per token
    mu = fit_smooth.predict(pred_df)
    pred_traj = pd.DataFrame({"year": yrs, "predicted_rate_per_token": mu.to_numpy(dtype=float)})

    return {
        "language": language,
        "cell": cell,
        "n_poems": int(len(sub)),
        "n_authors": int(sub["author"].nunique()),
        "deviance_smooth": float(fit_smooth.deviance),
        "deviance_null": float(fit_null.deviance) if fit_null is not None else np.nan,
        "deviance_drop": deviance_drop,
        "df_smooth_vs_null": df_diff,
        "p_smooth_vs_null_chi2": p_value,
        "aic_smooth": float(fit_smooth.aic),
        "bic_smooth": float(fit_smooth.bic),
        "predicted_traj": pred_traj,
    }


def _per_year_rates(poem_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-year corpus rate per cell, using token exposure as denominator."""
    yr = pd.to_numeric(poem_df["year"], errors="coerce")
    poem_df = poem_df.assign(year_int=yr)
    rows: list[dict[str, object]] = []
    for cell in PRIMARY_GLM_CELLS:
        sub = poem_df.dropna(subset=["year_int"]).copy()
        sub["_k"] = sub[cell].astype(int)
        sub["_ex"] = sub["exposure_n_tokens"].astype(float)
        agg = sub.groupby(["language_clean", "year_int"], sort=True).agg(
            k=("_k", "sum"),
            exposure=("_ex", "sum"),
        )
        agg["rate"] = agg["k"] / agg["exposure"].replace(0, np.nan)
        agg = agg.reset_index()
        agg["cell"] = cell
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def _pelt_breakpoint_with_ci(
    rate_series: pd.Series,
    *,
    n_bootstrap: int = 500,
    pen: float = 4.0,
    seed: int = 20260519,
) -> dict[str, object]:
    """Detect a single breakpoint via PELT with bootstrap 95% CI."""
    import ruptures as rpt

    y = rate_series.dropna().to_numpy(dtype=float)
    idx = rate_series.dropna().index.to_numpy(dtype=int)
    if len(y) < 6:
        return {
            "detected_year": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "covers_2022": False,
            "n_points": int(len(y)),
        }
    algo = rpt.Pelt(model="rbf").fit(y)
    bkps = algo.predict(pen=pen)
    if not bkps or bkps[0] >= len(y):
        return {
            "detected_year": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "covers_2022": False,
            "n_points": int(len(y)),
        }
    primary_break_year = float(idx[bkps[0]]) if bkps[0] < len(idx) else float("nan")

    rng = np.random.default_rng(seed)
    boot_years: list[float] = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(y), size=len(y))
        y_boot = y[sample_idx]
        try:
            bkps_b = rpt.Pelt(model="rbf").fit(y_boot).predict(pen=pen)
        except Exception:
            continue
        if not bkps_b or bkps_b[0] >= len(y):
            continue
        boot_years.append(float(idx[bkps_b[0]]))
    if boot_years:
        lo = float(np.percentile(boot_years, 2.5))
        hi = float(np.percentile(boot_years, 97.5))
    else:
        lo = hi = float("nan")
    covers = bool(lo <= 2022.0 <= hi) if not (np.isnan(lo) or np.isnan(hi)) else False
    return {
        "detected_year": primary_break_year,
        "ci_low": lo,
        "ci_high": hi,
        "covers_2022": covers,
        "n_points": int(len(y)),
        "n_bootstrap_valid": int(len(boot_years)),
    }


def _plot_trajectory(
    smooth_fits: list[dict[str, object]],
    rate_table: pd.DataFrame,
    breakpoints: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    by_cell = {f["cell"]: f for f in smooth_fits if f is not None}
    for cell in PRIMARY_GLM_CELLS:
        fig, axes = plt.subplots(1, len(LANGUAGE_STRATA[1:]), figsize=(11, 4), sharey=True)
        for ax, lang in zip(axes, LANGUAGE_STRATA[1:]):  # Ukrainian, Russian (skip pooled)
            sub_rate = rate_table.loc[
                (rate_table["cell"].eq(cell)) & (rate_table["language_clean"].eq(lang))
            ].sort_values("year_int")
            if sub_rate.empty:
                ax.set_visible(False)
                continue
            ax.scatter(sub_rate["year_int"], sub_rate["rate"], s=18, color="0.4", alpha=0.7)
            # Overlay the spline-predicted rate where available.
            fit = by_cell.get(cell)
            if fit is not None and fit["language"] == lang:
                traj = fit["predicted_traj"]
                ax.plot(traj["year"], traj["predicted_rate_per_token"], color="C0", lw=2)
            # Breakpoint annotation.
            bp = breakpoints.loc[(breakpoints["cell"].eq(cell)) & (breakpoints["language"].eq(lang))]
            if not bp.empty:
                row = bp.iloc[0]
                if not pd.isna(row["detected_year"]):
                    ax.axvline(row["detected_year"], color="crimson", lw=1.5, ls="--", alpha=0.8)
                    if not pd.isna(row["ci_low"]):
                        ax.axvspan(row["ci_low"], row["ci_high"], color="crimson", alpha=0.12)
            ax.axvline(2022.0, color="black", lw=1.0, ls=":", alpha=0.5)
            ax.set_title(f"{lang} · {cell}")
            ax.set_xlabel("Year of composition")
        axes[0].set_ylabel("Rate per token")
        fig.tight_layout()
        fig.savefig(out_dir / f"rate_trajectory_{cell}.png", dpi=200)
        plt.close(fig)


def load_and_build_poem_table(
    annot_path: Path,
    layer0_path: Path,
) -> pd.DataFrame:
    """Mirror Q1 GLM's load+build chain so this stage uses the same denominators."""
    df = pd.read_csv(annot_path, low_memory=False, on_bad_lines="skip")
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df["language_clean"] = df["language"].fillna("").astype(str).str.strip()
    if "is_repeat" in df.columns and "is_translation" in df.columns:
        df["is_repeat"] = normalize_bool_flag(df["is_repeat"])
        df["is_translation"] = normalize_bool_flag(df["is_translation"])
    elif layer0_path is not None and layer0_path.is_file():
        l0 = pd.read_csv(
            layer0_path,
            usecols=["poem_id", "Is repeat", "I.D. of original (if poem is a translation)"],
            low_memory=False,
        )
        oid = l0["I.D. of original (if poem is a translation)"]
        flags = pd.DataFrame(
            {
                "poem_id": l0["poem_id"].astype(str).str.strip(),
                "is_repeat": l0["Is repeat"].astype(str).str.lower().str.strip().eq("yes"),
                "is_translation": oid.notna() & oid.astype(str).str.strip().ne(""),
            }
        ).drop_duplicates(subset=["poem_id"], keep="first")
        df = df.merge(flags, on="poem_id", how="left")
        df["is_repeat"] = df["is_repeat"].fillna(False).astype(bool)
        df["is_translation"] = df["is_translation"].fillna(False).astype(bool)
    else:
        df = df.assign(is_repeat=False, is_translation=False)
    df = df.loc[~(df["is_repeat"] | df["is_translation"])].copy()
    df, _ = filter_annotation_for_inference_language(df)
    fv_df = resolve_finite_verb_counts_for_modeling(ROOT, exposure_type="n_finite_verbs")
    return build_poem_cell_table_with_exposure(df, finite_verb_df=fv_df)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Smooth-year smooth + PELT breakpoint (P2-2).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bspline-df", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--pelt-penalty", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    poem_full = load_and_build_poem_table(args.input.resolve(), args.layer0.resolve())

    rate_table = _per_year_rates(poem_full)
    rate_table.to_csv(out_dir / "rate_by_year.csv", index=False)

    smooth_records: list[dict[str, object]] = []
    breakpoint_records: list[dict[str, object]] = []
    for stratum in ("Ukrainian", "Russian"):
        sub_poem = filter_poems_by_language_stratum(poem_full, stratum)
        for cell in PRIMARY_GLM_CELLS:
            result = _fit_smooth_year_for_cell(sub_poem, cell, stratum, df_spline=args.bspline_df)
            if result is not None:
                rec = {k: v for k, v in result.items() if k != "predicted_traj"}
                smooth_records.append(rec)

            rate_series = (
                rate_table.loc[
                    (rate_table["cell"].eq(cell)) & (rate_table["language_clean"].eq(stratum))
                ]
                .set_index("year_int")["rate"]
            )
            bp = _pelt_breakpoint_with_ci(
                rate_series,
                n_bootstrap=args.n_bootstrap,
                pen=args.pelt_penalty,
            )
            breakpoint_records.append({"language": stratum, "cell": cell, **bp})

    if smooth_records:
        pd.DataFrame(smooth_records).to_csv(out_dir / "smooth_year_fits.csv", index=False)
    breakpoints = pd.DataFrame(breakpoint_records)
    breakpoints.to_csv(out_dir / "breakpoint_ci.csv", index=False)

    smooth_fits_for_plot: list[dict[str, object]] = []
    for stratum in ("Ukrainian", "Russian"):
        sub_poem = filter_poems_by_language_stratum(poem_full, stratum)
        for cell in PRIMARY_GLM_CELLS:
            result = _fit_smooth_year_for_cell(sub_poem, cell, stratum, df_spline=args.bspline_df)
            if result is not None:
                smooth_fits_for_plot.append(result)
    _plot_trajectory(smooth_fits_for_plot, rate_table, breakpoints, out_dir)

    log.info("Wrote smooth-year and breakpoint outputs to %s", out_dir)


if __name__ == "__main__":
    main()
