"""Q3 supplementary: sparse aggregated 2pl at author × period × stratum (not in Q1/Q2 BH family)."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum
from utils.workspace import prepare_analysis_environment

ENV_ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_OUTPUT = ENV_ROOT / "outputs" / "02_modeling_q3_sparse_2pl_aggregated"
PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"
PERIODS = (PERIOD_P1, PERIOD_P2)

DESCRIPTIVE_NOTE = (
    "Descriptive only. Aggregates the legacy morphologically-defined 2pl column "
    "(= 2pl_vy_polite_singular + 2pl_vy_true_plural splits combined). This is NOT the "
    "polite-singular cell; that cell has 23 events in the corpus and is excluded from "
    "the inferential tier (see Q1 README). 2pl is too sparse at poem level for the primary "
    "inferential family (not in Q1/Q2 BH). Aggregated Poisson + offset(log sum "
    "exposure_n_stanzas); cluster-robust SE by author where applicable."
)


def _aggregate_sparse(poem_tbl: pd.DataFrame, *, language_stratum: str) -> pd.DataFrame:
    sub = filter_poems_by_language_stratum(poem_tbl, language_stratum)
    sub = sub.loc[sub["period3"].isin(PERIODS)]
    if "include_in_offset_models" in sub.columns:
        sub = sub.loc[sub["include_in_offset_models"].astype(bool)]
    sub["_ex"] = sub["exposure_n_stanzas"].astype(float)
    sub = sub[sub["_ex"].gt(0)]
    if sub.empty:
        return pd.DataFrame()
    g = (
        sub.groupby(["author", "period3"], dropna=False)
        .agg(legacy_2pl_agg=("2pl", "sum"), exposure_n_stanzas=("exposure_n_stanzas", "sum"))
        .reset_index()
    )
    g["language_stratum"] = language_stratum
    g["log_exposure"] = np.log(g["exposure_n_stanzas"].astype(float))
    return g


def _fit_aggregate_poisson(adf: pd.DataFrame) -> dict[str, float | int | str] | None:
    if adf.empty or adf["period3"].nunique() < 2:
        return None
    cdf = adf.copy()
    cdf["k"] = cdf["legacy_2pl_agg"].astype(int)
    groups = cdf["author"].astype(str)
    formula = f"k ~ C(period3, Treatment('{PERIOD_P1}'))"
    fit = smf.glm(formula, data=cdf, family=sm.families.Poisson(), offset=cdf["log_exposure"]).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups},
    )
    term = f"C(period3, Treatment('{PERIOD_P1}'))[T.{PERIOD_P2}]"
    if term not in fit.params.index:
        return None
    ci = fit.conf_int().loc[term]
    coef = float(fit.params[term])
    return {
        "analysis_tier": "supplementary_sparse_2pl",
        "language_stratum": str(cdf["language_stratum"].iloc[0]),
        "n_agg_rows": int(len(cdf)),
        "n_authors": int(cdf["author"].nunique()),
        "model_family": "poisson_offset_cluster_author",
        "coef_post_vs_pre_log_mu": coef,
        "rate_ratio_post_vs_pre": float(np.exp(coef)),
        "ci95_low_log_mu": float(ci.iloc[0]),
        "ci95_high_log_mu": float(ci.iloc[1]),
        "rate_ratio_ci95_low": float(np.exp(ci.iloc[0])),
        "rate_ratio_ci95_high": float(np.exp(ci.iloc[1])),
        "p_value_clustered_author": float(fit.pvalues[term]),
        "descriptive_note": DESCRIPTIVE_NOTE,
    }


def _fit_zinb_optional(adf: pd.DataFrame) -> dict[str, float | int | str] | None:
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    except ImportError:
        return None

    if adf.shape[0] < 15:
        return None
    cdf = adf.copy()
    endog = cdf["legacy_2pl_agg"].astype(float).values
    exog = pd.DataFrame({"const": 1.0, "period_post": (cdf["period3"] == PERIOD_P2).astype(float).values})
    exog_infl = exog.copy()
    off = cdf["log_exposure"].astype(float).values

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        mod = ZeroInflatedNegativeBinomialP(
            endog,
            exog,
            exog_infl=exog_infl,
            offset=off,
        )
        res = mod.fit(method="nm", maxiter=500, disp=0)
    except Exception:
        return None
    finally:
        warnings.resetwarnings()

    coef = float(res.params.get("period_post", np.nan))
    return {
        "analysis_tier": "supplementary_sparse_2pl",
        "language_stratum": str(cdf["language_stratum"].iloc[0]),
        "n_agg_rows": int(len(cdf)),
        "n_authors": int(cdf["author"].nunique()),
        "model_family": "zinb_descriptive_marginal",
        "zinb_coef_period_post_raw": coef,
        "descriptive_note": DESCRIPTIVE_NOTE + " ZINB path is marginal and may not converge.",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Q3 supplementary sparse aggregated 2pl models.")
    ap.add_argument("--poem-table", type=Path, required=True, help="q1_poem_unit_cell_counts_12.csv from Q1")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--try-zinb", action="store_true", help="Also attempt descriptive ZINB (may skip on failure).")
    args = ap.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    poem_tbl = pd.read_csv(args.poem_table.resolve(), low_memory=False)
    poem_tbl["poem_id"] = poem_tbl["poem_id"].astype(str).str.strip()

    rows: list[dict[str, float | int | str]] = []
    for strat in LANGUAGE_STRATA:
        agg = _aggregate_sparse(poem_tbl, language_stratum=strat)
        r = _fit_aggregate_poisson(agg)
        if r:
            rows.append(r)
        if args.try_zinb:
            z = _fit_zinb_optional(agg)
            if z:
                rows.append(z)

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "q3_sparse_2pl_aggregated_supplementary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} fit rows)")


if __name__ == "__main__":
    main()
