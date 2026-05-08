"""Population-level binomial GLM for pronoun ratio indices."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum, primary_stratum_for_bh
from utils.stats_common import bh_adjust
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "outputs" / "02_modeling_ratio_indices" / "ratio_poem_level_table.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_ratio_q1_binomial"
PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"

RATIO_SPECS = (
    ("ratio_1p_collective", "n1p_suc", "n1p_tri"),
    ("ratio_2p_collective", "n2p_suc", "n2p_tri"),
    ("ratio_overall_plural", "nov_suc", "nov_tri"),
)


def _wild_bootstrap_binomial(cdf: pd.DataFrame, formula: str, term: str, *, b_reps: int, seed: int) -> float:
    fit = smf.glm(formula, data=cdf, family=sm.families.Binomial(), freq_weights=cdf["tri"]).fit()
    if term not in fit.params.index or term not in fit.bse.index:
        return np.nan
    se_obs = float(fit.bse[term])
    if not np.isfinite(se_obs) or se_obs <= 0:
        return np.nan
    t_obs = float(fit.params[term] / se_obs)
    mu = np.clip(fit.fittedvalues.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    y = cdf["y"].to_numpy(dtype=float)
    resid = y - mu
    groups = cdf["author"].astype(str).to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    extreme = 0
    valid = 0
    for _ in range(int(b_reps)):
        sign = {g: rng.choice([-1.0, 1.0]) for g in uniq}
        w = np.array([sign[g] for g in groups], dtype=float)
        y_star = np.clip(mu + w * resid, 1e-6, 1 - 1e-6)
        bdf = cdf.copy()
        bdf["y"] = y_star
        try:
            bfit = smf.glm(formula, data=bdf, family=sm.families.Binomial(), freq_weights=bdf["tri"]).fit()
        except Exception:
            continue
        if term not in bfit.params.index or term not in bfit.bse.index:
            continue
        se_b = float(bfit.bse[term])
        if not np.isfinite(se_b) or se_b <= 0:
            continue
        t_b = float(bfit.params[term] / se_b)
        if np.isfinite(t_b):
            valid += 1
            if abs(t_b) >= abs(t_obs):
                extreme += 1
    if valid == 0:
        return np.nan
    return float((extreme + 1) / (valid + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Q1 analog for ratio outcomes: clustered binomial GLM.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--bootstrap-reps", type=int, default=1999)
    ap.add_argument("--bootstrap-seed", type=int, default=20260508)
    args = ap.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dat = pd.read_csv(args.input.resolve(), low_memory=False)
    dat = dat.loc[dat["period3"].isin((PERIOD_P1, PERIOD_P2))].copy()
    dat["author"] = dat["author"].astype(str)

    base_rows: list[dict[str, object]] = []
    cop_rows: list[dict[str, object]] = []
    excl_rows: list[dict[str, object]] = []
    for stratum in LANGUAGE_STRATA:
        sub = filter_poems_by_language_stratum(dat, stratum)
        is_primary = primary_stratum_for_bh(stratum)
        for ratio_index, suc_col, tri_col in RATIO_SPECS:
            cdf = sub.copy()
            cdf["suc"] = pd.to_numeric(cdf[suc_col], errors="coerce").fillna(0.0)
            cdf["tri"] = pd.to_numeric(cdf[tri_col], errors="coerce").fillna(0.0)
            cdf = cdf.loc[cdf["tri"].gt(0)].copy()
            excl_rows.append(
                {
                    "language_stratum": stratum,
                    "ratio_index": ratio_index,
                    "n_poems_after_denom_gate": int(len(cdf)),
                    "n_authors_after_denom_gate": int(cdf["author"].nunique()) if not cdf.empty else 0,
                }
            )
            if cdf.empty or cdf["period3"].nunique() < 2 or float(cdf["suc"].sum()) <= 0:
                continue
            cdf["y"] = cdf["suc"] / cdf["tri"]
            formula = f"y ~ C(period3, Treatment('{PERIOD_P1}'))"
            term = f"C(period3, Treatment('{PERIOD_P1}'))[T.{PERIOD_P2}]"
            fit = smf.glm(formula, data=cdf, family=sm.families.Binomial(), freq_weights=cdf["tri"]).fit(
                cov_type="cluster", cov_kwds={"groups": cdf["author"].astype(str)}
            )
            if term not in fit.params.index:
                continue
            coef = float(fit.params[term])
            ci = fit.conf_int().loc[term]
            row = {
                "language_stratum": stratum,
                "ratio_index": ratio_index,
                "n_poems": int(len(cdf)),
                "n_authors": int(cdf["author"].nunique()),
                "coef_log_odds": coef,
                "OR_post_vs_pre": float(np.exp(coef)),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "OR_ci95_low": float(np.exp(ci.iloc[0])),
                "OR_ci95_high": float(np.exp(ci.iloc[1])),
                "se_clustered": float(fit.bse[term]),
                "z_value": float(fit.tvalues[term]),
                "p_value": float(fit.pvalues[term]),
                "exposure_type": "ratio_trials",
                "is_primary_stratum": bool(is_primary),
            }
            base_rows.append(row)
            cop_rows.append({**row, "model_variant": "binomial_cluster", "p_value_wild_cluster_bootstrap": np.nan})
            p_wild = _wild_bootstrap_binomial(
                cdf,
                formula,
                term,
                b_reps=int(args.bootstrap_reps),
                seed=int(args.bootstrap_seed)
                + int(hashlib.md5(f"{stratum}|{ratio_index}".encode("utf-8")).hexdigest()[:8], 16) % 100_000,
            )
            cop_rows.append({**row, "model_variant": "bootstrap_p", "p_value_wild_cluster_bootstrap": p_wild})

    base = pd.DataFrame(base_rows)
    cop = pd.DataFrame(cop_rows)
    excl = pd.DataFrame(excl_rows)
    if not base.empty:
        base["q_value_bh"] = np.nan
        sub = base.loc[base["is_primary_stratum"]]
        if not sub.empty:
            q = sub.groupby("language_stratum", group_keys=False)["p_value"].apply(bh_adjust)
            base.loc[q.index, "q_value_bh"] = q
    if not cop.empty:
        cop["q_value_bh"] = np.nan
        cop["q_value_bh_wild"] = np.nan
        sub_c = cop.loc[cop["is_primary_stratum"]]
        if not sub_c.empty:
            q1 = sub_c.groupby(["language_stratum", "model_variant"], group_keys=False)["p_value"].apply(bh_adjust)
            cop.loc[q1.index, "q_value_bh"] = q1
            q2 = (
                sub_c.loc[sub_c["model_variant"].eq("bootstrap_p")]
                .groupby("language_stratum", group_keys=False)["p_value_wild_cluster_bootstrap"]
                .apply(bh_adjust)
            )
            cop.loc[q2.index, "q_value_bh_wild"] = q2

    base.to_csv(out_dir / "ratio_q1_binomial_by_language.csv", index=False)
    cop.to_csv(out_dir / "ratio_q1_binomial_by_language_coprimary.csv", index=False)
    excl.to_csv(out_dir / "ratio_q1_exclusion_audit.csv", index=False)
    print(f"Wrote ratio Q1 binomial outputs to: {out_dir}")


if __name__ == "__main__":
    main()
