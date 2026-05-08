"""Within-author FE binomial models for pronoun ratio indices."""

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
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_ratio_q1b_within_author_fe"
PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"
PERIODS = (PERIOD_P1, PERIOD_P2)
MIN_AUTHORS_PER_CELL_FIT = 5
SMOOTH_NUM = 0.5
SMOOTH_DEN = 1.0

RATIO_SPECS = (
    ("ratio_1p_collective", "n1p_suc", "n1p_tri", "1pl", "1sg"),
    ("ratio_2p_collective", "n2p_suc", "n2p_tri", "2pl_vy_true_plural", "2sg"),
    ("ratio_overall_plural", "nov_suc", "nov_tri", "nov_suc", None),
)


def _interaction_author_term_label(param_name: str) -> str | None:
    marker = "C(author)[T."
    suf = "]:C(period3, Treatment('" + PERIOD_P1 + "'))[T." + PERIOD_P2 + "]"
    s = str(param_name)
    if s.startswith(marker) and s.endswith(suf):
        return s[len(marker) : -len(suf)]
    return None


def _bootstrap_delta(sub: pd.DataFrame, *, n_bootstrap: int, seed: int) -> tuple[float, float, float, bool]:
    pre = sub.loc[sub["period3"].eq(PERIOD_P1)]
    post = sub.loc[sub["period3"].eq(PERIOD_P2)]
    if pre.empty or post.empty:
        return np.nan, np.nan, np.nan, False
    suc_pre = float(pre["suc"].sum())
    tri_pre = float(pre["tri"].sum())
    suc_post = float(post["suc"].sum())
    tri_post = float(post["tri"].sum())
    corr = False
    if (suc_pre <= 0) or (suc_pre >= tri_pre) or (suc_post <= 0) or (suc_post >= tri_post):
        corr = True
    p_pre = (suc_pre + SMOOTH_NUM) / (tri_pre + SMOOTH_DEN)
    p_post = (suc_post + SMOOTH_NUM) / (tri_post + SMOOTH_DEN)
    delta = float(np.log(p_post / (1 - p_post)) - np.log(p_pre / (1 - p_pre)))
    if n_bootstrap <= 0:
        return delta, np.nan, np.nan, corr
    rng = np.random.default_rng(seed)
    pre_suc = pre["suc"].to_numpy(dtype=float)
    pre_tri = pre["tri"].to_numpy(dtype=float)
    post_suc = post["suc"].to_numpy(dtype=float)
    post_tri = post["tri"].to_numpy(dtype=float)
    n_pre = len(pre_suc)
    n_post = len(post_suc)
    idx_pre = rng.integers(0, n_pre, size=(n_bootstrap, n_pre))
    idx_post = rng.integers(0, n_post, size=(n_bootstrap, n_post))
    b_suc_pre = pre_suc[idx_pre].sum(axis=1)
    b_tri_pre = pre_tri[idx_pre].sum(axis=1)
    b_suc_post = post_suc[idx_post].sum(axis=1)
    b_tri_post = post_tri[idx_post].sum(axis=1)
    ppre = (b_suc_pre + SMOOTH_NUM) / (b_tri_pre + SMOOTH_DEN)
    ppost = (b_suc_post + SMOOTH_NUM) / (b_tri_post + SMOOTH_DEN)
    d = np.log(ppost / (1 - ppost)) - np.log(ppre / (1 - ppre))
    return delta, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), corr


def main() -> None:
    ap = argparse.ArgumentParser(description="Q1b analog for ratio outcomes.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dat = pd.read_csv(args.input.resolve(), low_memory=False)
    dat = dat.loc[dat["period3"].isin(PERIODS)].copy()
    dat["author"] = dat["author"].astype(str)

    inter_rows: list[dict[str, object]] = []
    boot_rows: list[dict[str, object]] = []
    sparse_rows: list[dict[str, object]] = []
    for stratum in LANGUAGE_STRATA:
        sub = filter_poems_by_language_stratum(dat, stratum)
        is_primary = primary_stratum_for_bh(stratum)
        for ratio_index, suc_col, tri_col, success_cell, failure_cell in RATIO_SPECS:
            cdf = sub.copy()
            cdf["suc"] = pd.to_numeric(cdf[suc_col], errors="coerce").fillna(0.0)
            cdf["tri"] = pd.to_numeric(cdf[tri_col], errors="coerce").fillna(0.0)
            cdf = cdf.loc[cdf["tri"].gt(0)].copy()
            if cdf.empty:
                continue
            keep_authors: list[str] = []
            for author, adf in cdf.groupby("author"):
                a_pre = adf.loc[adf["period3"].eq(PERIOD_P1)]
                a_post = adf.loc[adf["period3"].eq(PERIOD_P2)]
                if a_pre.empty or a_post.empty:
                    sparse_rows.append(
                        {
                            "language_stratum": stratum,
                            "ratio_index": ratio_index,
                            "author": author,
                            "qualifies": False,
                            "reason": "missing_period",
                        }
                    )
                    continue
                if failure_cell is None:
                    has_pre_mix = a_pre["suc"].sum() > 0 and (a_pre["tri"].sum() - a_pre["suc"].sum()) > 0
                    has_post_mix = a_post["suc"].sum() > 0 and (a_post["tri"].sum() - a_post["suc"].sum()) > 0
                else:
                    has_pre_mix = (a_pre[success_cell] > 0).any() and (a_pre[failure_cell] > 0).any()
                    has_post_mix = (a_post[success_cell] > 0).any() and (a_post[failure_cell] > 0).any()
                qualifies = bool(has_pre_mix and has_post_mix)
                sparse_rows.append(
                    {
                        "language_stratum": stratum,
                        "ratio_index": ratio_index,
                        "author": author,
                        "qualifies": qualifies,
                        "reason": "ok" if qualifies else "no_success_failure_mix_both_periods",
                    }
                )
                if qualifies:
                    keep_authors.append(author)
            cdf = cdf.loc[cdf["author"].isin(keep_authors)].copy()
            if cdf["author"].nunique() < MIN_AUTHORS_PER_CELL_FIT:
                inter_rows.append(
                    {
                        "language_stratum": stratum,
                        "ratio_index": ratio_index,
                        "author": np.nan,
                        "n_authors_in_cell_fit": int(cdf["author"].nunique()),
                        "coef_log_odds": np.nan,
                        "OR_author_period": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "se_hc3": np.nan,
                        "p_value": np.nan,
                        "q_value_bh_within_stratum_ratio": np.nan,
                        "fit_status": "not_fit_min_authors",
                        "is_primary_stratum": bool(is_primary),
                    }
                )
                continue
            cdf["y"] = cdf["suc"] / cdf["tri"]
            fit = smf.glm(
                f"y ~ C(author) + C(author):C(period3, Treatment('{PERIOD_P1}'))",
                data=cdf,
                family=sm.families.Binomial(),
                freq_weights=cdf["tri"],
            ).fit(cov_type="HC3")
            n_auth_fit = int(cdf["author"].nunique())
            for pname, coef in fit.params.items():
                author = _interaction_author_term_label(pname)
                if author is None:
                    continue
                ci = fit.conf_int().loc[str(pname)]
                inter_rows.append(
                    {
                        "language_stratum": stratum,
                        "ratio_index": ratio_index,
                        "author": author,
                        "n_authors_in_cell_fit": n_auth_fit,
                        "coef_log_odds": float(coef),
                        "OR_author_period": float(np.exp(coef)),
                        "ci95_low": float(ci.iloc[0]),
                        "ci95_high": float(ci.iloc[1]),
                        "se_hc3": float(fit.bse[str(pname)]),
                        "p_value": float(fit.pvalues[str(pname)]),
                        "fit_status": "fit",
                        "is_primary_stratum": bool(is_primary),
                    }
                )
            for author, adf in cdf.groupby("author"):
                seed = int(hashlib.md5(f"{stratum}|{ratio_index}|{author}".encode("utf-8")).hexdigest()[:8], 16)
                delta, lo, hi, corr = _bootstrap_delta(adf, n_bootstrap=int(args.n_bootstrap), seed=seed)
                pre = adf.loc[adf["period3"].eq(PERIOD_P1)]
                post = adf.loc[adf["period3"].eq(PERIOD_P2)]
                boot_rows.append(
                    {
                        "language_stratum": stratum,
                        "ratio_index": ratio_index,
                        "author": author,
                        "n_pre": int(len(pre)),
                        "n_post": int(len(post)),
                        "trials_pre": float(pre["tri"].sum()),
                        "trials_post": float(post["tri"].sum()),
                        "delta_logodds": delta,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_bootstrap_resamples": int(args.n_bootstrap),
                        "continuity_corrected": bool(corr),
                    }
                )
    inter_df = pd.DataFrame(inter_rows)
    if not inter_df.empty and "p_value" in inter_df.columns:
        inter_df["q_value_bh_within_stratum_ratio"] = np.nan
        fit_sub = inter_df.loc[inter_df["fit_status"].eq("fit") & inter_df["is_primary_stratum"].astype(bool)]
        if not fit_sub.empty:
            q = fit_sub.groupby(["language_stratum", "ratio_index"], group_keys=False)["p_value"].apply(bh_adjust)
            inter_df.loc[q.index, "q_value_bh_within_stratum_ratio"] = q
        if "is_primary_stratum" in inter_df.columns:
            inter_df = inter_df.drop(columns=["is_primary_stratum"])
    inter_df.to_csv(out_dir / "ratio_q1b_author_fe_interactions.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out_dir / "ratio_q1b_delta_bootstrap.csv", index=False)
    pd.DataFrame(sparse_rows).to_csv(out_dir / "ratio_q1b_sparsity_audit.csv", index=False)
    print(f"Wrote ratio Q1b outputs to: {out_dir}")


if __name__ == "__main__":
    main()
