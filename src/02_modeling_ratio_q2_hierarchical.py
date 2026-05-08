"""Hierarchical logistic models for pronoun ratio indices (Q2 analog)."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum
from utils.stats_common import bh_adjust
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "outputs" / "02_modeling_ratio_indices" / "ratio_poem_level_table.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_ratio_q2_hierarchical"
PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"

RATIO_SPECS = (
    ("ratio_1p_collective", "n1p_suc", "n1p_tri"),
    ("ratio_2p_collective", "n2p_suc", "n2p_tri"),
    ("ratio_overall_plural", "nov_suc", "nov_tri"),
)


def _detect_random_slope_var(posterior, term: str, group: str) -> str:
    cands = [x for x in posterior.data_vars if (term in x and "|" in x and group in x)]
    if not cands:
        raise KeyError(f"random slope var not found for {term}|{group}")
    exact = [x for x in cands if x == f"{term}|{group}"]
    return exact[0] if exact else cands[0]


def _author_dim_name(da) -> str:
    for dim in da.dims:
        if dim not in ("chain", "draw", "__obs__"):
            return dim
    raise KeyError("author dimension not found")


def _plot_caterpillar(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    d = df.sort_values("dev_mean_log_odds", ascending=True).copy()
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.28 * len(d) + 2.0)))
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.hlines(y, d["dev_hdi_low"], d["dev_hdi_high"], color="#4c78a8", linewidth=1.4)
    ax.plot(d["dev_mean_log_odds"], y, "o", color="#1a1a1a", markersize=3.2)
    ax.set_yticks(y)
    ax.set_yticklabels(d["author"].astype(str), fontsize=8)
    ax.set_xlabel("Author deviation from population period shift (log-odds)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hierarchical binomial ratio models via Bambi.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--random-seed", type=int, default=20260508)
    args = ap.parse_args()

    try:
        import bambi as bmb
    except ImportError as exc:
        raise ImportError("Bambi is required for ratio Q2 stage. Install with `pip install bambi`.") from exc

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dat = pd.read_csv(args.input.resolve(), low_memory=False)
    dat = dat.loc[dat["period3"].isin((PERIOD_P1, PERIOD_P2))].copy()
    dat["author"] = dat["author"].astype(str)
    dat["period_post"] = dat["period3"].eq(PERIOD_P2).astype(int)

    pop_rows: list[dict[str, object]] = []
    author_rows: list[dict[str, object]] = []
    for i, stratum in enumerate(LANGUAGE_STRATA):
        sub = filter_poems_by_language_stratum(dat, stratum)
        for j, (ratio_index, suc_col, tri_col) in enumerate(RATIO_SPECS):
            cdf = sub.copy()
            cdf["suc"] = pd.to_numeric(cdf[suc_col], errors="coerce").fillna(0.0)
            cdf["tri"] = pd.to_numeric(cdf[tri_col], errors="coerce").fillna(0.0)
            cdf = cdf.loc[cdf["tri"].gt(0)].copy()
            if cdf.empty:
                continue
            valid_authors = cdf.groupby(["author", "period3"]).size().unstack(fill_value=0)
            good_auth = valid_authors.index[(valid_authors >= 2).all(axis=1)]
            cdf = cdf.loc[cdf["author"].isin(good_auth)].copy()
            if cdf.empty or cdf["author"].nunique() < 2 or cdf["period_post"].nunique() < 2:
                continue
            seed = int(args.random_seed) + i * 10_007 + j * 97
            used_slope = False
            idata = None
            try:
                m = bmb.Model("suc | trials(tri) ~ period_post + (1 + period_post | author)", data=cdf, family="binomial")
                prior_slope = bmb.Prior("Normal", mu=0.0, sigma=bmb.Prior("HalfNormal", sigma=1.0))
                m.set_priors(priors={"period_post|author": prior_slope})
                idata = m.fit(
                    draws=int(args.draws),
                    tune=int(args.tune),
                    chains=int(args.chains),
                    cores=int(args.cores),
                    target_accept=float(args.target_accept),
                    random_seed=seed,
                )
                used_slope = True
            except Exception as exc:
                warnings.warn(f"Random slope failed for {stratum}/{ratio_index}: {exc}; fallback intercept-only.")
                try:
                    m = bmb.Model("suc | trials(tri) ~ period_post + (1 | author)", data=cdf, family="binomial")
                    idata = m.fit(
                        draws=int(args.draws),
                        tune=int(args.tune),
                        chains=int(args.chains),
                        cores=int(args.cores),
                        target_accept=float(args.target_accept),
                        random_seed=seed,
                    )
                    used_slope = False
                except Exception as exc2:
                    warnings.warn(f"Fallback failed for {stratum}/{ratio_index}: {exc2}; skipping.")
                    continue
            post = idata.posterior
            if "period_post" not in post.data_vars:
                continue
            fixed = post["period_post"].stack(sample=("chain", "draw")).values
            mu = float(np.mean(fixed))
            lo = float(np.quantile(fixed, 0.025))
            hi = float(np.quantile(fixed, 0.975))
            p_gt0 = float(np.mean(fixed > 0.0))
            fsr = float(min(p_gt0, 1.0 - p_gt0))
            pop_rows.append(
                {
                    "language_stratum": stratum,
                    "ratio_index": ratio_index,
                    "model_spec": "random_slope_binomial" if used_slope else "random_intercept_binomial_fallback",
                    "n_poems": int(len(cdf)),
                    "n_authors": int(cdf["author"].nunique()),
                    "population_shift_mean_log_odds": mu,
                    "population_shift_OR_mean": float(np.exp(mu)),
                    "hdi_low": lo,
                    "hdi_high": hi,
                    "OR_hdi_low": float(np.exp(lo)),
                    "OR_hdi_high": float(np.exp(hi)),
                    "p_direction_gt0": p_gt0,
                    "false_sign_risk": fsr,
                }
            )
            if not used_slope:
                continue
            rs_var = _detect_random_slope_var(post, term="period_post", group="author")
            rs = post[rs_var]
            adim = _author_dim_name(rs)
            rs_s = rs.stack(sample=("chain", "draw"))
            for author in rs_s.coords[adim].values:
                dev = rs_s.sel({adim: author}).values
                total = fixed + dev
                dev_m = float(np.mean(dev))
                dev_lo = float(np.quantile(dev, 0.025))
                dev_hi = float(np.quantile(dev, 0.975))
                total_m = float(np.mean(total))
                total_lo = float(np.quantile(total, 0.025))
                total_hi = float(np.quantile(total, 0.975))
                total_p = float(np.mean(total > 0.0))
                total_fsr = float(min(total_p, 1.0 - total_p))
                author_rows.append(
                    {
                        "language_stratum": stratum,
                        "ratio_index": ratio_index,
                        "author": str(author),
                        "dev_mean_log_odds": dev_m,
                        "dev_hdi_low": dev_lo,
                        "dev_hdi_high": dev_hi,
                        "dev_false_sign_risk": float(min(np.mean(dev > 0.0), 1.0 - np.mean(dev > 0.0))),
                        "total_shift_mean_log_odds": total_m,
                        "total_shift_hdi_low": total_lo,
                        "total_shift_hdi_high": total_hi,
                        "total_shift_OR_mean": float(np.exp(total_m)),
                        "total_shift_OR_hdi_low": float(np.exp(total_lo)),
                        "total_shift_OR_hdi_high": float(np.exp(total_hi)),
                        "total_shift_p_direction_gt0": total_p,
                        "total_shift_false_sign_risk": total_fsr,
                    }
                )
    pop_df = pd.DataFrame(pop_rows)
    author_df = pd.DataFrame(author_rows)
    if not pop_df.empty:
        pop_df["q_direction"] = pop_df.groupby("language_stratum", group_keys=False)[
            "false_sign_risk"
        ].apply(bh_adjust)
    if not author_df.empty:
        author_df["total_shift_q_direction"] = author_df.groupby(["language_stratum", "ratio_index"], group_keys=False)[
            "total_shift_false_sign_risk"
        ].apply(bh_adjust)
        for (stratum, ratio_index), g in author_df.groupby(["language_stratum", "ratio_index"]):
            _plot_caterpillar(
                g,
                fig_dir / f"fig_ratio_q2_caterpillar_{ratio_index}_{stratum}.pdf",
                title=f"{ratio_index} — {stratum}",
            )
    pop_df.to_csv(out_dir / "ratio_q2_population_shifts.csv", index=False)
    author_df.to_csv(out_dir / "ratio_q2_author_random_slopes.csv", index=False)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Ratio Q2 hierarchical logistic models (02bratq2)\n\n")
        f.write("- Family: `binomial`, aggregated response `suc | trials(tri)`.\n")
        f.write("- Preferred model: `(1 + period_post | author)`, fallback `(1 | author)`.\n")
        f.write("- Random-slope prior aligned with Q2 NB: `Normal(0, HalfNormal(1.0))` on `period_post|author`.\n")
        f.write("- Population-level `q_direction` applies BH within each language stratum across ratio indices.\n")
    print(f"Wrote ratio Q2 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
