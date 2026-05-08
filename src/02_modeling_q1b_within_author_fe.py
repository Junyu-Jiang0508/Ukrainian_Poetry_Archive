"""Q1b: within-author FE Poisson for author × period interactions (cultural-analytics forests)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum
from utils.finite_verb_exposure import resolve_finite_verb_counts_for_modeling
from utils.poem_cell_counts import build_poem_cell_table_with_exposure
from utils.pronoun_encoding import PRIMARY_GLM_CELLS
from utils.stats_common import bh_adjust
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_q1b_within_author_fe"

PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"
PERIODS = (PERIOD_P1, PERIOD_P2)


def _load_q1_helpers():
    path = ROOT / "src" / "02_modeling_q1_per_cell_glm.py"
    spec = importlib.util.spec_from_file_location("_q1_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_roster_authors(roster_path: Path | None) -> set[str] | None:
    if roster_path is None or not roster_path.is_file():
        return None
    r = pd.read_csv(roster_path, low_memory=False)
    if "included" not in r.columns or "author" not in r.columns:
        return None
    return set(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def _interaction_author_term_label(param_name: str) -> str | None:
    """Parse author slug from interaction term ``C(author)[T.NAME]:C(period3, …)[T.P2]``."""
    marker = "C(author)[T."
    suf = "]:C(period3, Treatment('" + PERIOD_P1 + "'))[T." + PERIOD_P2 + "]"
    pn = str(param_name)
    if not pn.startswith(marker) or not pn.endswith(suf):
        return None
    mid = pn[len(marker) : -len(suf)]
    return mid


# Pre-registration threshold: a cell is fit only if at least this many authors
# contribute ≥1 event in BOTH P1 and P2 to that cell. 5 (not 3) is the threshold
# we can defend in Methods §X — at 3 authors the FE design has 1 residual df per
# coefficient and HC1 SE is too noisy to support q-values. Russian × true-plural
# (4 qualifying authors) deliberately falls below this and produces no rows.
MIN_AUTHORS_PER_CELL_FIT = 5


def _fit_within_author_fe_per_cell(
    poem_df: pd.DataFrame,
    roster_authors: set[str] | None,
    *,
    language_stratum: str,
    cell: str,
    exposure_col: str = "exposure_n_stanzas",
) -> pd.DataFrame:
    """Parametric Q1b: Poisson FE with author×period interactions, HC3 SEs.

    Identifiability rule (strict per-cell): an author enters the fit only if they
    have ≥1 event of THIS cell in BOTH P1 and P2 — not merely ≥1 poem in each.
    The previous "any poem in both periods" rule produced separation when an
    author had 0 events in one period (MLE = ±∞) and IRLS reported coefs in the
    1e+10 / 1e-19 range with cluster-robust covariance numerically rank-deficient
    (cluster-on-FE-dimension collapses the meat matrix). HC3 is used instead of
    cluster-on-author because clustering on the FE dimension is degenerate, and
    HC3 is more conservative than HC1 in finite samples.
    """
    dat = poem_df.copy()
    if "include_in_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_offset_models"].astype(bool)].copy()
    dat = dat[dat["period3"].isin(PERIODS)]
    dat["author"] = dat["author"].astype(str).str.strip()
    dat = dat[dat["author"].ne("")]
    if roster_authors is not None:
        dat = dat[dat["author"].isin(roster_authors)]
    dat["_ex"] = dat[exposure_col].astype(float)
    dat = dat[dat["_ex"].gt(0) & np.isfinite(dat["_ex"])].copy()
    if dat.empty or dat["period3"].nunique() < 2:
        return pd.DataFrame()

    dat["log_exposure"] = np.log(dat["_ex"])
    dat["k"] = dat[cell].astype(int)

    per_author_period_k = dat.groupby(["author", "period3"])["k"].sum().unstack(fill_value=0)
    for col in PERIODS:
        if col not in per_author_period_k.columns:
            per_author_period_k[col] = 0
    both_pos = per_author_period_k[list(PERIODS)].gt(0).all(axis=1)
    keep_authors = set(both_pos[both_pos].index.astype(str).tolist())
    dat = dat.loc[dat["author"].isin(keep_authors)].copy()
    if dat["author"].nunique() < MIN_AUTHORS_PER_CELL_FIT or dat["period3"].nunique() < 2:
        log.info(
            "Q1b parametric: skip stratum=%s cell=%s (only %d author(s) with events in both periods; threshold=%d)",
            language_stratum,
            cell,
            int(dat["author"].nunique()),
            MIN_AUTHORS_PER_CELL_FIT,
        )
        return pd.DataFrame()

    formula = f"k ~ C(author) * C(period3, Treatment('{PERIOD_P1}'))"
    model = smf.glm(formula, data=dat, family=sm.families.Poisson(), offset=dat["log_exposure"])
    fit = None
    last_exc: Exception | None = None
    fit_attempts = (
        {"method": "irls", "maxiter": 200},
        {"method": "newton", "maxiter": 200, "disp": 0},
    )
    for attempt in fit_attempts:
        try:
            fit = model.fit(cov_type="HC3", **attempt)
            break
        except Exception as exc:  # numerical failures are still possible after filter
            last_exc = exc
            continue
    if fit is None:
        log.warning(
            "Q1b FE fit skipped for stratum=%s cell=%s due to numerical failure: %s",
            language_stratum,
            cell,
            last_exc,
        )
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    n_authors_in_fit = int(dat["author"].nunique())
    for pname, coef in fit.params.items():
        author_raw = _interaction_author_term_label(pname)
        if author_raw is None:
            continue
        pname_str = str(pname)
        ci = fit.conf_int().loc[pname_str]
        rows.append(
            {
                "language_stratum": language_stratum,
                "cell": cell,
                "author": author_raw,
                "n_authors_in_cell_fit": n_authors_in_fit,
                "interaction_coef_log_mu": float(coef),
                "interaction_rate_ratio": float(np.exp(coef)),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "se_hc3": float(fit.bse[pname_str]),
                "covariance_type": "HC3",
                "p_value": float(fit.pvalues[pname_str]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh_within_stratum_cell"] = out.groupby(["language_stratum", "cell"], group_keys=False)[
        "p_value"
    ].apply(bh_adjust)
    return out


def _bootstrap_author_delta_per_cell(
    poem_df: pd.DataFrame,
    roster_authors: set[str] | None,
    *,
    language_stratum: str,
    cell: str,
    exposure_col: str = "exposure_n_stanzas",
    n_bootstrap: int = 1000,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Descriptive per-author δ_i = log((k_post+0.5)/exp_post) − log((k_pre+0.5)/exp_pre).

    For each (stratum, cell, author) where the author has ≥1 poem in BOTH P1 and
    P2, compute δ from observed counts and a 95% CI by within-author poem
    bootstrap (B=1000 by default). Resampling is by poem within author within
    period (poems resampled with replacement; authors NOT resampled). Laplace
    smoothing keeps δ finite when k_pre or k_post = 0.

    Independent of the parametric MLE path; survives separation by construction.
    Output columns:
        language_stratum, cell, author, k_pre, k_post, exposure_pre, exposure_post,
        delta_log_rate, ci95_low, ci95_high, n_bootstrap_resamples
    """
    dat = poem_df.copy()
    if "include_in_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_offset_models"].astype(bool)].copy()
    dat = dat[dat["period3"].isin(PERIODS)]
    dat["author"] = dat["author"].astype(str).str.strip()
    dat = dat[dat["author"].ne("")]
    if roster_authors is not None:
        dat = dat[dat["author"].isin(roster_authors)]
    dat["_ex"] = dat[exposure_col].astype(float)
    dat = dat[dat["_ex"].gt(0) & np.isfinite(dat["_ex"])].copy()
    if dat.empty or dat["period3"].nunique() < 2:
        return pd.DataFrame()

    dat["k"] = dat[cell].astype(int)

    rows: list[dict[str, object]] = []
    for author, sub in dat.groupby("author"):
        pre = sub.loc[sub["period3"].eq(PERIOD_P1)]
        post = sub.loc[sub["period3"].eq(PERIOD_P2)]
        if pre.empty or post.empty:
            continue

        k_pre = int(pre["k"].sum())
        k_post = int(post["k"].sum())
        exp_pre = float(pre["_ex"].sum())
        exp_post = float(post["_ex"].sum())
        if exp_pre <= 0 or exp_post <= 0:
            continue

        delta = float(
            np.log((k_post + smoothing) / exp_post) - np.log((k_pre + smoothing) / exp_pre)
        )

        ci_low = np.nan
        ci_high = np.nan
        if n_bootstrap > 0:
            # Deterministic seed across runs and Python invocations. Python's
            # built-in hash() is salted by PYTHONHASHSEED, which would change
            # bootstrap CIs on rerun and look like result laundering.
            digest = hashlib.md5(
                f"{language_stratum}|{cell}|{author}".encode("utf-8")
            ).hexdigest()
            seed = int(digest[:8], 16)
            rng = np.random.default_rng(seed)
            pre_k = pre["k"].to_numpy(dtype=np.int64)
            pre_ex = pre["_ex"].to_numpy(dtype=np.float64)
            post_k = post["k"].to_numpy(dtype=np.int64)
            post_ex = post["_ex"].to_numpy(dtype=np.float64)
            n_pre = pre_k.shape[0]
            n_post = post_k.shape[0]

            idx_pre = rng.integers(0, n_pre, size=(n_bootstrap, n_pre))
            idx_post = rng.integers(0, n_post, size=(n_bootstrap, n_post))
            kb_pre = pre_k[idx_pre].sum(axis=1).astype(np.float64)
            eb_pre = pre_ex[idx_pre].sum(axis=1)
            kb_post = post_k[idx_post].sum(axis=1).astype(np.float64)
            eb_post = post_ex[idx_post].sum(axis=1)

            # Defensive: any bootstrap with zero exposure on either side is skipped
            valid = (eb_pre > 0) & (eb_post > 0)
            if not valid.any():
                ci_low = np.nan
                ci_high = np.nan
            else:
                d = np.log((kb_post[valid] + smoothing) / eb_post[valid]) - np.log(
                    (kb_pre[valid] + smoothing) / eb_pre[valid]
                )
                ci_low = float(np.percentile(d, 2.5))
                ci_high = float(np.percentile(d, 97.5))

        rows.append(
            {
                "language_stratum": language_stratum,
                "cell": cell,
                "author": str(author),
                "k_pre": k_pre,
                "k_post": k_post,
                "exposure_pre": exp_pre,
                "exposure_post": exp_post,
                "delta_log_rate": delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_bootstrap_resamples": int(n_bootstrap),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Q1b Poisson FE with author × period interactions.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    ap.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--poem-table",
        type=Path,
        default=None,
        help="Optional precomputed q1_poem_unit_cell_counts_12.csv (skips annotation rebuild).",
    )
    ap.add_argument("--strata", type=str, default="Ukrainian,Russian", help="Comma-separated LANGUAGE_STRATA subset")
    ap.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip the descriptive per-author δ bootstrap (parametric FE only).",
    )
    ap.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap iterations per (stratum, cell, author) for the descriptive δ CI (default 1000).",
    )
    args = ap.parse_args()

    want_strata = tuple(s.strip() for s in args.strata.split(",") if s.strip())
    for s in want_strata:
        if s not in LANGUAGE_STRATA:
            raise SystemExit(f"Unknown stratum {s!r}. Choose from {LANGUAGE_STRATA}")
        if s == "pooled_Ukrainian_Russian":
            log.warning("Q1b default skips pooled; pooled FE is unusually large.")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = load_roster_authors(args.roster.resolve() if args.roster else None)

    if args.poem_table is not None and args.poem_table.is_file():
        poem_full = pd.read_csv(args.poem_table.resolve(), low_memory=False)
        poem_full["poem_id"] = poem_full["poem_id"].astype(str).str.strip()
    else:
        q1 = _load_q1_helpers()
        filt = q1.load_and_filter(
            args.input.resolve(),
            args.layer0.resolve() if args.layer0 else None,
            language_audit_dir=out_dir / "language_stratum_audit",
        )
        fv_df = resolve_finite_verb_counts_for_modeling(ROOT, exposure_type="n_stanzas")
        poem_full = build_poem_cell_table_with_exposure(filt, finite_verb_df=fv_df)

    parts: list[pd.DataFrame] = []
    boot_parts: list[pd.DataFrame] = []
    for stratum in want_strata:
        poem_sub = filter_poems_by_language_stratum(poem_full, stratum)
        for cell in PRIMARY_GLM_CELLS:
            fe = _fit_within_author_fe_per_cell(
                poem_sub,
                roster,
                language_stratum=stratum,
                cell=cell,
            )
            if not fe.empty:
                parts.append(fe)

            if not args.no_bootstrap:
                bdf = _bootstrap_author_delta_per_cell(
                    poem_sub,
                    roster,
                    language_stratum=stratum,
                    cell=cell,
                    n_bootstrap=int(args.n_bootstrap),
                )
                if not bdf.empty:
                    boot_parts.append(bdf)

    out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out_path = out_dir / "q1b_within_author_fe_interactions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")

    if not args.no_bootstrap:
        boot_df = pd.concat(boot_parts, ignore_index=True) if boot_parts else pd.DataFrame()
        boot_path = out_dir / "q1b_within_author_delta_bootstrap.csv"
        boot_df.to_csv(boot_path, index=False)
        print(f"Wrote {boot_path} ({len(boot_df)} rows)")


if __name__ == "__main__":
    main()