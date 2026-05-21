"""Conditional logistic regression on the closed four-cell attention-allocation design.

The 02a primary model is a mixed-effects binomial (author random intercept).
This module provides a *conditional* logistic-regression sensitivity check
that eliminates the author intercept by conditioning on the within-author
total successes (Chamberlain 1980 conditional MLE).

Inputs use the long four-cell table built by ``build_poem_long_4cells`` in
``02_modeling_significance_core_contrasts.py``: one row per ``(poem_id, cell)``
with ``k``, ``n``, integer ``person`` / ``number``, and ``period3``.

The conditional likelihood requires trial-level (Bernoulli) data, so each
``(poem_id, cell)`` row is expanded to ``k`` success trials and ``n - k``
failure trials. Trials inherit the author, period3, person, number features
of their parent row. Authors with only one observed value of the closed-set
sufficient statistic (e.g. all successes or all failures within their
stratum) are dropped by the conditional likelihood; the diagnostics table
records these drops.

Backend selection (post-P0-2 refactor)
--------------------------------------
The default backend is **R / ``survival::clogit``** via
:mod:`utils.r_clogit_runner`. The previous ``statsmodels`` backend
(``statsmodels.discrete.conditional_models.ConditionalLogit``) reliably
failed with ``RecursionError: maximum recursion depth exceeded in
comparison`` at the corpus's stratum sizes — even after raising
``sys.setrecursionlimit`` to 50 000 — because its Python-recursive partition
function recurses linearly in stratum size and the ~1000 trials per author
overflow the C-stack on top of the Python recursion budget. The R backend
implements the same Chamberlain estimator via ``coxph``'s C partial-likelihood
machinery and is unaffected by stratum size.

The statsmodels backend is kept behind ``backend="statsmodels"`` for forensic
reproduction of the prior failure mode only; nothing should depend on it.
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from patsy import dmatrix

from utils.r_clogit_runner import (
    RClogitEnvironmentError,
    RClogitFitError,
    fit_clogit,
)

log = logging.getLogger(__name__)


# Right-hand side passed to either backend. No-intercept ``-1`` is harmless under
# clogit (the intercept is absorbed into the strata) and is retained so the
# patsy design matrix used by the statsmodels fallback matches the R design.
CONDITIONAL_LOGIT_FORMULA = (
    "-1 + person + number + period3 + person:number + person:period3 + number:period3 + person:number:period3"
)


# Term-name normalization for R coef labels. survival::clogit names integer
# predictors as bare names and factor levels as ``factor[T.Level]``. We map both
# styles back to the patsy / statsmodels-style names that the rest of the 02a
# pipeline already canonicalizes via ``_clogit_term_canonical``.
_R_TO_PATSY_TERM = {
    "period3P2_2022_plus": "period3[T.P2_2022_plus]",
    "person:period3P2_2022_plus": "person:period3[T.P2_2022_plus]",
    "number:period3P2_2022_plus": "number:period3[T.P2_2022_plus]",
    "person:number:period3P2_2022_plus": "person:number:period3[T.P2_2022_plus]",
}


@dataclass(frozen=True)
class ConditionalLogitResult:
    """Result envelope used by ``02_modeling_significance_core_contrasts.py``."""

    params: pd.Series
    cov: pd.DataFrame
    n_trials: int
    n_authors_used: int
    n_authors_dropped: int
    dropped_authors: list[str]
    convergence_status: str

    def cov_params(self) -> pd.DataFrame:
        return self.cov


def expand_to_trials(long_df: pd.DataFrame) -> pd.DataFrame:
    """Expand a ``(k, n)`` row into ``k`` success trials and ``n - k`` failures."""
    needed = {"k", "n", "author", "period3", "person", "number"}
    missing = sorted(needed - set(long_df.columns))
    if missing:
        raise ValueError(f"expand_to_trials: missing columns {missing}")
    df = long_df.copy()
    df["k"] = df["k"].astype(int).clip(lower=0)
    df["n"] = df["n"].astype(int).clip(lower=0)
    df = df.loc[df["n"] > 0].copy()
    if df.empty:
        return df.assign(y=pd.Series(dtype=int))

    success_idx = df.index.repeat(df["k"].to_numpy())
    failure_idx = df.index.repeat((df["n"] - df["k"]).to_numpy())
    base = df.loc[:, ["author", "period3", "person", "number"]]
    succ = base.loc[success_idx].assign(y=1)
    fail = base.loc[failure_idx].assign(y=0)
    out = pd.concat([succ, fail], ignore_index=True)
    out["author"] = out["author"].astype(str)
    out["period3"] = out["period3"].astype(str)
    out["person"] = out["person"].astype(int)
    out["number"] = out["number"].astype(int)
    return out


def _filter_authors_with_variation(trials: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop authors with no within-author y-variation. Such strata contribute zero to
    the conditional likelihood; we drop them upstream so the diagnostics report
    matches the strata actually used by the backend."""
    dropped: list[str] = []
    keep_authors = []
    for author, g in trials.groupby("author"):
        if g["y"].nunique() < 2:
            dropped.append(str(author))
            continue
        keep_authors.append(author)
    return trials[trials["author"].isin(keep_authors)].copy(), dropped


def _r_predictors_formula() -> str:
    """Strip the leading ``-1`` (no-intercept) from the patsy formula; survival::clogit
    has no intercept by construction (strata absorb it) and rejects the ``-1`` token
    when expressed against R's formula grammar."""
    rhs = CONDITIONAL_LOGIT_FORMULA
    if rhs.startswith("-1 + "):
        rhs = rhs[len("-1 + ") :]
    return rhs


def _rename_r_terms(params: pd.Series, cov: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Map ``survival::clogit`` coefficient names to the patsy-style names already
    canonicalized by :func:`_clogit_term_canonical` in 02a."""
    rename_map = {term: _R_TO_PATSY_TERM.get(term, term) for term in params.index}
    new_params = params.rename(index=rename_map)
    new_cov = cov.rename(index=rename_map, columns=rename_map)
    return new_params, new_cov


def _fit_via_r(trials: pd.DataFrame, dropped: list[str]) -> ConditionalLogitResult:
    """Call the R backend and wrap the result in the project's envelope."""
    try:
        r_result = fit_clogit(trials, predictors_formula=_r_predictors_formula())
    except RClogitEnvironmentError as env_err:
        log.warning("R clogit environment unavailable: %s", env_err)
        return ConditionalLogitResult(
            params=pd.Series(dtype=float),
            cov=pd.DataFrame(dtype=float),
            n_trials=int(len(trials)),
            n_authors_used=int(trials["author"].nunique()),
            n_authors_dropped=len(dropped),
            dropped_authors=dropped,
            convergence_status=f"environment_error: {env_err}",
        )
    except RClogitFitError as fit_err:
        log.warning("R clogit fit error: %s", fit_err)
        return ConditionalLogitResult(
            params=pd.Series(dtype=float),
            cov=pd.DataFrame(dtype=float),
            n_trials=int(len(trials)),
            n_authors_used=int(trials["author"].nunique()),
            n_authors_dropped=len(dropped),
            dropped_authors=dropped,
            convergence_status=f"fit_error: {fit_err}",
        )

    params, cov = _rename_r_terms(r_result.params, r_result.cov)
    n_authors_used = int(r_result.n_strata_informative or r_result.n_strata)
    status = "converged"
    if r_result.convergence_message:
        status = f"converged_with_message: {r_result.convergence_message}"
    if not np.isfinite(r_result.log_likelihood):
        # clogit returned parameters but the partial likelihood is non-finite —
        # the fit is degenerate (separation, perfect collinearity, or near-zero
        # within-stratum information). Treat as a failure so downstream consumers
        # do not propagate untrustworthy params.
        return ConditionalLogitResult(
            params=pd.Series(dtype=float),
            cov=pd.DataFrame(dtype=float),
            n_trials=int(r_result.n_obs),
            n_authors_used=n_authors_used,
            n_authors_dropped=len(dropped),
            dropped_authors=dropped,
            convergence_status=(
                f"fit_error: degenerate clogit (log_likelihood={r_result.log_likelihood}); "
                "likely within-author separation in the trial-level expansion of the four-cell "
                "design. Use the lme4 GLMM as the primary engine for this stratum."
            ),
        )

    return ConditionalLogitResult(
        params=params,
        cov=cov,
        n_trials=int(r_result.n_obs),
        n_authors_used=n_authors_used,
        n_authors_dropped=len(dropped),
        dropped_authors=dropped,
        convergence_status=status,
    )


def _fit_via_statsmodels(trials: pd.DataFrame, dropped: list[str]) -> ConditionalLogitResult:
    """Forensic fallback: the recursion-limited statsmodels path the user
    previously hit. Kept for backend comparison only; production runs should use
    the R backend."""
    from statsmodels.discrete.conditional_models import ConditionalLogit

    _RECURSION_LIMIT_OVERRIDE = 50_000

    trials = trials.copy()
    trials["period3"] = pd.Categorical(
        trials["period3"], categories=["P1_2014_2021", "P2_2022_plus"], ordered=False
    )
    trials = trials.sort_values("author", kind="mergesort").reset_index(drop=True)

    design = dmatrix(CONDITIONAL_LOGIT_FORMULA, data=trials, return_type="dataframe")
    endog = trials["y"].to_numpy(dtype=int)
    groups = trials["author"].astype(str).to_numpy()

    original_recursion_limit = sys.getrecursionlimit()
    largest_stratum = int(pd.Series(groups).value_counts().max()) if len(groups) else 0
    target_limit = max(original_recursion_limit, largest_stratum * 4 + 5_000, _RECURSION_LIMIT_OVERRIDE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sys.setrecursionlimit(target_limit)
            fit = ConditionalLogit(endog, design, groups=groups).fit(disp=False, method="bfgs")
            status = "converged"
        except Exception as exc:  # noqa: BLE001
            return ConditionalLogitResult(
                params=pd.Series(dtype=float),
                cov=pd.DataFrame(dtype=float),
                n_trials=int(len(trials)),
                n_authors_used=int(pd.Series(groups).nunique()),
                n_authors_dropped=len(dropped),
                dropped_authors=dropped,
                convergence_status=f"fit_error: {exc}",
            )
        finally:
            sys.setrecursionlimit(original_recursion_limit)

    params = pd.Series(fit.params, index=design.columns).astype(float)
    cov = pd.DataFrame(fit.cov_params(), index=design.columns, columns=design.columns).astype(float)
    return ConditionalLogitResult(
        params=params,
        cov=cov,
        n_trials=int(len(trials)),
        n_authors_used=int(pd.Series(groups).nunique()),
        n_authors_dropped=len(dropped),
        dropped_authors=dropped,
        convergence_status=status,
    )


def fit_conditional_logit(
    long_df: pd.DataFrame,
    *,
    backend: Literal["r", "statsmodels"] = "r",
) -> ConditionalLogitResult:
    """Fit Chamberlain conditional logit on the expanded trial-level data.

    Parameters
    ----------
    long_df:
        Per-poem-cell long table. Required columns: ``k``, ``n``, ``author``,
        ``period3``, ``person``, ``number``.
    backend:
        ``"r"`` (default) routes through :mod:`utils.r_clogit_runner`. The
        ``"statsmodels"`` path is retained as a forensic option only; it
        recurses past Python's stack on this corpus.
    """
    trials = expand_to_trials(long_df)
    trials, dropped = _filter_authors_with_variation(trials)
    if trials.empty:
        empty_idx = pd.Index([], name="term")
        return ConditionalLogitResult(
            params=pd.Series(dtype=float, index=empty_idx),
            cov=pd.DataFrame(index=empty_idx, columns=empty_idx, dtype=float),
            n_trials=0,
            n_authors_used=0,
            n_authors_dropped=len(dropped),
            dropped_authors=dropped,
            convergence_status="no_data_after_filter",
        )

    if backend == "r":
        return _fit_via_r(trials, dropped)
    if backend == "statsmodels":
        return _fit_via_statsmodels(trials, dropped)
    raise ValueError(f"Unknown backend: {backend!r}")
