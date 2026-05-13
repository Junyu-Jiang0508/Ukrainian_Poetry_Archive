"""Conditional logistic regression on the closed four-cell attention-allocation design.

The 02a primary model is a mixed-effects binomial (author random intercept).
This module provides a *conditional* logistic-regression sensitivity check
that eliminates the author intercept by conditioning on the within-author
total successes (Cox's conditional likelihood, ``statsmodels.discrete.
conditional_models.ConditionalLogit``).

Inputs use the long four-cell table built by ``build_poem_long_4cells`` in
``02_modeling_significance_core_contrasts.py``: one row per ``(poem_id, cell)``
with ``k``, ``n``, integer ``person`` / ``number``, and ``period3``.

The conditional likelihood requires trial-level (Bernoulli) data, so each
``(poem_id, cell)`` row is expanded to ``k`` success trials and ``n - k``
failure trials. Trials inherit the author, period3, person, number features
of their parent row. Authors with only one observed value of the closed-set
sufficient statistic (e.g. all successes or all failures within their
stratum, or no within-author variation in any predictor) are dropped by the
conditional likelihood; the diagnostics table records these drops.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.discrete.conditional_models import ConditionalLogit


# statsmodels' ConditionalLogit was designed for small matched-case-control strata
# (typically 1:N with N <= 10). When applied to author-level grouping with the
# ~1000 expanded Bernoulli trials per stratum that arise from this corpus, the
# internal partition-function computation recurses linearly in stratum size and
# triggers ``RecursionError: maximum recursion depth exceeded in comparison`` at
# Python's default limit of 1000. We pre-emptively raise the limit during the
# ``.fit()`` call and restore it afterwards.
_RECURSION_LIMIT_OVERRIDE = 50_000


CONDITIONAL_LOGIT_FORMULA = (
    "-1 + person + number + period3 + person:number + person:period3 + number:period3 + person:number:period3"
)


@dataclass(frozen=True)
class ConditionalLogitResult:
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
    """Expand a (k, n) row into k success Bernoulli trials and n-k failure trials."""
    needed = {"k", "n", "author", "period3", "person", "number"}
    missing = sorted(needed - set(long_df.columns))
    if missing:
        raise ValueError(f"expand_to_trials: missing columns {missing}")
    df = long_df.copy()
    df["k"] = df["k"].astype(int).clip(lower=0)
    df["n"] = df["n"].astype(int).clip(lower=0)
    # Drop empty rows.
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
    """Drop authors with no within-author variation in y or with degenerate strata."""
    dropped: list[str] = []
    by_author = trials.groupby("author")
    keep_authors = []
    for author, g in by_author:
        if g["y"].nunique() < 2:
            dropped.append(str(author))
            continue
        keep_authors.append(author)
    return trials[trials["author"].isin(keep_authors)].copy(), dropped


def fit_conditional_logit(long_df: pd.DataFrame) -> ConditionalLogitResult:
    """Fit the conditional logistic regression on the expanded trial-level data."""
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

    # period3 encoded with P1_2014_2021 as the reference; unordered categorical so
    # patsy uses Treatment contrasts (term name ``period3[T.P2_2022_plus]``) rather
    # than the polynomial ``period3.L`` coding implied by an ordered factor.
    trials["period3"] = pd.Categorical(
        trials["period3"], categories=["P1_2014_2021", "P2_2022_plus"], ordered=False
    )

    # Sort by author so the within-stratum slices statsmodels iterates over are
    # contiguous; this also makes the recursion depth deterministic for any
    # given stratum size.
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
