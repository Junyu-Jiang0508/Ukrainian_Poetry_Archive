"""Attention Allocation (closed first/second-person quartet) confirmatory contrasts.

Estimand
--------
This stage answers the **Attention Allocation** question: given any change in
absolute pronoun strength, how is attention reallocated **within** the closed
first/second-person four-cell sub-space ``{1sg, 1pl, 2sg, 2pl_vy_true_plural}``?
For each poem we form ``n_12 = sum(four cells)`` as the trial total and treat
each cell count as a binomial success. The closed denominator is the natural
sample space for the pragmatic-allocation question; it is **not** a
compositional-bias artifact, and it is intentionally distinct from the
**Absolute Salience** estimand answered by ``02_modeling_q1_per_cell_glm.py``
(per-cell Poisson / NB with ``log(exposure)`` offset).

Primary inference is **co-primary**:
* lme4 ``glmer`` binomial GLMM with random author intercept
  ``cbind(k, n - k) ~ person * number * period3 + (1 | author)``
  (via :func:`utils.r_glmm_runner.fit_glmer_binomial`).
* Cox conditional logistic regression
  (:func:`utils.conditional_logit_fit.fit_conditional_logit`),
  which eliminates the author intercept by conditioning on the within-author
  total successes.

Both engines avoid the incidental-parameter bias incurred by the unconditional
``+ C(author)`` MLE at our sample size (N ≈ 33 authors, median ~20–25
informative poems per author × cell × period). The legacy unconditional GLM is
retained but written into ``confirmatory_contrasts_main.csv`` only as a
sensitivity row with an explicit incidental-parameter caveat.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from utils.conditional_logit_fit import (
    ConditionalLogitResult,
    fit_conditional_logit,
)
from utils.pronoun_encoding import pronoun_class_sixway_column
from utils.r_glmm_runner import (
    RGlmmEnvironmentError,
    RGlmmFitError,
    RGlmmFitResult,
    fit_glmer_binomial,
)
from utils.stats_common import bh_adjust, mode_with_tie_order, normalize_bool_flag, period_three_way
from utils.workspace import prepare_analysis_environment

log = logging.getLogger(__name__)

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_significance_core_contrasts"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_ROSTER_GE8 = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_threshold_ge8.csv"

PERIODS = ["P1_2014_2021", "P2_2022_plus"]
CELL4 = ["1sg", "1pl", "2sg", "2pl"]
CELL6 = ["1sg", "1pl", "2sg", "2pl", "3sg", "3pl"]
QIRIMLI_CODES = {"Qirimli", "Russian, Qirimli", "Ukrainian, Qirimli"}
SWITCHERS = {"Iya Kiva", "Andrij Bondar", "Alex Averbuch", "Olena Boryshpolets"}
FORMULA = 'y ~ person * number * C(period3, Treatment("P1_2014_2021")) + C(author)'


def _mode_with_tie_order(series: pd.Series, preference: list[str]) -> str:
    return mode_with_tie_order(series, preference)


def load_and_filter(path: Path, layer0_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["poem_id"] = df["poem_id"].astype(str)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["period3"] = df["year_int"].map(period_three_way)
    df["person"] = df["person"].fillna("").str.strip()
    df["number"] = df["number"].fillna("").str.strip()
    df["person_number"] = pronoun_class_sixway_column(df)
    df["language_clean"] = df["language"].fillna("").str.strip()
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
    out = df.loc[~(df["is_repeat"] | df["is_translation"])].copy()
    out = out[~out["language_clean"].isin(QIRIMLI_CODES)].copy()
    return out


def build_poem_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    tok = df[df["pronoun_word"].notna()].copy()
    if tok.empty:
        return pd.DataFrame()
    ct = pd.crosstab(tok["poem_id"], tok["person_number"])
    for c in CELL6:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[CELL6].reset_index()
    ct["n_all6"] = ct[CELL6].sum(axis=1)
    ct["n_12"] = ct[CELL4].sum(axis=1)
    meta = df.groupby("poem_id", as_index=False).agg(
        author=("author", "first"),
        language_clean=("language_clean", "first"),
        year_int=("year_int", "first"),
    )
    out = ct.merge(meta, on="poem_id", how="left")
    out["period3"] = out["year_int"].map(period_three_way)
    return out


def load_roster_authors(roster_path: Path) -> list[str]:
    r = pd.read_csv(roster_path, low_memory=False)
    return sorted(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def build_poem_long_4cells(poem_cell: pd.DataFrame, min_n_12: int, roster_authors: list[str]) -> pd.DataFrame:
    core = poem_cell[
        poem_cell["period3"].isin(PERIODS)
        & (poem_cell["n_12"] >= int(min_n_12))
        & poem_cell["author"].isin(roster_authors)
    ].copy()
    if core.empty:
        return pd.DataFrame()
    core["period3"] = pd.Categorical(core["period3"], categories=PERIODS, ordered=True)
    rows: list[dict] = []
    for _, r in core.iterrows():
        denom = int(r["n_12"])
        if denom <= 0:
            continue
        for cell in CELL4:
            person = 1 if cell.startswith("1") else 0
            number = 1 if cell.endswith("pl") else 0
            k = int(r[cell])
            rows.append(
                {
                    "poem_id": r["poem_id"],
                    "author": r.get("author", ""),
                    "language_clean": r.get("language_clean", ""),
                    "year_int": r.get("year_int", np.nan),
                    "period3": r["period3"],
                    "cell": cell,
                    "person": person,
                    "number": number,
                    "k": k,
                    "n": denom,
                    "y": k / denom,
                }
            )
    return pd.DataFrame(rows)


CANONICAL_PERIOD = "period"
CANONICAL_PERSON_X_PERIOD = "person:period"
CANONICAL_NUMBER_X_PERIOD = "number:period"
CANONICAL_PXNXPERIOD = "person:number:period"


def _legacy_glm_term_canonical(raw: str) -> str | None:
    """Map a statsmodels ``smf.glm`` term name back to a canonical name.

    Returns ``None`` for any term that should be dropped from the contrast
    machinery (author fixed-effect dummies, intercept, main effects we don't
    test directly).
    """
    period_marker = 'C(period3, Treatment("P1_2014_2021"))[T.P2_2022_plus]'
    if raw == period_marker:
        return CANONICAL_PERIOD
    if raw == f"person:{period_marker}":
        return CANONICAL_PERSON_X_PERIOD
    if raw == f"number:{period_marker}":
        return CANONICAL_NUMBER_X_PERIOD
    if raw == f"person:number:{period_marker}":
        return CANONICAL_PXNXPERIOD
    return None


def _glmer_term_canonical(raw: str) -> str | None:
    """Map an lme4 ``glmer`` fixed-effect term name to canonical.

    lme4 with formula ``person * number * period3 + (1|author)`` and factor
    levels ``c("P1_2014_2021", "P2_2022_plus")`` emits term names like
    ``period3P2_2022_plus`` and ``person:period3P2_2022_plus``.
    """
    suf = "period3P2_2022_plus"
    if raw == suf:
        return CANONICAL_PERIOD
    if raw == f"person:{suf}":
        return CANONICAL_PERSON_X_PERIOD
    if raw == f"number:{suf}":
        return CANONICAL_NUMBER_X_PERIOD
    if raw == f"person:number:{suf}":
        return CANONICAL_PXNXPERIOD
    return None


def _clogit_term_canonical(raw: str) -> str | None:
    """Map a patsy / ConditionalLogit term name to canonical."""
    suf = "period3[T.P2_2022_plus]"
    if raw == suf:
        return CANONICAL_PERIOD
    if raw == f"person:{suf}":
        return CANONICAL_PERSON_X_PERIOD
    if raw == f"number:{suf}":
        return CANONICAL_NUMBER_X_PERIOD
    if raw == f"person:number:{suf}":
        return CANONICAL_PXNXPERIOD
    return None


_ENGINE_TO_CANONICALIZER = {
    "legacy_glm": _legacy_glm_term_canonical,
    "glmm": _glmer_term_canonical,
    "clogit": _clogit_term_canonical,
}


def _canonicalize_params_cov(
    params: pd.Series, cov: pd.DataFrame, engine: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Reduce ``params`` / ``cov`` to canonical contrast-relevant rows/cols."""
    if engine not in _ENGINE_TO_CANONICALIZER:
        raise ValueError(f"_canonicalize_params_cov: unknown engine {engine!r}")
    fn = _ENGINE_TO_CANONICALIZER[engine]
    rename: dict[str, str] = {}
    for raw in params.index.astype(str):
        canonical = fn(raw)
        if canonical is not None:
            rename[raw] = canonical
    if not rename:
        empty_idx = pd.Index([], name=params.index.name)
        return (
            pd.Series(dtype=float, index=empty_idx),
            pd.DataFrame(index=empty_idx, columns=empty_idx, dtype=float),
        )
    keep_raw = list(rename.keys())
    params_canon = params.loc[keep_raw].rename(rename).astype(float)
    cov_canon = (
        cov.loc[keep_raw, keep_raw]
        .rename(index=rename, columns=rename)
        .astype(float)
    )
    return params_canon, cov_canon


def _build_canonical_contrasts() -> list[tuple[str, dict[str, float]]]:
    """The three confirmatory contrasts expressed in canonical term names."""
    return [
        ("P2_vs_P1_2sg_cell_shift", {CANONICAL_PERIOD: 1.0, CANONICAL_PERSON_X_PERIOD: 1.0}),
        ("P2_vs_P1_1pl_cell_shift", {CANONICAL_PERIOD: 1.0, CANONICAL_NUMBER_X_PERIOD: 1.0}),
        ("P2_vs_P1_person_x_number", {CANONICAL_PXNXPERIOD: 1.0}),
    ]


def _build_contrast_specs(names: list[str]) -> list[tuple[str, np.ndarray]]:
    """Legacy entry-point: build contrast vectors aligned to raw statsmodels term names."""
    vec = {n: i for i, n in enumerate(names)}

    def _term(period: str, suffix: str = "") -> str:
        base = f'C(period3, Treatment("P1_2014_2021"))[T.{period}]'
        return base if not suffix else f"{suffix}:{base}"

    def _v(weights: dict[str, float]) -> np.ndarray:
        v = np.zeros(len(names), dtype=float)
        for k, w in weights.items():
            if k in vec:
                v[vec[k]] = float(w)
        return v

    tests = [
        ("P2_vs_P1_2sg_cell_shift", _v({_term("P2_2022_plus"): 1.0, _term("P2_2022_plus", "person"): 1.0})),
        ("P2_vs_P1_1pl_cell_shift", _v({_term("P2_2022_plus"): 1.0, _term("P2_2022_plus", "number"): 1.0})),
        ("P2_vs_P1_person_x_number", _v({_term("P2_2022_plus", "person:number"): 1.0})),
    ]
    return tests


def evaluate_contrasts_generic(
    params: pd.Series,
    cov: pd.DataFrame,
    long_df: pd.DataFrame,
    p_value_col: str = "p_value",
) -> pd.DataFrame:
    """Compute the three confirmatory contrasts from a canonicalized ``(params, cov)`` pair.

    ``params`` / ``cov`` must already be indexed by the canonical term names
    emitted by :func:`_canonicalize_params_cov`. Missing canonical terms result
    in NaN estimates rather than zeros so that engine-level dropouts (e.g. a
    contrast that the conditional likelihood absorbs) are visible in the table.
    """
    names = params.index.astype(str).tolist()
    name_pos = {n: i for i, n in enumerate(names)}
    beta = params.to_numpy(dtype=float)
    cov_mat = cov.to_numpy(dtype=float)
    rows: list[dict] = []
    for label, weights in _build_canonical_contrasts():
        v = np.zeros(len(names), dtype=float)
        all_present = True
        for term, w in weights.items():
            if term in name_pos:
                v[name_pos[term]] = float(w)
            else:
                all_present = False
        if not all_present or len(names) == 0:
            rows.append(
                {
                    "contrast": label,
                    "estimate_logit": np.nan,
                    "se": np.nan,
                    "z_value": np.nan,
                    p_value_col: np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "odds_ratio": np.nan,
                    "n_poems": int(long_df["poem_id"].nunique()) if "poem_id" in long_df else 0,
                    "n_rows_long": int(len(long_df)),
                }
            )
            continue
        est = float(np.dot(v, beta))
        var = float(np.dot(v, np.dot(cov_mat, v)))
        se = float(np.sqrt(max(0.0, var)))
        z = est / se if se > 0 else np.nan
        p = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        rows.append(
            {
                "contrast": label,
                "estimate_logit": est,
                "se": se,
                "z_value": z,
                p_value_col: p,
                "ci95_low": est - 1.96 * se if np.isfinite(se) else np.nan,
                "ci95_high": est + 1.96 * se if np.isfinite(se) else np.nan,
                "odds_ratio": float(np.exp(est)),
                "n_poems": int(long_df["poem_id"].nunique()) if "poem_id" in long_df else 0,
                "n_rows_long": int(len(long_df)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and p_value_col in out.columns:
        out[f"q_value_bh_{p_value_col}_family"] = bh_adjust(out[p_value_col])
    return out


def fit_glm(long_df: pd.DataFrame):
    return smf.glm(FORMULA, data=long_df, family=sm.families.Binomial(), freq_weights=long_df["n"]).fit()


def fit_glm_clustered_author(fit, long_df: pd.DataFrame):
    groups = long_df["author"].astype(str)
    if groups.nunique() < 2:
        return None
    try:
        return smf.glm(
            FORMULA,
            data=long_df,
            family=sm.families.Binomial(),
            freq_weights=long_df["n"],
        ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        return None


GLMM_FORMULA = "cbind(k, n - k) ~ person * number * period3 + (1 | author)"


def fit_glmm_primary(long_df: pd.DataFrame) -> tuple[RGlmmFitResult | None, dict]:
    """Fit the co-primary binomial GLMM with random author intercept.

    Returns the fit result (or ``None`` on env/fit failure) and a status dict
    describing convergence / availability for inclusion in
    ``coprimary_engine_status.json``.
    """
    needed = {"k", "n", "author", "period3", "person", "number"}
    missing = sorted(needed - set(long_df.columns))
    if missing:
        return None, {
            "engine": "glmm",
            "status": "missing_columns",
            "detail": f"missing columns: {missing}",
        }
    payload = long_df[["poem_id", "k", "n", "author", "period3", "person", "number"]].copy()
    payload["author"] = payload["author"].astype(str)
    payload["period3"] = payload["period3"].astype(str)
    try:
        result = fit_glmer_binomial(payload, GLMM_FORMULA)
        return result, {
            "engine": "glmm",
            "status": "fit",
            "convergence_message": result.convergence_message,
            "optimizer": result.optimizer,
            "n_obs": int(result.n_obs),
            "n_authors": int(result.n_authors),
            "random_intercept_sd_author": float(result.random_intercept_sd_author),
        }
    except RGlmmEnvironmentError as exc:
        log.warning("lme4 GLMM unavailable: %s", exc)
        return None, {"engine": "glmm", "status": "lme4_unavailable", "detail": str(exc)}
    except RGlmmFitError as exc:
        log.warning("lme4 glmer failed to converge: %s", exc)
        return None, {"engine": "glmm", "status": "fit_error", "detail": str(exc)}


def fit_clogit_primary(long_df: pd.DataFrame) -> tuple[ConditionalLogitResult | None, dict]:
    """Fit the Cox conditional logit on the four-cell long table."""
    needed = {"k", "n", "author", "period3", "person", "number"}
    missing = sorted(needed - set(long_df.columns))
    if missing:
        return None, {
            "engine": "clogit",
            "status": "missing_columns",
            "detail": f"missing columns: {missing}",
        }
    try:
        result = fit_conditional_logit(long_df[["k", "n", "author", "period3", "person", "number"]])
    except Exception as exc:  # noqa: BLE001
        log.warning("Conditional logit failed: %s", exc)
        return None, {"engine": "clogit", "status": "fit_error", "detail": str(exc)}
    return result, {
        "engine": "clogit",
        "status": result.convergence_status,
        "n_trials": int(result.n_trials),
        "n_authors_used": int(result.n_authors_used),
        "n_authors_dropped": int(result.n_authors_dropped),
    }


def _coprimary_rows_for_engine(
    params: pd.Series,
    cov: pd.DataFrame,
    long_df: pd.DataFrame,
    *,
    engine: str,
    raw_engine_label: str,
    n_authors_used: int,
    convergence_status: str,
) -> pd.DataFrame:
    """Produce the per-engine block of ``confirmatory_contrasts_coprimary.csv``."""
    params_canon, cov_canon = _canonicalize_params_cov(params, cov, engine)
    df = evaluate_contrasts_generic(params_canon, cov_canon, long_df, p_value_col="p_value")
    df = df.rename(columns={"q_value_bh_p_value_family": "q_value_bh_within_engine"})
    df["engine"] = raw_engine_label
    df["n_authors_used"] = int(n_authors_used)
    df["convergence_status"] = str(convergence_status)
    return df


def _coprimary_unavailable_block(
    *, long_df: pd.DataFrame, raw_engine_label: str, convergence_status: str
) -> pd.DataFrame:
    """Emit a NaN-filled placeholder block when an engine could not be fit."""
    placeholders = []
    for label, _weights in _build_canonical_contrasts():
        placeholders.append(
            {
                "contrast": label,
                "estimate_logit": np.nan,
                "se": np.nan,
                "z_value": np.nan,
                "p_value": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "odds_ratio": np.nan,
                "n_poems": int(long_df["poem_id"].nunique()) if "poem_id" in long_df else 0,
                "n_rows_long": int(len(long_df)),
                "q_value_bh_within_engine": np.nan,
                "engine": raw_engine_label,
                "n_authors_used": 0,
                "convergence_status": convergence_status,
            }
        )
    return pd.DataFrame(placeholders)


def evaluate_contrasts(fit, long_df: pd.DataFrame, p_value_col: str = "p_value") -> pd.DataFrame:
    names = fit.params.index.tolist()
    tests = _build_contrast_specs(names)
    cov = fit.cov_params().to_numpy(dtype=float)
    beta = fit.params.to_numpy(dtype=float)
    rows: list[dict] = []
    for label, v in tests:
        est = float(np.dot(v, beta))
        se = float(np.sqrt(max(0.0, np.dot(v, np.dot(cov, v)))))
        z = est / se if se > 0 else np.nan
        p = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        rows.append(
            {
                "contrast": label,
                "estimate_logit": est,
                "se": se,
                "z_value": z,
                p_value_col: p,
                "ci95_low": est - 1.96 * se if np.isfinite(se) else np.nan,
                "ci95_high": est + 1.96 * se if np.isfinite(se) else np.nan,
                "odds_ratio": float(np.exp(est)),
                "n_poems": int(long_df["poem_id"].nunique()),
                "n_rows_long": int(len(long_df)),
            }
        )
    out = pd.DataFrame(rows)
    out[f"q_value_bh_{p_value_col}_family"] = bh_adjust(out[p_value_col])
    return out


def wild_cluster_bootstrap(long_df: pd.DataFrame, fit, b_reps: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    names = fit.params.index.tolist()
    tests = _build_contrast_specs(names)
    obs = evaluate_contrasts(fit, long_df).set_index("contrast")
    resid = long_df["y"].to_numpy(dtype=float) - fit.fittedvalues.to_numpy(dtype=float)
    mu = fit.fittedvalues.to_numpy(dtype=float)
    authors = long_df["author"].astype(str).to_numpy()
    uniq = sorted(np.unique(authors).tolist())

    extreme = {name: 0 for name, _ in tests}
    valid = {name: 0 for name, _ in tests}
    t_obs = {name: float(obs.loc[name, "z_value"]) for name, _ in tests}
    start = time.time()

    for _ in range(int(b_reps)):
        w = {a: rng.choice([-1.0, 1.0]) for a in uniq}
        signs = np.array([w[a] for a in authors], dtype=float)
        y_star = np.clip(mu + signs * resid, 1e-5, 1 - 1e-5)
        bdat = long_df.copy()
        bdat["y"] = y_star
        try:
            bfit = fit_glm(bdat)
        except Exception:
            continue
        bnames = bfit.params.index.tolist()
        if bnames != names:
            continue
        bcov = bfit.cov_params().to_numpy(dtype=float)
        bbeta = bfit.params.to_numpy(dtype=float)
        for cname, v in tests:
            best = float(np.dot(v, bbeta))
            bse = float(np.sqrt(max(0.0, np.dot(v, np.dot(bcov, v)))))
            bt = best / bse if bse > 0 else np.nan
            if np.isfinite(bt) and np.isfinite(t_obs[cname]):
                valid[cname] += 1
                if abs(bt) >= abs(t_obs[cname]):
                    extreme[cname] += 1

    pvals = {
        name: (extreme[name] + 1.0) / (valid[name] + 1.0) if valid[name] > 0 else np.nan
        for name, _ in tests
    }
    elapsed = time.time() - start
    log = {"b_reps": int(b_reps), "seed": int(seed), "elapsed_sec": round(elapsed, 2), "valid_reps": valid}
    wdf = pd.DataFrame({"contrast": list(pvals.keys()), "p_value_wild_bootstrap": list(pvals.values())})
    return wdf, log


def run_model_variant(long_df: pd.DataFrame, variant: str, drop_author: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dat = long_df.copy()
    if variant == "drop_2014":
        dat = dat[dat["year_int"] > 2014].copy()
    elif variant == "drop_switchers":
        dat = dat[~dat["author"].isin(SWITCHERS)].copy()
    elif variant == "leave_one_author_out" and drop_author:
        dat = dat[dat["author"] != drop_author].copy()
    if dat.empty:
        return pd.DataFrame(), pd.DataFrame()
    fit = fit_glm(dat)
    cdf = evaluate_contrasts(fit, dat)
    cdf["variant"] = variant
    cdf["dropped_author"] = drop_author or ""
    coef_df = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "p_value": fit.pvalues.values})
    coef_df["variant"] = variant
    coef_df["dropped_author"] = drop_author or ""
    return cdf, coef_df


def build_per_author_contrasts(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for author, ad in long_df.groupby("author", sort=True):
        if ad["period3"].nunique() < 2:
            continue
        try:
            fit = smf.glm(
                'y ~ person * number * C(period3, Treatment("P1_2014_2021"))',
                data=ad,
                family=sm.families.Binomial(),
                freq_weights=ad["n"],
            ).fit()
        except Exception:
            continue
        cdf = evaluate_contrasts(fit, ad)
        for cname in ("P2_vs_P1_1pl_cell_shift", "P2_vs_P1_2sg_cell_shift"):
            rr = cdf[cdf["contrast"].eq(cname)]
            if rr.empty:
                continue
            rows.append(
                {
                    "author": author,
                    "contrast": cname,
                    "estimate_logit": float(rr["estimate_logit"].iloc[0]),
                    "ci95_low": float(rr["ci95_low"].iloc[0]),
                    "ci95_high": float(rr["ci95_high"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_forest(main_df: pd.DataFrame, sens_df: pd.DataFrame, out_path: Path) -> None:
    order = main_df["contrast"].tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 6))
    m = main_df.set_index("contrast").loc[order]
    ax.errorbar(m["estimate_logit"], y, xerr=[m["estimate_logit"] - m["ci95_low"], m["ci95_high"] - m["estimate_logit"]], fmt="o", color="black", label="main")
    for variant, g in sens_df[sens_df["variant"] != "leave_one_author_out"].groupby("variant"):
        gg = g.set_index("contrast").reindex(order)
        ax.scatter(gg["estimate_logit"], y, s=28, alpha=0.8, label=variant)
    lo = sens_df[sens_df["variant"] == "leave_one_author_out"]
    if not lo.empty:
        lo_agg = lo.groupby("contrast")["estimate_logit"].agg(["min", "max"]).reindex(order)
        ax.hlines(y, lo_agg["min"], lo_agg["max"], color="gray", linewidth=2, alpha=0.7, label="LOAO range")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Log-odds estimate")
    ax.set_title("Confirmatory contrasts with sensitivity overlays")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_caterpillar(per_author: pd.DataFrame, out_path: Path) -> None:
    if per_author.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    for i, cname in enumerate(["P2_vs_P1_2sg_cell_shift", "P2_vs_P1_1pl_cell_shift"]):
        ax = axes[i]
        d = per_author[per_author["contrast"] == cname].sort_values("estimate_logit")
        y = np.arange(len(d))
        ax.hlines(y, d["ci95_low"], d["ci95_high"], color="#4c78a8", alpha=0.8)
        ax.plot(d["estimate_logit"], y, "o", color="#4c78a8")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(d["author"])
        ax.set_title(cname)
        ax.set_xlabel("Log-odds")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-period confirmatory contrast models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--roster-ge8", type=Path, default=DEFAULT_ROSTER_GE8)
    parser.add_argument("--min-n12-per-poem", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=1999)
    parser.add_argument("--bootstrap-seed", type=int, default=20260505)
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(args.input.resolve(), args.layer0.resolve() if args.layer0 else None)
    poem_cell = build_poem_cell_table(df)
    roster_main = load_roster_authors(args.roster.resolve())
    roster_ge8 = load_roster_authors(args.roster_ge8.resolve())

    long_main = build_poem_long_4cells(poem_cell, args.min_n12_per_poem, roster_main)
    long_ge8 = build_poem_long_4cells(poem_cell, args.min_n12_per_poem, roster_ge8)

    engine_status: list[dict] = []
    glmm_result, glmm_status = fit_glmm_primary(long_main)
    clogit_result, clogit_status = fit_clogit_primary(long_main)
    engine_status.append(glmm_status)
    engine_status.append(clogit_status)

    coprimary_frames: list[pd.DataFrame] = []
    if glmm_result is not None:
        coprimary_frames.append(
            _coprimary_rows_for_engine(
                glmm_result.params,
                glmm_result.cov,
                long_main,
                engine="glmm",
                raw_engine_label="glmm_lme4_random_author",
                n_authors_used=int(glmm_result.n_authors),
                convergence_status=str(glmm_result.convergence_message or "converged"),
            )
        )
    else:
        coprimary_frames.append(
            _coprimary_unavailable_block(
                long_df=long_main,
                raw_engine_label="glmm_lme4_random_author",
                convergence_status=str(glmm_status.get("status", "unavailable")),
            )
        )
    if clogit_result is not None and len(clogit_result.params) > 0:
        coprimary_frames.append(
            _coprimary_rows_for_engine(
                clogit_result.params,
                clogit_result.cov,
                long_main,
                engine="clogit",
                raw_engine_label="clogit_cox_conditional",
                n_authors_used=int(clogit_result.n_authors_used),
                convergence_status=str(clogit_result.convergence_status),
            )
        )
    else:
        coprimary_frames.append(
            _coprimary_unavailable_block(
                long_df=long_main,
                raw_engine_label="clogit_cox_conditional",
                convergence_status=str(clogit_status.get("status", "unavailable")),
            )
        )
    coprimary = pd.concat(coprimary_frames, ignore_index=True)
    if coprimary["p_value"].notna().any():
        coprimary["q_value_bh_pooled"] = bh_adjust(coprimary["p_value"])
    else:
        coprimary["q_value_bh_pooled"] = np.nan
    coprimary = coprimary[
        [
            "engine",
            "contrast",
            "estimate_logit",
            "ci95_low",
            "ci95_high",
            "odds_ratio",
            "se",
            "z_value",
            "p_value",
            "q_value_bh_within_engine",
            "q_value_bh_pooled",
            "n_authors_used",
            "n_poems",
            "n_rows_long",
            "convergence_status",
        ]
    ]
    coprimary.to_csv(out / "confirmatory_contrasts_coprimary.csv", index=False)
    with (out / "coprimary_engine_status.json").open("w", encoding="utf-8") as f:
        json.dump(engine_status, f, indent=2, ensure_ascii=False, default=str)

    if glmm_result is not None:
        pd.DataFrame(
            {
                "term": glmm_result.params.index,
                "coef": glmm_result.params.values,
                "se": np.sqrt(np.diag(glmm_result.cov.to_numpy(dtype=float))),
            }
        ).to_csv(out / "unified_model_coefficients_glmm.csv", index=False)
    if clogit_result is not None and len(clogit_result.params) > 0:
        pd.DataFrame(
            {
                "term": clogit_result.params.index,
                "coef": clogit_result.params.values,
                "se": np.sqrt(np.diag(clogit_result.cov.to_numpy(dtype=float))),
            }
        ).to_csv(out / "unified_model_coefficients_clogit.csv", index=False)

    fit_main = fit_glm(long_main)
    fit_main_cluster = fit_glm_clustered_author(fit_main, long_main)
    confirm_naive = evaluate_contrasts(fit_main, long_main, p_value_col="p_value_naive_glm")
    if fit_main_cluster is not None:
        confirm_cluster = evaluate_contrasts(fit_main_cluster, long_main, p_value_col="p_value_clustered_author")
    else:
        confirm_cluster = confirm_naive.rename(
            columns={
                "se": "se",
                "z_value": "z_value",
                "p_value_naive_glm": "p_value_clustered_author",
                "q_value_bh_p_value_naive_glm_family": "q_value_bh_p_value_clustered_author_family",
            }
        )
    wb_df, wb_log = wild_cluster_bootstrap(long_main, fit_main, args.bootstrap_reps, args.bootstrap_seed)
    confirm_main = confirm_cluster.merge(
        confirm_naive[
            [
                "contrast",
                "se",
                "z_value",
                "p_value_naive_glm",
                "q_value_bh_p_value_naive_glm_family",
            ]
        ].rename(columns={"se": "se_naive_glm", "z_value": "z_value_naive_glm"}),
        on="contrast",
        how="left",
    )
    confirm_main = confirm_main.rename(columns={"se": "se_clustered_author", "z_value": "z_value_clustered_author"})
    confirm_main = confirm_main.merge(wb_df, on="contrast", how="left")
    confirm_main["q_value_bh_wild_bootstrap_family"] = bh_adjust(confirm_main["p_value_wild_bootstrap"])
    confirm_main["model_label"] = "legacy_fe_glm_plus_author_dummies"
    confirm_main["sensitivity_caveat"] = (
        "Unconditional MLE with N≈33 author dummies and median 20–25 informative "
        "poems per author×cell×period; subject to incidental-parameter bias of "
        "order O(1/T). Reported as sensitivity only; primary inference is the "
        "co-primary GLMM + conditional logit (confirmatory_contrasts_coprimary.csv)."
    )
    confirm_main = confirm_main[
        [
            "contrast",
            "model_label",
            "estimate_logit",
            "p_value_wild_bootstrap",
            "q_value_bh_wild_bootstrap_family",
            "p_value_clustered_author",
            "q_value_bh_p_value_clustered_author_family",
            "p_value_naive_glm",
            "q_value_bh_p_value_naive_glm_family",
            "se_clustered_author",
            "z_value_clustered_author",
            "se_naive_glm",
            "z_value_naive_glm",
            "ci95_low",
            "ci95_high",
            "odds_ratio",
            "n_poems",
            "n_rows_long",
            "sensitivity_caveat",
        ]
    ]
    confirm_main.to_csv(out / "confirmatory_contrasts_main.csv", index=False)
    pd.DataFrame({"term": fit_main.params.index, "coef": fit_main.params.values, "p_value": fit_main.pvalues.values}).to_csv(
        out / "unified_model_coefficients.csv", index=False
    )

    sens_frames: list[pd.DataFrame] = []
    for v in ("drop_2014", "drop_switchers"):
        cdf, _ = run_model_variant(long_main, variant=v)
        if not cdf.empty:
            sens_frames.append(cdf)
    cdf_ge8, _ = run_model_variant(long_ge8, variant="roster_ge8")
    if not cdf_ge8.empty:
        sens_frames.append(cdf_ge8)
    for a in sorted(set(long_main["author"].astype(str))):
        cdf_loo, _ = run_model_variant(long_main, variant="leave_one_author_out", drop_author=a)
        if not cdf_loo.empty:
            sens_frames.append(cdf_loo)
    sensitivity = pd.concat(sens_frames, ignore_index=True) if sens_frames else pd.DataFrame()
    sensitivity.to_csv(out / "confirmatory_contrasts_sensitivity.csv", index=False)

    per_author = build_per_author_contrasts(long_main)
    per_author.to_csv(out / "per_author_contrast_estimates.csv", index=False)

    plot_forest(confirm_main, sensitivity, fig_dir / "fig1_confirmatory_forest_with_sensitivity.pdf")
    plot_caterpillar(per_author, fig_dir / "fig2_per_author_caterpillar.pdf")

    with (out / "wild_bootstrap_log.txt").open("w", encoding="utf-8") as f:
        f.write(f"B={wb_log['b_reps']}\n")
        f.write(f"seed={wb_log['seed']}\n")
        f.write(f"elapsed_sec={wb_log['elapsed_sec']}\n")
        f.write(f"valid_reps={wb_log['valid_reps']}\n")

    with (out / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Attention Allocation (closed 4-cell) — confirmatory model outputs\n\n")
        f.write("Estimand: within the closed first/second-person quartet ")
        f.write("`{1sg, 1pl, 2sg, 2pl_vy_true_plural}`, how is attention reallocated across periods?\n\n")
        f.write("## Primary inference (co-primary)\n\n")
        f.write("`confirmatory_contrasts_coprimary.csv` reports both engines side by side:\n\n")
        f.write("- `glmm_lme4_random_author` — binomial GLMM with random author intercept ")
        f.write("`cbind(k, n - k) ~ person * number * period3 + (1 | author)` via R / lme4. ")
        f.write("Fixed-effect inference uses Wald CI / p-values from `vcov(fit)`.\n")
        f.write("- `clogit_cox_conditional` — Cox conditional logistic regression on trial-expanded "
                "data; the author intercept is eliminated via the conditional likelihood, "
                "avoiding incidental-parameter bias at N≈33.\n\n")
        f.write("BH q-values are reported both within each engine ")
        f.write("(`q_value_bh_within_engine`) and pooled across the 6-row family ")
        f.write("(`q_value_bh_pooled`). Convergence and engine availability are recorded in ")
        f.write("`coprimary_engine_status.json`.\n\n")
        f.write("## Sensitivity (demoted)\n\n")
        f.write("`confirmatory_contrasts_main.csv` retains the legacy unconditional GLM with ")
        f.write("`+ C(author)` author dummies plus wild cluster bootstrap. Because this estimator ")
        f.write("incurs incidental-parameter bias of order O(1/T) at our sample size, it is ")
        f.write("reported as sensitivity only; see `sensitivity_caveat` column. Do **not** ")
        f.write("interpret `p_value_wild_bootstrap` as the headline inferential result.\n")
        f.write("`p_value_naive_glm` is reported for reference only because it ignores author clustering.\n")

    print(f"Wrote two-period outputs to: {out}")


if __name__ == "__main__":
    main()

