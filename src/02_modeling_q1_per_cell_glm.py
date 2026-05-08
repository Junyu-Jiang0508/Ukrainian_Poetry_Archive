"""Q1: first/second-person per-cell period shift (poem level, language strata).

Per-cell Poisson GLM among the four frequentist cells (PRIMARY_GLM_CELLS_FREQUENTIST):
``1sg, 1pl, 2sg, 2pl_vy_true_plural`` with offset ``log(exposure_n_stanzas)``.
``2pl_vy_polite_singular`` is excluded from frequentist inference (sparsity →
separation; see Q1 README) but retained on the Bayesian path in Q2 and as a
column in ``q1_poem_unit_cell_counts_12.csv``. Legacy aggregated ``2pl`` is also
kept in the counts CSV but not in the inferential loop.

Strata: pooled Ukrainian∪Russian (descriptive; excluded from primary BH family),
Ukrainian-only, Russian-only. Crimean Tatar / Qirimli poems are excluded upstream
(see Methods).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.language_strata import (
    LANGUAGE_STRATA,
    filter_annotation_for_inference_language,
    filter_poems_by_language_stratum,
    primary_stratum_for_bh,
)
from utils.finite_verb_exposure import resolve_finite_verb_counts_for_modeling
from utils.poem_cell_counts import build_poem_cell_table_with_exposure
from utils.pronoun_encoding import PRIMARY_GLM_CELLS, pronoun_class_sixway_column
from utils.stats_common import bh_adjust, normalize_bool_flag, period_three_way
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_q1_per_cell_glm"

PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"
PERIODS = (PERIOD_P1, PERIOD_P2)


def _exposure_column_name(exposure_type: str) -> str:
    if exposure_type == "n_stanzas":
        return "exposure_n_stanzas"
    if exposure_type == "n_tokens":
        return "exposure_n_tokens"
    if exposure_type == "n_finite_verbs":
        return "exposure_n_finite_verbs"
    if exposure_type == "n_finite_verbs_excl_imperative":
        return "exposure_n_finite_verbs_excl_imperative"
    raise ValueError(f"Unknown exposure_type: {exposure_type!r}")


def _wild_cluster_bootstrap_single_coef(
    cdf: pd.DataFrame,
    formula: str,
    term: str,
    *,
    b_reps: int,
    seed: int,
) -> float:
    if int(b_reps) <= 0:
        return np.nan
    fit = smf.glm(formula, data=cdf, family=sm.families.Poisson(), offset=cdf["log_exposure"]).fit()
    if term not in fit.params.index:
        return np.nan
    se_obs = float(fit.bse[term]) if term in fit.bse.index else np.nan
    if not np.isfinite(se_obs) or se_obs <= 0:
        return np.nan
    t_obs = float(fit.params[term] / se_obs)
    resid = cdf["k"].to_numpy(dtype=float) - fit.fittedvalues.to_numpy(dtype=float)
    mu = fit.fittedvalues.to_numpy(dtype=float)
    groups = cdf["author"].astype(str).to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(int(seed))
    extreme = 0
    valid = 0
    for _ in range(int(b_reps)):
        w_map = {g: rng.choice([-1.0, 1.0]) for g in uniq}
        signs = np.array([w_map[g] for g in groups], dtype=float)
        y_star = np.clip(mu + signs * resid, 1e-8, None)
        bdf = cdf.copy()
        bdf["k"] = y_star
        try:
            bfit = smf.glm(formula, data=bdf, family=sm.families.Poisson(), offset=bdf["log_exposure"]).fit()
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
    return float((extreme + 1.0) / (valid + 1.0))


def build_exposure_diagnostics(poem_df: pd.DataFrame) -> pd.DataFrame:
    """Compact distribution diagnostics for each available exposure definition."""
    specs = (
        ("n_stanzas", "exposure_n_stanzas"),
        ("n_tokens", "exposure_n_tokens"),
        ("n_finite_verbs", "exposure_n_finite_verbs"),
        ("n_finite_verbs_excl_imperative", "exposure_n_finite_verbs_excl_imperative"),
    )
    rows: list[dict[str, object]] = []
    n_poems = int(len(poem_df))
    for exposure_type, col in specs:
        if col not in poem_df.columns:
            continue
        s = pd.to_numeric(poem_df[col], errors="coerce")
        if s.isna().all():
            continue
        s = s.fillna(0.0)
        rows.append(
            {
                "exposure_type": exposure_type,
                "column": col,
                "n_poems": n_poems,
                "min": float(s.min()) if n_poems else np.nan,
                "p25": float(s.quantile(0.25)) if n_poems else np.nan,
                "median": float(s.quantile(0.50)) if n_poems else np.nan,
                "p75": float(s.quantile(0.75)) if n_poems else np.nan,
                "max": float(s.max()) if n_poems else np.nan,
                "share_eq_0": float((s == 0).mean()) if n_poems else np.nan,
                "share_eq_1": float((s == 1).mean()) if n_poems else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_roster_authors(roster_path: Path | None) -> set[str] | None:
    if roster_path is None or not roster_path.is_file():
        return None
    r = pd.read_csv(roster_path, low_memory=False)
    if "included" not in r.columns or "author" not in r.columns:
        return None
    return set(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def load_and_filter(
    path: Path,
    layer0_path: Path | None,
    *,
    language_audit_dir: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["period3"] = df["year_int"].map(period_three_way)
    df["person"] = df["person"].fillna("").astype(str).str.strip()
    df["number"] = df["number"].fillna("").astype(str).str.strip()
    df["person_number"] = pronoun_class_sixway_column(df)
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

    out = df.loc[~(df["is_repeat"] | df["is_translation"])].copy()
    out, _ = filter_annotation_for_inference_language(
        out, audit_dir=language_audit_dir, audit_filename="dropped_poems_language_constraints.csv"
    )
    return out


def fit_q1_poisson_per_cell(
    poem_df: pd.DataFrame,
    roster_authors: set[str] | None,
    min_total: int,
    *,
    language_stratum: str,
    period_col: str = "period3",
    period_reference: str = PERIOD_P1,
    period_treatment: str = PERIOD_P2,
    exposure_type: str = "n_stanzas",
) -> pd.DataFrame:
    dat = poem_df.copy()
    if exposure_type == "n_finite_verbs" and "include_in_fv_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_fv_offset_models"].astype(bool)].copy()
    elif exposure_type == "n_finite_verbs_excl_imperative" and "include_in_fv_excl_imp_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_fv_excl_imp_offset_models"].astype(bool)].copy()
    elif exposure_type in ("n_stanzas", "n_tokens") and "include_in_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_offset_models"].astype(bool)].copy()
    dat = dat[dat[period_col].isin((period_reference, period_treatment))]
    dat = dat[dat["n_total"].ge(int(min_total))]
    if roster_authors is not None:
        dat = dat[dat["author"].astype(str).isin(roster_authors)]
    if dat.empty:
        return pd.DataFrame()

    ex_col = _exposure_column_name(exposure_type)

    rows: list[dict[str, object]] = []
    is_primary = primary_stratum_for_bh(language_stratum)

    for cell in PRIMARY_GLM_CELLS:
        cdf = dat.copy()
        cdf["k"] = cdf[cell].astype(int)
        cdf["_ex"] = cdf[ex_col].astype(float)
        cdf = cdf[cdf["_ex"].gt(0) & np.isfinite(cdf["_ex"])].copy()
        if cdf.empty:
            continue
        cdf["log_exposure"] = np.log(cdf["_ex"])
        if cdf[period_col].nunique() < 2 or int(cdf["k"].sum()) == 0:
            continue
        groups = cdf["author"].astype(str)
        formula = f"k ~ C({period_col}, Treatment('{period_reference}'))"
        fit = smf.glm(formula, data=cdf, family=sm.families.Poisson(), offset=cdf["log_exposure"]).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        term = f"C({period_col}, Treatment('{period_reference}'))[T.{period_treatment}]"
        if term not in fit.params.index:
            continue
        ci = fit.conf_int().loc[term]
        coef = float(fit.params[term])
        rows.append(
            {
                "language_stratum": language_stratum,
                "cell": cell,
                "n_poems": int(len(cdf)),
                "n_authors": int(groups.nunique()),
                "coef_post_vs_pre_log_mu": coef,
                "rate_ratio_post_vs_pre": float(np.exp(coef)),
                "ci95_low_log_mu": float(ci.iloc[0]),
                "ci95_high_log_mu": float(ci.iloc[1]),
                "rate_ratio_ci95_low": float(np.exp(ci.iloc[0])),
                "rate_ratio_ci95_high": float(np.exp(ci.iloc[1])),
                "se_clustered_author": float(fit.bse[term]),
                "z_value_clustered_author": float(fit.tvalues[term]),
                "p_value_clustered_author": float(fit.pvalues[term]),
                "exposure_type": exposure_type,
                "is_primary_stratum": bool(is_primary),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["q_value_bh_within_stratum"] = np.nan
    sub = out.loc[out["is_primary_stratum"]]
    if not sub.empty:
        q = sub.groupby("language_stratum", group_keys=False)["p_value_clustered_author"].apply(bh_adjust)
        out.loc[q.index, "q_value_bh_within_stratum"] = q
    return out


def fit_q1_coprimary_per_cell(
    poem_df: pd.DataFrame,
    roster_authors: set[str] | None,
    min_total: int,
    *,
    language_stratum: str,
    period_col: str = "period3",
    period_reference: str = PERIOD_P1,
    period_treatment: str = PERIOD_P2,
    exposure_type: str = "n_stanzas",
    bootstrap_reps: int = 1999,
    bootstrap_seed: int = 20260508,
) -> pd.DataFrame:
    dat = poem_df.copy()
    if exposure_type == "n_finite_verbs" and "include_in_fv_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_fv_offset_models"].astype(bool)].copy()
    elif exposure_type == "n_finite_verbs_excl_imperative" and "include_in_fv_excl_imp_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_fv_excl_imp_offset_models"].astype(bool)].copy()
    elif exposure_type in ("n_stanzas", "n_tokens") and "include_in_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_offset_models"].astype(bool)].copy()
    dat = dat[dat[period_col].isin((period_reference, period_treatment))]
    dat = dat[dat["n_total"].ge(int(min_total))]
    if roster_authors is not None:
        dat = dat[dat["author"].astype(str).isin(roster_authors)]
    if dat.empty:
        return pd.DataFrame()
    ex_col = _exposure_column_name(exposure_type)
    rows: list[dict[str, object]] = []
    is_primary = primary_stratum_for_bh(language_stratum)
    for cell in PRIMARY_GLM_CELLS:
        cdf = dat.copy()
        cdf["k"] = cdf[cell].astype(int)
        cdf["_ex"] = cdf[ex_col].astype(float)
        cdf = cdf[cdf["_ex"].gt(0) & np.isfinite(cdf["_ex"])].copy()
        if cdf.empty:
            continue
        cdf["log_exposure"] = np.log(cdf["_ex"])
        if cdf[period_col].nunique() < 2 or int(cdf["k"].sum()) == 0:
            continue
        groups = cdf["author"].astype(str)
        formula = f"k ~ C({period_col}, Treatment('{period_reference}'))"
        term = f"C({period_col}, Treatment('{period_reference}'))[T.{period_treatment}]"
        # Poisson + clustered SE
        try:
            fit_pois = smf.glm(formula, data=cdf, family=sm.families.Poisson(), offset=cdf["log_exposure"]).fit(
                cov_type="cluster", cov_kwds={"groups": groups}
            )
            if term in fit_pois.params.index:
                ci = fit_pois.conf_int().loc[term]
                coef = float(fit_pois.params[term])
                p_wild = _wild_cluster_bootstrap_single_coef(
                    cdf,
                    formula,
                    term,
                    b_reps=bootstrap_reps,
                    seed=bootstrap_seed
                    + int(
                        hashlib.md5(f"{language_stratum}|{cell}|{exposure_type}".encode("utf-8")).hexdigest()[:8], 16
                    )
                    % 100_000,
                )
                rows.append(
                    {
                        "language_stratum": language_stratum,
                        "cell": cell,
                        "model_variant": "poisson_cluster",
                        "n_poems": int(len(cdf)),
                        "n_authors": int(groups.nunique()),
                        "coef_post_vs_pre_log_mu": coef,
                        "rate_ratio_post_vs_pre": float(np.exp(coef)),
                        "ci95_low_log_mu": float(ci.iloc[0]),
                        "ci95_high_log_mu": float(ci.iloc[1]),
                        "rate_ratio_ci95_low": float(np.exp(ci.iloc[0])),
                        "rate_ratio_ci95_high": float(np.exp(ci.iloc[1])),
                        "se_clustered_author": float(fit_pois.bse[term]),
                        "z_value_clustered_author": float(fit_pois.tvalues[term]),
                        "p_value_clustered_author": float(fit_pois.pvalues[term]),
                        "p_value_wild_cluster_bootstrap": p_wild,
                        "exposure_type": exposure_type,
                        "is_primary_stratum": bool(is_primary),
                    }
                )
        except Exception:
            pass
        # NB + clustered SE
        try:
            fit_nb = smf.negativebinomial(formula, data=cdf, offset=cdf["log_exposure"]).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": groups}
            )
            if term in fit_nb.params.index:
                ci_nb = fit_nb.conf_int().loc[term]
                coef_nb = float(fit_nb.params[term])
                rows.append(
                    {
                        "language_stratum": language_stratum,
                        "cell": cell,
                        "model_variant": "negative_binomial_cluster",
                        "n_poems": int(len(cdf)),
                        "n_authors": int(groups.nunique()),
                        "coef_post_vs_pre_log_mu": coef_nb,
                        "rate_ratio_post_vs_pre": float(np.exp(coef_nb)),
                        "ci95_low_log_mu": float(ci_nb.iloc[0]),
                        "ci95_high_log_mu": float(ci_nb.iloc[1]),
                        "rate_ratio_ci95_low": float(np.exp(ci_nb.iloc[0])),
                        "rate_ratio_ci95_high": float(np.exp(ci_nb.iloc[1])),
                        "se_clustered_author": float(fit_nb.bse[term]),
                        "z_value_clustered_author": float(fit_nb.tvalues[term]),
                        "p_value_clustered_author": float(fit_nb.pvalues[term]),
                        "p_value_wild_cluster_bootstrap": np.nan,
                        "exposure_type": exposure_type,
                        "is_primary_stratum": bool(is_primary),
                    }
                )
        except Exception:
            pass
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh_within_stratum"] = np.nan
    out["q_value_bh_wild_within_stratum"] = np.nan
    sub = out.loc[out["is_primary_stratum"]]
    if not sub.empty:
        q_cluster = (
            sub.groupby(["language_stratum", "model_variant"], group_keys=False)["p_value_clustered_author"].apply(bh_adjust)
        )
        out.loc[q_cluster.index, "q_value_bh_within_stratum"] = q_cluster
        q_wild = (
            sub.loc[sub["model_variant"].eq("poisson_cluster")]
            .groupby(["language_stratum", "model_variant"], group_keys=False)["p_value_wild_cluster_bootstrap"]
            .apply(bh_adjust)
        )
        out.loc[q_wild.index, "q_value_bh_wild_within_stratum"] = q_wild
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(
        "Crimean Tatar / Qirimli language codes excluded from inference: see Methods §X "
        "(not an accidental omission of a stratum)."
    )

    parser = argparse.ArgumentParser(
        description="Q1: poem-level 1st/2nd-person per-cell GLMs with Ukrainian/Russian/pooled strata."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-total-per-poem",
        type=int,
        default=0,
        help="Keep poems with combined primary-cell count ≥ this value (default 0).",
    )
    parser.add_argument(
        "--strata",
        type=str,
        default=",".join(LANGUAGE_STRATA),
        help=f"Comma-separated subset of: {','.join(LANGUAGE_STRATA)}",
    )
    parser.add_argument(
        "--exposure-type",
        type=str,
        default="n_stanzas",
        choices=("n_stanzas", "n_tokens", "n_finite_verbs", "n_finite_verbs_excl_imperative"),
        help=(
            "Offset column: exposure_n_stanzas (primary), exposure_n_tokens (robustness), "
            "exposure_n_finite_verbs or exposure_n_finite_verbs_excl_imperative (FV robustness)."
        ),
    )
    parser.add_argument(
        "--finite-verb-counts",
        type=Path,
        default=None,
        help="Path to stanza_finite_verb_counts.csv (default: data/To_run/00_filtering/...).",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1999,
        help="Wild-cluster bootstrap repetitions for Q1 co-primary Poisson path.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260508,
        help="Random seed for Q1 co-primary Poisson wild-cluster bootstrap.",
    )
    args = parser.parse_args()

    want_strata = tuple(s.strip() for s in args.strata.split(",") if s.strip())
    for s in want_strata:
        if s not in LANGUAGE_STRATA:
            raise SystemExit(f"Unknown stratum {s!r}. Choose from {LANGUAGE_STRATA}")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "language_stratum_audit"

    filtered = load_and_filter(
        args.input.resolve(),
        args.layer0.resolve() if args.layer0 else None,
        language_audit_dir=audit_dir,
    )
    roster_authors = load_roster_authors(args.roster.resolve() if args.roster else None)
    fv_df = resolve_finite_verb_counts_for_modeling(
        ROOT,
        exposure_type=args.exposure_type,
        finite_verb_csv=args.finite_verb_counts,
    )
    poem_full = build_poem_cell_table_with_exposure(
        filtered,
        finite_verb_df=fv_df,
        discontinuity_manifest_path=out_dir / "stanza_index_discontinuities.csv",
    )
    build_exposure_diagnostics(poem_full).to_csv(out_dir / "q1_exposure_diagnostics.csv", index=False)

    frames: list[pd.DataFrame] = []
    coprimary_frames: list[pd.DataFrame] = []
    for stratum in want_strata:
        poem_sub = filter_poems_by_language_stratum(poem_full, stratum)
        qdf = fit_q1_poisson_per_cell(
            poem_sub,
            roster_authors,
            args.min_total_per_poem,
            language_stratum=stratum,
            exposure_type=args.exposure_type,
        )
        if not qdf.empty:
            frames.append(qdf)
        cdf = fit_q1_coprimary_per_cell(
            poem_sub,
            roster_authors,
            args.min_total_per_poem,
            language_stratum=stratum,
            exposure_type=args.exposure_type,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
        )
        if not cdf.empty:
            coprimary_frames.append(cdf)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_coprimary = pd.concat(coprimary_frames, ignore_index=True) if coprimary_frames else pd.DataFrame()

    poem_full.to_csv(out_dir / "q1_poem_unit_cell_counts_12.csv", index=False)

    if args.exposure_type == "n_stanzas":
        glm_out_name = "q1_poem_per_cell_glm_by_language.csv"
        coprimary_out_name = "q1_poem_per_cell_glm_by_language_coprimary.csv"
    else:
        # Single-underscore + descriptive suffix to match the robustness-file
        # naming convention (e.g. ..._robust_period_invasion_20220224.csv).
        glm_out_name = f"q1_poem_per_cell_glm_by_language_offset_{args.exposure_type}.csv"
        coprimary_out_name = f"q1_poem_per_cell_glm_by_language_offset_{args.exposure_type}_coprimary.csv"
    combined.to_csv(out_dir / glm_out_name, index=False)
    combined_coprimary.to_csv(out_dir / coprimary_out_name, index=False)

    readme_path = out_dir / "README.md"
    if not readme_path.is_file() or args.exposure_type == "n_stanzas":
        with readme_path.open("w", encoding="utf-8") as f:
            f.write("# Q1 Per-cell Poisson GLM (1st/2nd person, poem level)\n\n")
            f.write(
                "- Primary cells (inference loop, 4-cell): `{1sg, 1pl, 2sg, 2pl_vy_true_plural}`. "
                "`2pl_vy_polite_singular` is dropped from inference (23 events / 18 poems / 13 authors; "
                "only 3 authors have events in both periods → separation in FE designs and degenerate "
                "cluster-robust covariance in main GLM). The polite-singular column is still emitted "
                "in `q1_poem_unit_cell_counts_12.csv` for future stanza-level work. Legacy aggregated "
                "`2pl` is also kept in the counts CSV.\n"
            )
            f.write(
                "- Offset: primary analysis uses `exposure_n_stanzas` "
                "(`q1_poem_per_cell_glm_by_language.csv`); token sensitivity is run side-by-side via "
                "`--exposure-type=n_tokens` and written to "
                "`q1_poem_per_cell_glm_by_language_offset_n_tokens.csv`. The two should be compared "
                "because 74% of poems have `exposure_n_stanzas == 1`, so the stanza-offset is "
                "degenerate for half the corpus.\n"
            )
            f.write(
                "- Additional robustness option: `--exposure-type=n_finite_verbs` writes "
                "`q1_poem_per_cell_glm_by_language_offset_n_finite_verbs.csv`, using finite-verb "
                "counts as a syntactic opportunity denominator. "
                "`--exposure-type=n_finite_verbs_excl_imperative` excludes `Mood=Imp` from the FV "
                "denominator (sensitivity). Precompute counts once via `00e_compute_finite_verb_exposure.py` "
                "→ `data/To_run/00_filtering/stanza_finite_verb_counts.csv`. Exposure diagnostics for all "
                "available denominators are written to `q1_exposure_diagnostics.csv`.\n"
            )
            f.write("- Strata: pooled (non-primary BH), Ukrainian, Russian.\n")
            f.write(
                "- Model: Poisson with `offset(log_exposure)`; clustered SE by author; BH-FDR within "
                "stratum among primary strata rows only.\n"
            )
            f.write(
                "- Co-primary inference file: `*_coprimary.csv` contains parallel estimates for "
                "`poisson_cluster` (with wild-cluster bootstrap p-values) and "
                "`negative_binomial_cluster`.\n"
            )
            f.write("\n## Limitations of the finite-verb offset\n\n")
            f.write(
                "- **Imperatives**: default FV counts treat `Mood=Imp` as finite (same syntactic slot as "
                "other finite verbs but often pro-drop without explicit subject pronouns). Compare "
                "`n_finite_verbs` vs `n_finite_verbs_excl_imperative` outputs and "
                "`02_modeling_robustness_offset_comparison.py`.\n"
            )
            f.write(
                "- **Zero copula / ellipsis**: lines like nominal predicates without a finite verb "
                "contribute pronouns but not to the FV denominator — possible period-asymmetric bias.\n"
            )
            f.write(
                "- **Stanza pipeline**: FV stage uses Stanza `tokenize,pos,lemma`; "
                "`01_annotation_pronoun_detection.py` uses `tokenize,pos,lemma,depparse`. "
                "See `finite_verb_validation_pipeline_agreement.csv` from "
                "`02_modeling_finite_verb_validation_sample.py`.\n"
            )
            f.write(
                "- **Stanza index gaps**: poems where `nunique(stanza_index) != max(stanza_index)` are "
                "listed in `stanza_index_discontinuities.csv` (upstream segmentation; exposure sums only "
                "observed stanza indices).\n"
            )

    print(f"Wrote Q1 outputs to: {out_dir} (exposure_type={args.exposure_type})")


if __name__ == "__main__":
    main()
