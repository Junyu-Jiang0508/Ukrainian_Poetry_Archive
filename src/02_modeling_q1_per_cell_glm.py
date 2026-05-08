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


def build_exposure_diagnostics(poem_df: pd.DataFrame) -> pd.DataFrame:
    """Compact distribution diagnostics for each available exposure definition."""
    specs = (
        ("n_stanzas", "exposure_n_stanzas"),
        ("n_tokens", "exposure_n_tokens"),
        ("n_finite_verbs", "exposure_n_finite_verbs"),
    )
    rows: list[dict[str, object]] = []
    n_poems = int(len(poem_df))
    for exposure_type, col in specs:
        if col not in poem_df.columns:
            continue
        s = pd.to_numeric(poem_df[col], errors="coerce").fillna(0.0)
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
    elif "include_in_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_offset_models"].astype(bool)].copy()
    dat = dat[dat[period_col].isin((period_reference, period_treatment))]
    dat = dat[dat["n_total"].ge(int(min_total))]
    if roster_authors is not None:
        dat = dat[dat["author"].astype(str).isin(roster_authors)]
    if dat.empty:
        return pd.DataFrame()

    if exposure_type == "n_stanzas":
        ex_col = "exposure_n_stanzas"
    elif exposure_type == "n_tokens":
        ex_col = "exposure_n_tokens"
    elif exposure_type == "n_finite_verbs":
        ex_col = "exposure_n_finite_verbs"
    else:
        raise ValueError(f"Unknown exposure_type: {exposure_type!r}")

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
        choices=("n_stanzas", "n_tokens", "n_finite_verbs"),
        help=(
            "Offset column: exposure_n_stanzas (primary), exposure_n_tokens (robustness), "
            "or exposure_n_finite_verbs (syntactic-slot robustness)."
        ),
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
    poem_full = build_poem_cell_table_with_exposure(filtered)
    build_exposure_diagnostics(poem_full).to_csv(out_dir / "q1_exposure_diagnostics.csv", index=False)

    frames: list[pd.DataFrame] = []
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

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    poem_full.to_csv(out_dir / "q1_poem_unit_cell_counts_12.csv", index=False)

    if args.exposure_type == "n_stanzas":
        glm_out_name = "q1_poem_per_cell_glm_by_language.csv"
    else:
        # Single-underscore + descriptive suffix to match the robustness-file
        # naming convention (e.g. ..._robust_period_invasion_20220224.csv).
        glm_out_name = f"q1_poem_per_cell_glm_by_language_offset_{args.exposure_type}.csv"
    combined.to_csv(out_dir / glm_out_name, index=False)

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
                "counts as a syntactic opportunity denominator. Exposure diagnostics for all available "
                "denominators are written to `q1_exposure_diagnostics.csv`.\n"
            )
            f.write("- Strata: pooled (non-primary BH), Ukrainian, Russian.\n")
            f.write(
                "- Model: Poisson with `offset(log_exposure)`; clustered SE by author; BH-FDR within "
                "stratum among primary strata rows only.\n"
            )

    print(f"Wrote Q1 outputs to: {out_dir} (exposure_type={args.exposure_type})")


if __name__ == "__main__":
    main()
