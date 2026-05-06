"""Q1: first/second-person per-cell period shift (poem level, language strata).

One-vs-rest binomial GLM per cell among {1sg,1pl,2sg,2pl} only: denominator is the
count of 1st/2nd-person pronouns in the poem (third person excluded). Strata:
pooled Ukrainian∪Russian, Ukrainian-only, Russian-only (exact language tags).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum
from utils.pronoun_encoding import pronoun_class_sixway_column
from utils.stats_common import bh_adjust, normalize_bool_flag, period_three_way
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_q1_per_cell_glm"

PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"
PERIODS = [PERIOD_P1, PERIOD_P2]
# First/second person only; denominator per poem = sum of these four cells.
CELL12 = ["1sg", "1pl", "2sg", "2pl"]
QIRIMLI_CODES = {"Qirimli", "Russian, Qirimli", "Ukrainian, Qirimli"}


def load_roster_authors(roster_path: Path | None) -> set[str] | None:
    if roster_path is None or not roster_path.is_file():
        return None
    r = pd.read_csv(roster_path, low_memory=False)
    if "included" not in r.columns or "author" not in r.columns:
        return None
    return set(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def load_and_filter(path: Path, layer0_path: Path | None) -> pd.DataFrame:
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
    out = out[~out["language_clean"].isin(QIRIMLI_CODES)].copy()
    out = out[out["person_number"].isin(CELL12)].copy()
    return out


def build_poem_cell_table_12(df: pd.DataFrame) -> pd.DataFrame:
    """Poem × cell counts for 1st/2nd person only; n_total = sum over those four cells."""
    counts = (
        df.groupby(["poem_id", "person_number"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for cell in CELL12:
        if cell not in counts.columns:
            counts[cell] = 0
    counts = counts[["poem_id"] + CELL12]
    counts["n_total"] = counts[CELL12].sum(axis=1).astype(int)

    meta = (
        df.groupby("poem_id", as_index=False)
        .agg(
            author=("author", "first"),
            language_clean=("language_clean", "first"),
            year_int=("year_int", "first"),
        )
        .copy()
    )
    meta["period3"] = meta["year_int"].map(period_three_way)
    return counts.merge(meta, on="poem_id", how="left")


def fit_one_vs_rest_models(
    poem_df: pd.DataFrame,
    roster_authors: set[str] | None,
    min_total: int,
    *,
    language_stratum: str,
) -> pd.DataFrame:
    dat = poem_df.copy()
    dat = dat[dat["period3"].isin(PERIODS)]
    dat = dat[dat["n_total"] >= int(min_total)]
    if roster_authors is not None:
        dat = dat[dat["author"].astype(str).isin(roster_authors)]
    if dat.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for cell in CELL12:
        cdf = dat.copy()
        cdf["k"] = cdf[cell].astype(int)
        cdf["n_other"] = cdf["n_total"].astype(int) - cdf["k"]
        if cdf["k"].nunique() < 2 or cdf["period3"].nunique() < 2:
            continue
        groups = cdf["author"].astype(str)
        fit = smf.glm(
            f"k + n_other ~ C(period3, Treatment('{PERIOD_P1}'))",
            data=cdf,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": groups})
        term = f"C(period3, Treatment('{PERIOD_P1}'))[T.{PERIOD_P2}]"
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
                "coef_post_vs_pre_log_odds": coef,
                "odds_ratio_post_vs_pre": float(np.exp(coef)),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "or_ci95_low": float(np.exp(ci.iloc[0])),
                "or_ci95_high": float(np.exp(ci.iloc[1])),
                "se_clustered_author": float(fit.bse[term]),
                "z_value_clustered_author": float(fit.tvalues[term]),
                "p_value_clustered_author": float(fit.pvalues[term]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh_within_stratum"] = out.groupby("language_stratum", group_keys=False)["p_value_clustered_author"].apply(
        bh_adjust
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q1: poem-level 1st/2nd-person per-cell GLMs with Ukrainian/Russian/pooled strata."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-total-per-poem", type=int, default=1, help="Min 1st+2nd-person pronouns per poem.")
    parser.add_argument(
        "--strata",
        type=str,
        default=",".join(LANGUAGE_STRATA),
        help=f"Comma-separated subset of: {','.join(LANGUAGE_STRATA)}",
    )
    args = parser.parse_args()

    want_strata = tuple(s.strip() for s in args.strata.split(",") if s.strip())
    for s in want_strata:
        if s not in LANGUAGE_STRATA:
            raise SystemExit(f"Unknown stratum {s!r}. Choose from {LANGUAGE_STRATA}")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered = load_and_filter(args.input.resolve(), args.layer0.resolve() if args.layer0 else None)
    roster_authors = load_roster_authors(args.roster.resolve() if args.roster else None)
    poem_full = build_poem_cell_table_12(filtered)

    frames: list[pd.DataFrame] = []
    for stratum in want_strata:
        poem_sub = filter_poems_by_language_stratum(poem_full, stratum)
        qdf = fit_one_vs_rest_models(
            poem_sub,
            roster_authors,
            args.min_total_per_poem,
            language_stratum=stratum,
        )
        if not qdf.empty:
            frames.append(qdf)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    poem_full.to_csv(out_dir / "q1_poem_unit_cell_counts_12.csv", index=False)
    combined.to_csv(out_dir / "q1_poem_per_cell_glm_by_language.csv", index=False)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Q1 Per-cell one-vs-rest GLM (1st/2nd person, poem level)\n\n")
        f.write("- Cells: `1sg, 1pl, 2sg, 2pl` only. Denominator per poem = sum of those counts (3rd person excluded).\n")
        f.write("- Unit: poem. Language strata: `pooled_Ukrainian_Russian`, `Ukrainian`, `Russian` (exact `language` tags).\n")
        f.write("- Model: `k + n_other ~ C(period3)`; SE clustered by author. BH within each language stratum.\n")

    print(f"Wrote Q1 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
