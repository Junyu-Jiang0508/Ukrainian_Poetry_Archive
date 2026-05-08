"""Build poem-level pronoun ratio index tables used by ratio analyses."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "outputs" / "02_modeling_q1_per_cell_glm" / "q1_poem_unit_cell_counts_12.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_ratio_indices"
PERIODS = ("P1_2014_2021", "P2_2022_plus")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=num.index, dtype=float)
    m = den.gt(0)
    out.loc[m] = num.loc[m] / den.loc[m]
    return out


def _load_roster(roster_path: Path) -> set[str] | None:
    if not roster_path.is_file():
        return None
    rdf = pd.read_csv(roster_path, low_memory=False)
    if "author" not in rdf.columns or "included" not in rdf.columns:
        return None
    return set(rdf.loc[rdf["included"].astype(bool), "author"].astype(str))


def _build_ratio_table(df: pd.DataFrame, *, include_polite_vy: bool) -> pd.DataFrame:
    out = df.copy()
    out["n1p_suc"] = out["1pl"].astype(float)
    out["n1p_tri"] = out["1sg"].astype(float) + out["1pl"].astype(float)
    out["n2p_suc"] = out["2pl_vy_true_plural"].astype(float)
    out["n2p_tri"] = out["2sg"].astype(float) + out["2pl_vy_true_plural"].astype(float)
    if include_polite_vy:
        out["n2p_tri"] = out["n2p_tri"] + out["2pl_vy_polite_singular"].astype(float)
    out["nov_suc"] = out["1pl"].astype(float) + out["2pl_vy_true_plural"].astype(float)
    out["nov_tri"] = (
        out["1sg"].astype(float)
        + out["1pl"].astype(float)
        + out["2sg"].astype(float)
        + out["2pl_vy_true_plural"].astype(float)
    )
    out["fv_rate_1pl"] = _safe_div(out["1pl"].astype(float), out["exposure_n_finite_verbs"].astype(float))
    out["fv_rate_1sg"] = _safe_div(out["1sg"].astype(float), out["exposure_n_finite_verbs"].astype(float))
    out["fv_rate_2sg"] = _safe_div(out["2sg"].astype(float), out["exposure_n_finite_verbs"].astype(float))
    out["fv_rate_2pl_true"] = _safe_div(out["2pl_vy_true_plural"].astype(float), out["exposure_n_finite_verbs"].astype(float))
    out["prop_1p_collective"] = _safe_div(out["n1p_suc"], out["n1p_tri"])
    out["prop_2p_collective"] = _safe_div(out["n2p_suc"], out["n2p_tri"])
    out["prop_overall_plural"] = _safe_div(out["nov_suc"], out["nov_tri"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ratio index tables from Q1 poem-cell counts.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    in_df = pd.read_csv(args.input.resolve(), low_memory=False)
    in_df["author"] = in_df["author"].astype(str)
    in_df = in_df.loc[in_df["period3"].isin(PERIODS)].copy()

    n_start = len(in_df)
    roster = _load_roster(args.roster.resolve())
    if roster is not None:
        in_df = in_df.loc[in_df["author"].isin(roster)].copy()
    n_after_roster = len(in_df)

    fv = pd.to_numeric(in_df["exposure_n_finite_verbs"], errors="coerce")
    in_df = in_df.loc[fv.notna() & fv.gt(0)].copy()
    n_after_fv = len(in_df)
    if n_after_fv == 0:
        warnings.warn(
            "No poems pass FV gate (`exposure_n_finite_verbs > 0`). "
            "Check finite-verb exposure availability (e.g., run 00e stage).",
            stacklevel=1,
        )

    main_tbl = _build_ratio_table(in_df, include_polite_vy=False)
    sens_tbl = _build_ratio_table(in_df, include_polite_vy=True)
    main_tbl.to_csv(out_dir / "ratio_poem_level_table.csv", index=False)
    sens_tbl.to_csv(out_dir / "ratio_poem_level_table_include_polite_vy.csv", index=False)

    audit_rows: list[dict[str, object]] = []
    for ratio_key, tri_col in (
        ("ratio_1p_collective", "n1p_tri"),
        ("ratio_2p_collective", "n2p_tri"),
        ("ratio_overall_plural", "nov_tri"),
    ):
        tri = pd.to_numeric(main_tbl[tri_col], errors="coerce").fillna(0.0)
        kept = main_tbl.loc[tri.gt(0)].copy()
        audit_rows.append(
            {
                "ratio_index": ratio_key,
                "n_start_period_filtered": n_start,
                "n_after_roster": n_after_roster,
                "n_after_fv_gate": n_after_fv,
                "n_after_ratio_denom_gt0": int(len(kept)),
                "n_dropped_ratio_denom_eq0": int(len(main_tbl) - len(kept)),
                "n_authors_after_ratio_denom_gt0": int(kept["author"].nunique()) if not kept.empty else 0,
            }
        )
    pd.DataFrame(audit_rows).to_csv(out_dir / "ratio_exclusion_audit.csv", index=False)

    by_period = (
        main_tbl.assign(fv_gate_pass=True)
        .groupby(["period3", "language_clean"], dropna=False)
        .size()
        .rename("n_poems")
        .reset_index()
    )
    by_period.to_csv(out_dir / "ratio_exclusion_audit_by_period_language.csv", index=False)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Ratio poem-level builder (02brat)\n\n")
        f.write("- Input: `q1_poem_unit_cell_counts_12.csv` (canonical poem-level count table).\n")
        f.write("- Eligibility gate: roster-included authors and `exposure_n_finite_verbs > 0`.\n")
        f.write("- Main output excludes polite-singular ви from 2P denominator.\n")
        f.write("- Sensitivity output includes polite-singular ви in 2P denominator.\n")
        f.write("- Denominator-zero exclusions are model-specific and audited in CSV files.\n")
        f.write(
            f"\nExclusion cascade: start={n_start}, after_roster={n_after_roster}, after_fv_gate={n_after_fv}.\n"
        )

    print(f"Wrote ratio builder outputs to: {out_dir}")


if __name__ == "__main__":
    main()
