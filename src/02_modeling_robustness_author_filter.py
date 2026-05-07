"""Robustness: Q1 Poisson+offset while varying min poems-per-period author inclusion (selection sensitivity)."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import pandas as pd

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"


def _load_q1():
    path = ROOT / "src" / "02_modeling_q1_per_cell_glm.py"
    spec = importlib.util.spec_from_file_location("_q1_rob_auth", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eligible_authors_by_min_per_period(poem_tbl: pd.DataFrame, minimum: int) -> set[str]:
    """Authors with at least ``minimum`` distinct poems in P1 and in P2 (calendar strata)."""
    sub = poem_tbl.loc[poem_tbl["period3"].isin((PERIOD_P1, PERIOD_P2))].copy()
    counts = (
        sub.groupby(["author", "period3"])["poem_id"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns=lambda c: str(c))
    )
    for col in (PERIOD_P1, PERIOD_P2):
        if col not in counts.columns:
            counts[col] = 0
    ok = counts[PERIOD_P1].ge(int(minimum)) & counts[PERIOD_P2].ge(int(minimum))
    return set(counts.index[ok].astype(str).tolist())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-per-period-scan",
        type=str,
        default="3,5,10,15",
        help="Comma-separated min poems per calendar period thresholds",
    )
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "02_modeling_robustness_author_threshold")
    args = ap.parse_args()

    thresholds = tuple(int(x.strip()) for x in args.min_per_period_scan.split(",") if x.strip())

    q1 = _load_q1()
    out_base = args.output.resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    audit = out_base / "language_stratum_audit"

    from utils.poem_cell_counts import build_poem_cell_table_with_exposure

    filtered = q1.load_and_filter(
        q1.DEFAULT_INPUT.resolve(),
        q1.DEFAULT_LAYER0.resolve() if q1.DEFAULT_LAYER0.is_file() else None,
        language_audit_dir=audit,
    )
    poem_full = build_poem_cell_table_with_exposure(filtered)
    roster = q1.load_roster_authors(q1.DEFAULT_ROSTER.resolve() if q1.DEFAULT_ROSTER.is_file() else None)

    all_rows: list[pd.DataFrame] = []
    for m in thresholds:
        auth_ok = eligible_authors_by_min_per_period(poem_full, m)
        poem_m = poem_full.loc[poem_full["author"].astype(str).isin(auth_ok)].copy()
        summary = {
            "min_poems_per_period": m,
            "n_authors_eligible": len(auth_ok),
            "n_poems": int(poem_m["poem_id"].nunique()),
        }
        frames = []
        for stratum in LANGUAGE_STRATA:
            sub = filter_poems_by_language_stratum(poem_m, stratum)
            g = q1.fit_q1_poisson_per_cell(
                sub,
                roster,
                0,
                language_stratum=stratum,
                exposure_type="n_stanzas",
            )
            if not g.empty:
                g = g.copy()
                for k, v in summary.items():
                    g[k] = v
                frames.append(g)
        if frames:
            all_rows.append(pd.concat(frames, ignore_index=True))
        else:
            all_rows.append(pd.DataFrame([summary]))

    comb = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    path = out_base / "q1_poem_per_cell_glm_robust_author_min_per_period.csv"
    comb.to_csv(path, index=False)
    print(f"Wrote {path} ({len(comb)} rows)")


if __name__ == "__main__":
    main()
