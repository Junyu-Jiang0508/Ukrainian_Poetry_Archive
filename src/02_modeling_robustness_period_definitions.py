"""Robustness: Q1 Poisson+offset under alternate period encodings (writes one CSV per spec)."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import pandas as pd

from utils.finite_verb_exposure import resolve_finite_verb_counts_for_modeling
from utils.poem_cell_counts import build_poem_cell_table_with_exposure
from utils.stats_common import (
    period_p1_p2_exclude_pre_2014,
    period_p1_p2_invasion_precise,
    period_three_way,
)
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)


def _load_q1():
    path = ROOT / "src" / "02_modeling_q1_per_cell_glm.py"
    spec = importlib.util.spec_from_file_location("_q1_rob_period", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _merge_posted_dates(df: pd.DataFrame, layer0: Path) -> pd.DataFrame:
    l0 = pd.read_csv(layer0, low_memory=False, usecols=["poem_id", "Date posted"])
    l0["poem_id"] = l0["poem_id"].astype(str).str.strip()
    l0["_posted"] = pd.to_datetime(l0["Date posted"], errors="coerce")
    l0 = l0.drop_duplicates(subset=["poem_id"], keep="first")
    return df.merge(l0[["poem_id", "_posted"]], on="poem_id", how="left")


def spec_primary_calendar(df: pd.DataFrame, layer0: Path | None) -> pd.DataFrame:
    out = df.copy()
    out["period3"] = out["year_int"].map(period_three_way)
    return out


def spec_triple_drop_pre2014(df: pd.DataFrame, layer0: Path | None) -> pd.DataFrame:
    out = df.copy()
    out["period3"] = out["year_int"].map(period_p1_p2_exclude_pre_2014)
    return out


def spec_invasion_20220224(df: pd.DataFrame, layer0: Path | None) -> pd.DataFrame:
    if layer0 is None or not layer0.is_file():
        raise SystemExit("layer0 required for invasion_20220224 robustness")
    out = df.copy()
    m = _merge_posted_dates(out, layer0)
    if m["_posted"].isna().mean() > 0.95:
        log.warning("Nearly all rows missing parsed Date posted; invasion spec may degenerate.")
    m["period3"] = period_p1_p2_invasion_precise(m["_posted"])
    return m.drop(columns=["_posted"], errors="ignore")


def spec_author_onset_le2014(df: pd.DataFrame, layer0: Path | None) -> pd.DataFrame:
    out = df.copy()
    out["period3"] = out["year_int"].map(period_three_way)
    onset = out.groupby(out["author"].astype(str))["year_int"].transform("min")
    out = out.loc[onset.le(2014)].copy()
    return out


SPECS = {
    "primary_calendar": spec_primary_calendar,
    "triple_drop_pre2014": spec_triple_drop_pre2014,
    "invasion_20220224": spec_invasion_20220224,
    "author_onset_le2014": spec_author_onset_le2014,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, default=",".join(SPECS), help="Comma subset of robustness specs")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "02_modeling_robustness_period")
    args = ap.parse_args()

    want = tuple(s.strip() for s in args.spec.split(",") if s.strip())
    for s in want:
        if s not in SPECS:
            raise SystemExit(f"Unknown spec {s!r}. Options: {list(SPECS)}")

    q1 = _load_q1()
    out_base = args.output.resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    audit = out_base / "language_stratum_audit"

    filtered = q1.load_and_filter(
        q1.DEFAULT_INPUT.resolve(),
        q1.DEFAULT_LAYER0.resolve() if q1.DEFAULT_LAYER0.is_file() else None,
        language_audit_dir=audit,
    )
    layer0_path = q1.DEFAULT_LAYER0 if q1.DEFAULT_LAYER0.is_file() else None

    roster = q1.load_roster_authors(q1.DEFAULT_ROSTER.resolve() if q1.DEFAULT_ROSTER.is_file() else None)
    fv_df = resolve_finite_verb_counts_for_modeling(ROOT, exposure_type="n_stanzas")

    for name in want:
        fn = SPECS[name]
        d2 = fn(filtered.copy(), layer0_path)
        poem = build_poem_cell_table_with_exposure(d2, finite_verb_df=fv_df)
        frames = []
        from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum

        for stratum in LANGUAGE_STRATA:
            sub = filter_poems_by_language_stratum(poem, stratum)
            g = q1.fit_q1_poisson_per_cell(
                sub,
                roster,
                0,
                language_stratum=stratum,
                exposure_type="n_stanzas",
            )
            if not g.empty:
                g = g.copy()
                g["robustness_period_spec"] = name
                frames.append(g)
        comb = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        path = out_base / f"q1_poem_per_cell_glm_robust_period_{name}.csv"
        comb.to_csv(path, index=False)
        print(f"Wrote {path} ({len(comb)} rows)")


if __name__ == "__main__":
    main()
