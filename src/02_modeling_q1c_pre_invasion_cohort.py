"""Q1c: Q1 GLM restricted to authors with first observed year ≤ 2014 (pre-invasion cohort).

Exploratory cohort restriction motivated post-hoc by the aggregate-null observed in
Q1 main. The intent is to isolate war-induced shift from war-induced cohort entry —
i.e. ask "did the same peace-time poets change?" rather than "did the corpus average
change?" Authors whose first observed year exceeds 2014 (entered writing after the
2014 mobilization) are removed.

Not in the main BH family (`is_primary_stratum=False` on every row → q-value NaN).
The existing robustness-period file
`outputs/02_modeling_robustness_period/q1_poem_per_cell_glm_robust_period_author_onset_le2014.csv`
is intentionally NOT deleted; this script writes a parallel, more reviewer-friendly
output with cohort author lists exposed for inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum
from utils.finite_verb_exposure import resolve_finite_verb_counts_for_modeling
from utils.poem_cell_counts import build_poem_cell_table_with_exposure
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_q1c_pre_invasion_cohort_glm"
COHORT_ONSET_THRESHOLD = 2014


def _load_q1():
    path = ROOT / "src" / "02_modeling_q1_per_cell_glm.py"
    spec = importlib.util.spec_from_file_location("_q1_q1c_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_period_specs():
    path = ROOT / "src" / "02_modeling_robustness_period_definitions.py"
    spec = importlib.util.spec_from_file_location("_robp_for_q1c", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cohort_roster_for_stratum(poem_tbl: pd.DataFrame, language_stratum: str) -> pd.DataFrame:
    """Per-author roster for one stratum: language_stratum, author, n_poems_in_cohort,
    rank_within_stratum (1 = most poems). Sorted by descending n_poems."""
    sub = filter_poems_by_language_stratum(poem_tbl, language_stratum)
    if "include_in_offset_models" in sub.columns:
        sub = sub.loc[sub["include_in_offset_models"].astype(bool)]
    if sub.empty:
        return pd.DataFrame(
            columns=["language_stratum", "author", "n_poems_in_cohort", "rank_within_stratum"]
        )
    counts = (
        sub.groupby("author").size().rename("n_poems_in_cohort").sort_values(ascending=False)
    )
    out = counts.reset_index()
    out["language_stratum"] = language_stratum
    out["rank_within_stratum"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out[["language_stratum", "author", "n_poems_in_cohort", "rank_within_stratum"]]


def _write_readme(out_dir: Path) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        "# Q1c Pre-invasion Cohort GLM (exploratory)\n\n"
        "- Cohort definition: authors whose first observed year in the corpus is ≤ "
        f"{COHORT_ONSET_THRESHOLD}. The intent is to isolate war-induced shift from "
        "war-induced cohort entry — i.e. ask whether the same peace-time poets changed, "
        "rather than whether the corpus average changed.\n"
        "- Inference status: **exploratory cohort restriction motivated post-hoc by the "
        "aggregate null in Q1 main; not in the main BH family.** `is_primary_stratum` is "
        "False on every row → `q_value_bh_within_stratum` is NaN.\n"
        "- Cells (4-cell, frequentist): `{1sg, 1pl, 2sg, 2pl_vy_true_plural}`. "
        "Polite-singular ви is excluded on the frequentist path (see Q1 README); Q2 "
        "Bayesian path retains it.\n"
        "- Strata: pooled (descriptive), Ukrainian, Russian.\n"
        "- Model: same as Q1 main — Poisson with `offset(log exposure_n_stanzas)` and "
        "cluster-robust SE by author.\n\n"
        "## Files\n\n"
        "- `q1c_pre_invasion_cohort_glm.csv` — one row per (stratum × cell). "
        "`cohort_n_authors` reports the cohort size for that stratum.\n"
        "- `q1c_cohort_author_roster.csv` — one row per (stratum × author): "
        "`language_stratum, author, n_poems_in_cohort, rank_within_stratum`. "
        "Machine-readable; meant for forest plots and reviewer spot-checks.\n\n"
        "## Sibling file\n\n"
        "`outputs/02_modeling_robustness_period/q1_poem_per_cell_glm_robust_period_author_onset_le2014.csv` "
        "contains the same numerical estimates under sensitivity-context framing and is "
        "kept in place; this Q1c directory is the canonical citation for the discussion "
        "section.\n",
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Q1c exploratory pre-invasion cohort GLM.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = out_dir / "language_stratum_audit"

    q1 = _load_q1()
    robp = _load_period_specs()

    filtered = q1.load_and_filter(
        q1.DEFAULT_INPUT.resolve(),
        q1.DEFAULT_LAYER0.resolve() if q1.DEFAULT_LAYER0.is_file() else None,
        language_audit_dir=audit,
    )
    layer0_path = q1.DEFAULT_LAYER0 if q1.DEFAULT_LAYER0.is_file() else None
    roster = q1.load_roster_authors(q1.DEFAULT_ROSTER.resolve() if q1.DEFAULT_ROSTER.is_file() else None)

    cohort_filtered = robp.spec_author_onset_le2014(filtered.copy(), layer0_path)
    fv_df = resolve_finite_verb_counts_for_modeling(ROOT, exposure_type="n_stanzas")
    poem_tbl = build_poem_cell_table_with_exposure(cohort_filtered, finite_verb_df=fv_df)

    roster_parts: list[pd.DataFrame] = []
    frames: list[pd.DataFrame] = []
    for stratum in LANGUAGE_STRATA:
        roster_df = _cohort_roster_for_stratum(poem_tbl, stratum)
        if not roster_df.empty:
            roster_parts.append(roster_df)
        n_auth = int(roster_df.shape[0])

        sub = filter_poems_by_language_stratum(poem_tbl, stratum)
        g = q1.fit_q1_poisson_per_cell(
            sub,
            roster,
            0,
            language_stratum=stratum,
            exposure_type="n_stanzas",
        )
        if g.empty:
            continue
        g = g.copy()
        # Demote BH: this contrast is exploratory by construction.
        g["is_primary_stratum"] = False
        g["q_value_bh_within_stratum"] = float("nan")
        g["cohort_definition"] = f"author_first_observed_year_le_{COHORT_ONSET_THRESHOLD}"
        g["cohort_n_authors"] = n_auth
        frames.append(g)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = out_dir / "q1c_pre_invasion_cohort_glm.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows)")

    roster_df_all = pd.concat(roster_parts, ignore_index=True) if roster_parts else pd.DataFrame()
    roster_path = out_dir / "q1c_cohort_author_roster.csv"
    roster_df_all.to_csv(roster_path, index=False)
    print(f"Wrote {roster_path} ({len(roster_df_all)} rows)")

    _write_readme(out_dir)


if __name__ == "__main__":
    main()
