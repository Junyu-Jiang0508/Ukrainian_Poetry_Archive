"""Export token-level pronoun_annotation rows for roster-included authors (analysis filters).

Applies the same preprocessing as ``02_modeling_q1_per_cell_glm.load_and_filter`` and
keeps rows whose ``author`` appears in the roster with ``included`` true. Writes a CSV
under ``data/`` for inspection or downstream tools.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "data" / "analysis_subsets" / "pronoun_annotation_roster_included.csv"


def _load_q1_module():
    path = ROOT / "src" / "02_modeling_q1_per_cell_glm.py"
    spec = importlib.util.spec_from_file_location("_q1_per_cell_glm_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export pronoun_annotation.csv filtered like Q1 + roster-included authors."
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    ap.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--no-roster",
        action="store_true",
        help="Skip roster filter; only apply load_and_filter exclusions.",
    )
    ap.add_argument(
        "--authors",
        type=str,
        default="",
        help="Comma-separated author names (exact strings). If set, overrides --roster.",
    )
    args = ap.parse_args()

    q1 = _load_q1_module()
    filtered = q1.load_and_filter(args.input.resolve(), args.layer0.resolve())

    manual = tuple(a.strip() for a in args.authors.split(",") if a.strip())
    if manual:
        authors = set(manual)
        out_df = filtered[filtered["author"].astype(str).isin(authors)].copy()
        missing = sorted(authors - set(out_df["author"].astype(str).unique()))
        if missing:
            print("Warning: these --authors never appear in filtered data:", "; ".join(missing))
    elif args.no_roster:
        out_df = filtered
    else:
        roster_authors = q1.load_roster_authors(args.roster.resolve() if args.roster else None)
        if roster_authors is None:
            raise SystemExit(
                f"Roster not usable (missing columns or file): {args.roster}. "
                "Use --no-roster to export without author filter, or fix --roster path."
            )
        out_df = filtered[filtered["author"].astype(str).isin(roster_authors)].copy()

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    n_authors = out_df["author"].astype(str).nunique()
    n_poems = out_df["poem_id"].astype(str).nunique()
    n_rows = len(out_df)
    print(f"Wrote {out_path} ({n_rows} rows, {n_poems} poems, {n_authors} authors)")


if __name__ == "__main__":
    main()
