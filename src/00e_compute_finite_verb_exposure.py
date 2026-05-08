"""Stage 00e: precompute Stanza finite-verb counts per stanza (run once, persist to CSV).

Writes ``data/To_run/00_filtering/stanza_finite_verb_counts.csv`` for downstream
Q1/Q2/Q1b joins (avoids re-parsing the corpus on every modeling stage).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils.finite_verb_exposure import compute_finite_verb_counts_table, init_stanza_finite_verb_pipeline
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_LAYER1 = ROOT / "data" / "To_run" / "00_filtering" / "layer1_stanzas_one_per_row.csv"
DEFAULT_OUTPUT = ROOT / "data" / "To_run" / "00_filtering" / "stanza_finite_verb_counts.csv"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Precompute per-stanza finite-verb counts (Stanza, Ukrainian).")
    ap.add_argument("--input", type=Path, default=DEFAULT_LAYER1, help="layer1 stanzas CSV")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N unique stanzas (smoke test)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()

    out_path = args.output.resolve()
    if out_path.is_file() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out_path} (use --force)")

    layer1 = args.input.resolve()
    if not layer1.is_file():
        raise SystemExit(f"Missing layer1 input: {layer1}")

    df = pd.read_csv(layer1, low_memory=False)
    if "poem_id" not in df.columns or "stanza_index" not in df.columns:
        raise SystemExit(f"{layer1} must have poem_id and stanza_index")
    text_col = "stanza_text" if "stanza_text" in df.columns else "stanza_ukr"
    if text_col not in df.columns:
        raise SystemExit(f"{layer1} must have stanza_text or stanza_ukr")

    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["poem_id", "stanza_index"], keep="first")
    df = df.sort_values(["poem_id", "stanza_index"]).reset_index(drop=True)
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()
        log.info("Limiting to first %s rows after dedup", args.limit)

    nlp = init_stanza_finite_verb_pipeline()
    work = df[["poem_id", "stanza_index", text_col]].rename(columns={text_col: "stanza_text"})
    counts = compute_finite_verb_counts_table(work, nlp, text_col="stanza_text")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_path, index=False)
    log.info("Wrote %s (%s stanza rows)", out_path, len(counts))


if __name__ == "__main__":
    main()
