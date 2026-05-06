"""Filter To_run layer0/layer1 CSVs for public corpus uniqueness."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from utils.public_list_filters import (
    has_translation_marker,
    load_allowed_author_names,
    normalize_author_name,
)
from utils.workspace import repository_root

_ROOT = repository_root()
_DEFAULT_LAYER0 = _ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
_DEFAULT_LAYER1 = _ROOT / "data" / "To_run" / "00_filtering" / "layer1_stanzas_one_per_row.csv"
_DEFAULT_AUTHOR = _ROOT / "data" / "raw" / "author.xlsx"
_DEFAULT_REVIEW_QUEUE = _ROOT / "data" / "To_run" / "00_filtering" / "layer0_human_review_queue.csv"
_DEFAULT_REVIEW_JSON = _ROOT / "data" / "To_run" / "00_filtering" / "layer0_human_review_queue_for_gpt.json"

AUTHOR_OF_POEM_COL = "Author of poem"
ORIGINAL_LANGUAGE_COL = "Original language (if post is a translation)"
URL_ORIGINAL_COL = "URL of original (if poem is a translation)"
POEM_ID_COL = "poem_id"


def filter_layer0(
    df: pd.DataFrame,
    allowed_authors: set[str],
) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if AUTHOR_OF_POEM_COL not in df.columns:
        raise KeyError(f"missing {AUTHOR_OF_POEM_COL!r}")
    if POEM_ID_COL not in df.columns:
        raise KeyError(f"missing {POEM_ID_COL!r}")
    author_ok = (
        df[AUTHOR_OF_POEM_COL]
        .fillna("")
        .astype(str)
        .map(normalize_author_name)
        .isin(allowed_authors)
    )
    trans_cols = []
    if ORIGINAL_LANGUAGE_COL in df.columns:
        trans_cols.append(df[ORIGINAL_LANGUAGE_COL].map(has_translation_marker))
    if URL_ORIGINAL_COL in df.columns:
        trans_cols.append(df[URL_ORIGINAL_COL].map(has_translation_marker))
    if trans_cols:
        is_translation = trans_cols[0]
        for c in trans_cols[1:]:
            is_translation = is_translation | c
    else:
        is_translation = pd.Series(False, index=df.index)
    keep = author_ok & ~is_translation
    return df.loc[keep].copy()


def sync_human_review_artifacts(
    layer0: Path,
    review_csv: Path,
    review_json: Path | None,
    *,
    backup: bool,
) -> int:
    """Drop review-queue rows whose parent_id no longer exists in layer0 (and JSON items)."""
    layer0 = layer0.resolve()
    review_csv = review_csv.resolve()
    if not layer0.is_file():
        print(f"missing layer0: {layer0}", file=sys.stderr)
        return 1
    if not review_csv.is_file():
        print(f"missing review queue: {review_csv}", file=sys.stderr)
        return 1

    df0 = pd.read_csv(layer0, low_memory=False)
    df0.columns = df0.columns.str.strip()
    if "parent_id" not in df0.columns:
        print("layer0 has no parent_id column", file=sys.stderr)
        return 1
    parents = set(df0["parent_id"].astype(str))

    q = pd.read_csv(review_csv, low_memory=False)
    if "parent_id" not in q.columns:
        print("review queue has no parent_id column", file=sys.stderr)
        return 1
    n_q = len(q)
    q_f = q[q["parent_id"].astype(str).isin(parents)].copy()

    if backup:
        shutil.copy2(review_csv, review_csv.with_suffix(review_csv.suffix + ".bak"))

    q_f.to_csv(review_csv, index=False, encoding="utf-8-sig")
    print(f"review_queue: {n_q} -> {len(q_f)} (removed {n_q - len(q_f)})")

    if review_json is not None:
        jp = review_json.resolve()
        if jp.is_file():
            payload = json.loads(jp.read_text(encoding="utf-8"))
            items = payload.get("items")
            if isinstance(items, list):
                n_j = len(items)
                kept = [x for x in items if str(x.get("parent_id", "")) in parents]
                payload["items"] = kept
                payload["row_count"] = len(kept)
                if backup:
                    shutil.copy2(jp, jp.with_suffix(jp.suffix + ".bak"))
                jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"review_json: {n_j} -> {len(kept)} (removed {n_j - len(kept)})")
            else:
                print(f"skip json (no items array): {jp}", file=sys.stderr)
        else:
            print(f"skip missing json: {jp}", file=sys.stderr)

    if backup:
        parts = [f"{review_csv.name}.bak"]
        if review_json is not None and review_json.resolve().is_file():
            parts.append(f"{review_json.name}.bak")
        print("backups:", ", ".join(parts))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "To rebuild layer0/layer1/review_queue from the raw database (overwrites To_run outputs), run:\n"
            "  PYTHONPATH=src python3 src/00_filtering.py\n"
            "Use --sync-review-queue-only after trimming layer0 so the human-review CSV/JSON match parent_ids."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--layer0", type=Path, default=_DEFAULT_LAYER0)
    p.add_argument("--layer1", type=Path, default=_DEFAULT_LAYER1)
    p.add_argument("--author-xlsx", type=Path, default=_DEFAULT_AUTHOR)
    p.add_argument(
        "--review-queue",
        type=Path,
        default=_DEFAULT_REVIEW_QUEUE,
        help="layer0_human_review_queue.csv (used with --sync-review-queue-only).",
    )
    p.add_argument(
        "--review-json",
        type=Path,
        default=_DEFAULT_REVIEW_JSON,
        help="layer0_human_review_queue_for_gpt.json (used with --sync-review-queue-only).",
    )
    p.add_argument(
        "--sync-review-queue-only",
        action="store_true",
        help="Only align human-review CSV/JSON to current layer0 parent_ids; do not filter layer0/layer1.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak copies before overwriting inputs.",
    )
    args = p.parse_args(argv)

    layer0 = args.layer0.resolve()
    layer1 = args.layer1.resolve()
    author_xlsx = args.author_xlsx.resolve()

    if args.sync_review_queue_only:
        return sync_human_review_artifacts(
            layer0,
            args.review_queue,
            args.review_json,
            backup=not args.no_backup,
        )

    if not layer0.is_file() or not layer1.is_file():
        print("missing layer0 or layer1 input", file=sys.stderr)
        return 1
    if not author_xlsx.is_file():
        print(f"missing {author_xlsx}", file=sys.stderr)
        return 1

    allowed = load_allowed_author_names(author_xlsx)
    df0 = pd.read_csv(layer0, low_memory=False)
    n0 = len(df0)
    df0_f = filter_layer0(df0, allowed)
    kept_ids = set(df0_f[POEM_ID_COL].astype(str))

    df1 = pd.read_csv(layer1, low_memory=False)
    n1 = len(df1)
    df1_f = df1[df1[POEM_ID_COL].astype(str).isin(kept_ids)].copy()

    if not args.no_backup:
        shutil.copy2(layer0, layer0.with_suffix(layer0.suffix + ".bak"))
        shutil.copy2(layer1, layer1.with_suffix(layer1.suffix + ".bak"))

    df0_f.to_csv(layer0, index=False, encoding="utf-8-sig")
    df1_f.to_csv(layer1, index=False, encoding="utf-8-sig")

    print(f"layer0: {n0} -> {len(df0_f)} (removed {n0 - len(df0_f)})")
    print(f"layer1: {n1} -> {len(df1_f)} (removed {n1 - len(df1_f)})")
    print(f"allowed author keys: {len(allowed)}")
    if not args.no_backup:
        print(f"backups: {layer0.name}.bak, {layer1.name}.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
