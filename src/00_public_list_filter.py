"""Build the public-list corpus subset of ukrpoetry_database.csv."""
from __future__ import annotations

import argparse
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
_DATA_RAW = _ROOT / "data" / "raw"
_DATA_PROCESSED = _ROOT / "data" / "processed"

DEFAULT_AUTHOR_XLSX = _DATA_RAW / "author.xlsx"
DEFAULT_DATABASE_CSV = _DATA_RAW / "ukrpoetry_database.csv"
DEFAULT_CORPUS_OUT = _DATA_PROCESSED / "ukrpoetry_database_public_list.csv"

PRONOUN_DETAILED = _ROOT / "outputs" / "01_annotation_pronoun_detection" / "ukrainian_pronouns_detailed.csv"
PRONOUN_PROJECTION = _ROOT / "outputs" / "01_annotation_pronoun_detection" / "ukrainian_pronouns_projection_final.csv"
GPT_DETAILED = _ROOT / "outputs" / "01_annotation_pronoun_detection" / "gpt_annotation_detailed.csv"

AUTHOR_OF_POEM_COL = "Author of poem"
LANGUAGE_COL = "Language"
ORIGINAL_LANGUAGE_COL = "Original language (if post is a translation)"


def filter_database_to_public_corpus(
    df: pd.DataFrame,
    allowed_authors: set[str],
    *,
    ukrainian_only: bool = False,
    exclude_translation_posts: bool = True,
) -> pd.DataFrame:
    """Return rows whose author is in ``allowed_authors``."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if AUTHOR_OF_POEM_COL not in df.columns:
        raise KeyError(
            f"database missing {AUTHOR_OF_POEM_COL!r} (have {list(df.columns)})"
        )
    author_col = df[AUTHOR_OF_POEM_COL].fillna("").astype(str).map(normalize_author_name)
    keep = author_col.isin(allowed_authors)
    if exclude_translation_posts:
        if ORIGINAL_LANGUAGE_COL not in df.columns:
            print(
                f"warning: missing {ORIGINAL_LANGUAGE_COL!r}; "
                "not excluding translation posts",
                file=sys.stderr,
            )
        else:
            not_translation = ~df[ORIGINAL_LANGUAGE_COL].map(has_translation_marker)
            keep = keep & not_translation
    if ukrainian_only:
        if LANGUAGE_COL not in df.columns:
            raise KeyError(
                f"ukrainian_only=True but missing {LANGUAGE_COL!r} "
                f"(have {list(df.columns)})"
            )
        lang_ok = (
            df[LANGUAGE_COL].fillna("").astype(str).str.strip().str.lower() == "ukrainian"
        )
        keep = keep & lang_ok
    return df.loc[keep].copy()


def _filter_csv_by_ids(
    src: Path,
    id_set: set[str],
    dest: Path,
    *,
    id_column: str = "ID",
) -> int | None:
    if not src.is_file():
        return None
    sub = pd.read_csv(src, low_memory=False)
    if id_column not in sub.columns:
        print(f"skip (no {id_column!r}): {src}", file=sys.stderr)
        return None
    sub = sub[sub[id_column].astype(str).isin(id_set)].copy()
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dest, index=False, encoding="utf-8-sig")
    return len(sub)


def _filter_csv_by_author(
    src: Path,
    allowed: set[str],
    dest: Path,
    *,
    author_column: str = "author",
) -> int | None:
    if not src.is_file():
        return None
    sub = pd.read_csv(src, low_memory=False)
    if author_column not in sub.columns:
        print(f"skip (no {author_column!r}): {src}", file=sys.stderr)
        return None
    auth = sub[author_column].fillna("").astype(str).map(normalize_author_name)
    sub = sub[auth.isin(allowed)].copy()
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dest, index=False, encoding="utf-8-sig")
    return len(sub)


def write_derivative_public_lists(id_set: set[str], allowed_authors: set[str]) -> None:
    """Sync pronoun / GPT exports under data/processed/ when source files exist."""
    pairs: list[tuple[Path, Path, str]] = [
        (PRONOUN_DETAILED, _DATA_PROCESSED / "ukrainian_pronouns_detailed_public_list.csv", "ids"),
        (
            PRONOUN_PROJECTION,
            _DATA_PROCESSED / "ukrainian_pronouns_projection_final_public_list.csv",
            "ids",
        ),
    ]
    for src, dest, mode in pairs:
        if mode == "ids":
            n = _filter_csv_by_ids(src, id_set, dest)
            if n is not None:
                print(f"wrote {dest.name} rows={n}")
            else:
                print(f"skip missing: {src}")

    gpt_dest = _DATA_PROCESSED / "gpt_annotation_detailed_public_list.csv"
    n = _filter_csv_by_author(GPT_DETAILED, allowed_authors, gpt_dest)
    if n is not None:
        print(f"wrote {gpt_dest.name} rows={n}")
    else:
        print(f"skip missing: {GPT_DETAILED}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build public-list corpus from ukrpoetry_database.csv + author.xlsx."
    )
    p.add_argument(
        "--author-xlsx",
        type=Path,
        default=DEFAULT_AUTHOR_XLSX,
        help=f"Author metadata (default: {DEFAULT_AUTHOR_XLSX})",
    )
    p.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_CSV,
        help=f"Full database CSV (default: {DEFAULT_DATABASE_CSV})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CORPUS_OUT,
        help=f"Filtered corpus CSV (default: {DEFAULT_CORPUS_OUT})",
    )
    p.add_argument(
        "--ukrainian-only",
        action="store_true",
        help="Also require Language=ukrainian (default: no Language column filter).",
    )
    p.add_argument(
        "--include-translation-posts",
        action="store_true",
        help=(
            "Keep rows with non-empty Original language (if post is a translation); "
            "default is to drop them to avoid duplicate originals vs translations."
        ),
    )
    p.add_argument(
        "--corpus-only",
        action="store_true",
        help="Only write the main corpus CSV; skip pronoun/GPT derivative exports.",
    )
    args = p.parse_args(argv)

    author_xlsx = args.author_xlsx.resolve()
    database_csv = args.database.resolve()
    out_csv = args.output.resolve()

    if not author_xlsx.is_file():
        print(f"missing: {author_xlsx}", file=sys.stderr)
        return 1
    if not database_csv.is_file():
        print(f"missing: {database_csv}", file=sys.stderr)
        return 1

    allowed = load_allowed_author_names(author_xlsx)
    print(f"allowed author name keys: {len(allowed)}")

    df_full = pd.read_csv(database_csv, low_memory=False)
    df_pub = filter_database_to_public_corpus(
        df_full,
        allowed,
        ukrainian_only=args.ukrainian_only,
        exclude_translation_posts=not args.include_translation_posts,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pub.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"wrote {out_csv} rows={len(df_pub)}")

    if "ID" not in df_pub.columns:
        print("warning: corpus has no ID column; skipping derivatives", file=sys.stderr)
        return 0

    id_set = set(df_pub["ID"].astype(str))
    print(f"distinct poem IDs: {len(id_set)}")

    if not args.corpus_only:
        write_derivative_public_lists(id_set, allowed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
