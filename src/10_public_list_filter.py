"""Filter corpus to authors marked Yes in data/raw/author.xlsx \"Include in public list\". Writes under data/processed/."""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
AUTHOR_XLSX = os.path.join(DATA_RAW, "author.xlsx")
DATABASE_CSV = os.path.join(DATA_RAW, "ukrpoetry_database.csv")
PRONOUN_DETAILED = os.path.join(ROOT, "outputs", "01_pronoun_detection", "ukrainian_pronouns_detailed.csv")
PRONOUN_PROJECTION = os.path.join(ROOT, "outputs", "01_pronoun_detection", "ukrainian_pronouns_projection_final.csv")
GPT_DETAILED = os.path.join(ROOT, "outputs", "01_pronoun_detection", "gpt_annotation_detailed.csv")

ALIAS_COLS = [
    "Author",
    "Facebook name",
    "Alias (English transliteration)",
    "Alias (Russian)",
    "Alias (Ukrainian)",
]


def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().split())


def _is_public_yes(value) -> bool:
    if pd.isna(value):
        return False
    v = str(value).strip().lower()
    return v in ("yes", "y", "1", "true")


def load_allowed_author_names(path: str) -> set[str]:
    df = pd.read_excel(path)
    pub = df[df["Include in public list"].map(_is_public_yes)]
    names: set[str] = set()
    for col in ALIAS_COLS:
        if col not in pub.columns:
            continue
        for v in pub[col].dropna().astype(str):
            t = _norm_name(v)
            if t and t.lower() != "nan":
                names.add(t)
    return names


def filter_database_csv(path: str, allowed: set[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    lang_ok = df["Language"].fillna("").astype(str).str.strip().str.lower() == "ukrainian"
    author_col = df["Author of poem"].fillna("").astype(str).map(_norm_name)
    keep = lang_ok & author_col.isin(allowed)
    return df.loc[keep].copy(), keep


def filter_csv_by_ids(path: str, id_set: set[str], out_path: str, id_column: str = "ID") -> int | None:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    if id_column not in df.columns:
        print(f"skip no column {id_column}: {path}")
        return None
    sub = df[df[id_column].astype(str).isin(id_set)].copy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8-sig")
    return len(sub)


def filter_csv_by_author(path: str, allowed: set[str], out_path: str, author_column: str = "author") -> int | None:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    if author_column not in df.columns:
        print(f"skip no column {author_column}: {path}")
        return None
    auth = df[author_column].fillna("").astype(str).map(_norm_name)
    sub = df[auth.isin(allowed)].copy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8-sig")
    return len(sub)


def main() -> None:
    if not os.path.isfile(AUTHOR_XLSX):
        print(f"missing: {AUTHOR_XLSX}")
        sys.exit(1)
    if not os.path.isfile(DATABASE_CSV):
        print(f"missing: {DATABASE_CSV}")
        sys.exit(1)

    allowed = load_allowed_author_names(AUTHOR_XLSX)
    print(f"allowed author keys: {len(allowed)}")

    df_pub, _ = filter_database_csv(DATABASE_CSV, allowed)
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    out_db = os.path.join(DATA_PROCESSED, "ukrpoetry_database_public_list.csv")
    df_pub.to_csv(out_db, index=False, encoding="utf-8-sig")
    print(f"wrote {out_db} rows={len(df_pub)}")

    id_set = set(df_pub["ID"].astype(str))
    print(f"poem ids: {len(id_set)}")

    n = filter_csv_by_ids(
        PRONOUN_DETAILED,
        id_set,
        os.path.join(DATA_PROCESSED, "ukrainian_pronouns_detailed_public_list.csv"),
    )
    if n is not None:
        print(f"wrote ukrainian_pronouns_detailed_public_list.csv n={n}")
    else:
        print("skip: ukrainian_pronouns_detailed.csv missing")

    n = filter_csv_by_ids(
        PRONOUN_PROJECTION,
        id_set,
        os.path.join(DATA_PROCESSED, "ukrainian_pronouns_projection_final_public_list.csv"),
    )
    if n is not None:
        print(f"wrote ukrainian_pronouns_projection_final_public_list.csv n={n}")
    else:
        print("skip: ukrainian_pronouns_projection_final.csv missing")

    n = filter_csv_by_author(
        GPT_DETAILED,
        allowed,
        os.path.join(DATA_PROCESSED, "gpt_annotation_detailed_public_list.csv"),
    )
    if n is not None:
        print(f"wrote gpt_annotation_detailed_public_list.csv n={n}")
    else:
        print("skip: gpt_annotation_detailed.csv missing")


if __name__ == "__main__":
    main()
