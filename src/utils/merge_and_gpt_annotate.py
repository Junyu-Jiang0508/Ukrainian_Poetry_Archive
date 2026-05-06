"""Merge two manual annotator CSVs; align GPT detailed; write confusion matrices and metrics (no API)."""
import os

import repo_bootstrap

repo_bootstrap.prepare_repo(__file__)

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

DATA_DIR = "data"
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
OUTPUT_DIR = os.path.join("outputs", "01_pronoun_detection")
FILE_JUNYU = os.path.join(ANNOTATIONS_DIR, "manual_annotation_result_Junyu.csv")
FILE_JUNYU1 = os.path.join(ANNOTATIONS_DIR, "manual_annotation_result_junyu1.csv")
GPT_DETAILED = os.path.join(OUTPUT_DIR, "gpt_annotation_detailed.csv")
MERGED_WITH_GPT = os.path.join(ANNOTATIONS_DIR, "manual_annotation_merged_with_gpt.csv")
OUTPUT_MERGED = os.path.join(ANNOTATIONS_DIR, "manual_annotation_merged.csv")


def _normalize(s: str) -> str:
    return " ".join(str(s).strip().split()).strip()


def _norm_cat(s) -> str:
    s = str(s).strip() if pd.notna(s) else ""
    if s in ("Sing", "Sg"):
        return "Singular"
    if s in ("Plur", "Pl"):
        return "Plural"
    return s


def merge_manual_annotations() -> pd.DataFrame:
    dfs = []
    for path, name in [(FILE_JUNYU, "Junyu"), (FILE_JUNYU1, "junyu1")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["annotator"] = name
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"missing {FILE_JUNYU} or {FILE_JUNYU1}")
    return pd.concat(dfs, ignore_index=True)


def load_merged_with_gpt() -> pd.DataFrame | None:
    if os.path.exists(MERGED_WITH_GPT):
        df = pd.read_csv(MERGED_WITH_GPT)
        if "person_gpt" in df.columns or "is_dropped_gpt" in df.columns:
            return df
    return None


def prepare_gpt_from_detailed(gpt_df: pd.DataFrame) -> pd.DataFrame:
    gpt_df = gpt_df.copy()
    gpt_df = gpt_df[gpt_df["source_mapping"].notna() & (gpt_df["source_mapping"] != "IMPLIED")]
    gpt_df["context_norm"] = gpt_df["context_sentence_ukr"].fillna("").astype(str).apply(_normalize)
    gpt_df["pronoun_norm"] = gpt_df["source_mapping"].fillna("").astype(str).str.strip().str.lower()
    gpt_df = gpt_df.rename(columns={
        "original_id": "ID",
        "is_pro_drop": "is_dropped_gpt",
    })
    gpt_df["person_gpt"] = gpt_df["person"].apply(_norm_cat)
    gpt_df["number_gpt"] = gpt_df["number"].apply(_norm_cat)
    return gpt_df[["ID", "context_norm", "pronoun_norm", "person_gpt", "number_gpt", "is_dropped_gpt"]].drop_duplicates(
        subset=["ID", "context_norm", "pronoun_norm"]
    )


def merge_manual_with_gpt(merged: pd.DataFrame, gpt: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["context_norm"] = merged["context"].fillna("").astype(str).apply(_normalize)
    merged["pronoun_norm"] = merged["pronoun"].fillna("").astype(str).str.strip().str.lower()
    merged["ID"] = merged["ID"].astype(str)
    gpt["ID"] = gpt["ID"].astype(str)

    df = merged.merge(gpt, on=["ID", "context_norm", "pronoun_norm"], how="left")
    return df


def save_confusion_matrix(y_true, y_pred, labels, name: str, out_dir: str) -> str | None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.index.name = "actual"
    path = os.path.join(out_dir, f"confusion_matrix_{name}.csv")
    try:
        df_cm.to_csv(path, encoding="utf-8-sig")
        return path
    except PermissionError:
        print(f"permission denied: {path}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("merge manual annotators...")
    merged = merge_manual_annotations()
    merged.to_csv(OUTPUT_MERGED, index=False, encoding="utf-8-sig")
    print(f"saved {OUTPUT_MERGED} n={len(merged)}")

    df = load_merged_with_gpt()
    if df is not None:
        print("load merged+GPT file")
    else:
        print("load gpt_annotation_detailed.csv")
        if not os.path.exists(GPT_DETAILED):
            print(f"missing {GPT_DETAILED}")
            return
        gpt_raw = pd.read_csv(GPT_DETAILED)
        gpt_explicit = gpt_raw[gpt_raw["source_mapping"].notna() & (gpt_raw["source_mapping"] != "IMPLIED")]
        print(f"gpt rows={len(gpt_raw)} explicit={len(gpt_explicit)}")
        gpt = prepare_gpt_from_detailed(gpt_raw)
        df = merge_manual_with_gpt(merged, gpt)
        print(f"manual rows={len(merged)}")

    has_gpt = df["person_gpt"].notna() | df["number_gpt"].notna() | (df["is_dropped_gpt"].notna() if "is_dropped_gpt" in df.columns else False)
    if "is_dropped_gpt" not in df.columns:
        has_gpt = df["person_gpt"].notna() | df["number_gpt"].notna()
    matched_count = has_gpt.sum()
    unmatched_count = (~has_gpt).sum()
    df = df[has_gpt].copy()
    if df.empty:
        print("no gpt rows to compare")
        return

    print(f"matched={matched_count}")
    if unmatched_count > 0:
        print(f"unmatched={unmatched_count}")

    df["person_norm"] = df["person"].apply(_norm_cat)
    df["number_norm"] = df["number"].apply(_norm_cat)
    if "person_gpt" in df.columns:
        df["person_gpt_norm"] = df["person_gpt"].apply(_norm_cat)
    if "number_gpt" in df.columns:
        df["number_gpt_norm"] = df["number_gpt"].apply(_norm_cat)
    manual_drop = df["is_dropped"].map(lambda x: bool(x) if pd.notna(x) else False)
    gpt_drop = df["is_dropped_gpt"].map(lambda x: bool(x) if pd.notna(x) else False)

    print("confusion matrices + metrics...")
    metrics_rows = []
    saved = []

    y_true = manual_drop.astype(int)
    y_pred = gpt_drop.astype(int)
    labels_drop = [0, 1]
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels_drop, zero_division=0)
    for i, label in enumerate(["False", "True"]):
        metrics_rows.append({
            "metric": "is_dropped", "class": label,
            "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4), "support": int(s[i]),
        })
    p = save_confusion_matrix(y_true, y_pred, labels_drop, "is_dropped", OUTPUT_DIR)
    if p:
        saved.append(p)

    if "person_norm" in df.columns and "person_gpt_norm" in df.columns:
        y_true = df["person_norm"]
        y_pred = df["person_gpt_norm"]
        labels_p = [c for c in ["1st", "2nd", "3rd", "Impersonal"] if (y_true == c).any() or (y_pred == c).any()]
        if labels_p:
            p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels_p, zero_division=0)
            for i, label in enumerate(labels_p):
                metrics_rows.append({
                    "metric": "person", "class": label,
                    "precision": round(p[i], 4), "recall": round(r[i], 4),
                    "f1": round(f1[i], 4), "support": int(s[i]),
                })
            p = save_confusion_matrix(y_true, y_pred, labels_p, "person", OUTPUT_DIR)
            if p:
                saved.append(p)

    if "number_norm" in df.columns and "number_gpt_norm" in df.columns:
        y_true = df["number_norm"]
        y_pred = df["number_gpt_norm"]
        labels_n = [c for c in ["Singular", "Plural", "None"] if (y_true == c).any() or (y_pred == c).any()]
        if labels_n:
            p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels_n, zero_division=0)
            for i, label in enumerate(labels_n):
                metrics_rows.append({
                    "metric": "number", "class": label,
                    "precision": round(p[i], 4), "recall": round(r[i], 4),
                    "f1": round(f1[i], 4), "support": int(s[i]),
                })
            p = save_confusion_matrix(y_true, y_pred, labels_n, "number", OUTPUT_DIR)
            if p:
                saved.append(p)

    if "perspective_primary_gpt" in df.columns and "perspective_primary" in df.columns:
        pers_df = df.drop_duplicates(subset=["ID"])[["ID", "perspective_primary", "perspective_primary_gpt"]]
        y_true = pers_df["perspective_primary"].apply(lambda x: _norm_cat(x) if pd.notna(x) else "")
        y_pred = pers_df["perspective_primary_gpt"].apply(lambda x: str(x) if pd.notna(x) else "")
        labels_pers = [x for x in sorted(set(y_true.unique()) | set(y_pred.unique())) if str(x).strip()]
        if labels_pers:
            labels_pers = list(labels_pers)
            p = save_confusion_matrix(y_true, y_pred, labels_pers, "perspective", OUTPUT_DIR)
            if p:
                saved.append(p)

    metrics_path = os.path.join(OUTPUT_DIR, "validation_metrics.csv")
    try:
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"metrics -> {metrics_path}")
    except PermissionError:
        alt_path = os.path.join(OUTPUT_DIR, "validation_metrics_new.csv")
        try:
            pd.DataFrame(metrics_rows).to_csv(alt_path, index=False, encoding="utf-8-sig")
            print(f"metrics -> {alt_path}")
        except PermissionError:
            print(f"cannot write {metrics_path}")

    print("confusion matrices:")
    for p in saved:
        print(f"   {p}")


if __name__ == "__main__":
    main()
