"""Validate manual vs GPT on data/annotations/manual_annotation_merged_with_gpt.csv."""
import os

import repo_bootstrap

repo_bootstrap.prepare_repo(__file__)

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from utils.label_normalization import normalize_person_number_label

MERGED_FILE = os.path.join("data", "annotations", "manual_annotation_merged_with_gpt.csv")
METRICS_OUTPUT = os.path.join("outputs", "01_annotation_pronoun_detection", "validation_merged_metrics.csv")
ERROR_OUTPUT = os.path.join("outputs", "01_annotation_pronoun_detection", "validation_merged_errors.csv")


def main():
    if not os.path.exists(MERGED_FILE):
        print(f"missing {MERGED_FILE}")
        return

    df = pd.read_csv(MERGED_FILE)
    has_gpt = df["person_gpt"].notna() | df["number_gpt"].notna() | df["is_dropped_gpt"].notna()
    df = df[has_gpt].copy()
    if df.empty:
        print("no gpt rows")
        return

    for col in ["person", "number"]:
        if col in df.columns:
            df[f"{col}_norm"] = df[col].apply(normalize_person_number_label)
        if f"{col}_gpt" in df.columns:
            df[f"{col}_gpt_norm"] = df[f"{col}_gpt"].apply(normalize_person_number_label)

    manual_drop = df["is_dropped"].fillna(False).astype(bool)
    gpt_drop = df["is_dropped_gpt"].fillna(False).astype(bool)

    metrics_rows = []
    total = len(df)

    y_true = manual_drop.astype(int)
    y_pred = gpt_drop.astype(int)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    for i, label in enumerate(["False", "True"]):
        metrics_rows.append({
            "metric": "is_dropped", "class": label,
            "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4), "support": int(s[i]),
        })

    if "person_norm" in df.columns and "person_gpt_norm" in df.columns:
        y_true = df["person_norm"]
        y_pred = df["person_gpt_norm"]
        labels = [c for c in ["1st", "2nd", "3rd", "Impersonal"] if (y_true == c).any() or (y_pred == c).any()]
        if labels:
            p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
            for i, label in enumerate(labels):
                metrics_rows.append({
                    "metric": "person", "class": label,
                    "precision": round(p[i], 4), "recall": round(r[i], 4),
                    "f1": round(f1[i], 4), "support": int(s[i]),
                })

    if "number_norm" in df.columns and "number_gpt_norm" in df.columns:
        y_true = df["number_norm"]
        y_pred = df["number_gpt_norm"]
        labels = [c for c in ["Singular", "Plural", "None"] if (y_true == c).any() or (y_pred == c).any()]
        if labels:
            p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
            for i, label in enumerate(labels):
                metrics_rows.append({
                    "metric": "number", "class": label,
                    "precision": round(p[i], 4), "recall": round(r[i], 4),
                    "f1": round(f1[i], 4), "support": int(s[i]),
                })

    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"n={total}")
    print(f"metrics -> {METRICS_OUTPUT}")

    all_agree = (manual_drop == gpt_drop)
    if "person_norm" in df.columns and "person_gpt_norm" in df.columns:
        all_agree = all_agree & (df["person_norm"] == df["person_gpt_norm"])
    if "number_norm" in df.columns and "number_gpt_norm" in df.columns:
        all_agree = all_agree & (df["number_norm"] == df["number_gpt_norm"])

    errors = df[~all_agree]
    if len(errors) > 0:
        out_cols = ["ID", "annotator", "pronoun", "context", "person", "person_gpt", "number", "number_gpt", "is_dropped", "is_dropped_gpt"]
        out_cols = [c for c in out_cols if c in errors.columns]
        errors[out_cols].to_csv(ERROR_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"errors -> {ERROR_OUTPUT} n={len(errors)}")
    else:
        print("no disagreements.")


if __name__ == "__main__":
    main()
