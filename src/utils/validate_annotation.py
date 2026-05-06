"""Compare manual_annotation_result.csv to GPT (detailed or test); write metrics and error CSV."""
import os

import repo_bootstrap

repo_bootstrap.prepare_repo(__file__)

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from utils.label_normalization import normalize_person_number_label

MANUAL_FILE = "outputs/01_pronoun_detection/manual_annotation_result.csv"
GPT_DETAILED_FILE = "outputs/01_pronoun_detection/gpt_annotation_detailed.csv"
GPT_TEST_FILE = "outputs/01_pronoun_detection/gpt_annotation_test_result.csv"
ERROR_OUTPUT = "outputs/01_pronoun_detection/validation_error_cases.csv"
METRICS_OUTPUT = "outputs/01_pronoun_detection/validation_metrics.csv"


def _normalize_sent(s: str) -> str:
    return " ".join(str(s).strip().split())


def _prepare_gpt_detailed(gpt_df: pd.DataFrame) -> pd.DataFrame:
    gpt_df = gpt_df.copy()
    gpt_df["context_sentence_ukr"] = gpt_df["context_sentence_ukr"].fillna("").astype(str)
    gpt_df = gpt_df[gpt_df["source_mapping"].notna() & (gpt_df["source_mapping"] != "IMPLIED")]
    gpt_df["context_norm"] = gpt_df["context_sentence_ukr"].apply(_normalize_sent)
    gpt_df = gpt_df.drop_duplicates(subset=["original_id", "context_sentence_ukr", "source_mapping"])
    gpt_df = gpt_df.rename(columns={
        "original_id": "ID",
        "source_mapping": "pronoun",
        "is_pro_drop": "is_dropped",
    })
    gpt_df["context"] = gpt_df["context_sentence_ukr"]
    if "person" in gpt_df.columns:
        gpt_df["person"] = gpt_df["person"].apply(normalize_person_number_label)
    if "number" in gpt_df.columns:
        gpt_df["number"] = gpt_df["number"].apply(normalize_person_number_label)
    cols = ["ID", "pronoun", "is_dropped", "context", "context_norm"]
    for c in ["person", "number"]:
        if c in gpt_df.columns:
            cols.append(c)
    return gpt_df[cols]


def _load_gpt_for_validation() -> tuple[pd.DataFrame, str]:
    manual = pd.read_csv(MANUAL_FILE)
    manual_ids = set(manual["ID"].astype(str).unique())

    if os.path.exists(GPT_DETAILED_FILE):
        gpt_det = pd.read_csv(GPT_DETAILED_FILE)
        gpt_prep = _prepare_gpt_detailed(gpt_det)
        gpt_ids = set(gpt_prep["ID"].astype(str).unique())
        if manual_ids & gpt_ids:
            return gpt_prep, "gpt_annotation_detailed.csv"

    if os.path.exists(GPT_TEST_FILE):
        gpt_test = pd.read_csv(GPT_TEST_FILE)
        gpt_test = gpt_test.rename(columns={"is_dropped": "is_dropped"})
        return gpt_test, "gpt_annotation_test_result.csv"

    return pd.DataFrame(), ""


def main():
    if not os.path.exists(MANUAL_FILE):
        print(f"missing manual: {MANUAL_FILE}")
        return

    manual = pd.read_csv(MANUAL_FILE)
    gpt, gpt_src = _load_gpt_for_validation()
    if gpt.empty:
        print(f"missing gpt: {GPT_DETAILED_FILE} or {GPT_TEST_FILE}")
        return

    print(f"gpt source: {gpt_src}")

    use_context_key = "context_norm" in gpt.columns
    if use_context_key:
        manual["context_norm"] = manual["context"].fillna("").astype(str).apply(_normalize_sent)
        key = ["ID", "context_norm", "pronoun"]
    else:
        for col in ["ID", "pronoun", "position"]:
            if col in manual.columns:
                manual[col] = manual[col].astype(str)
            if col in gpt.columns:
                gpt[col] = gpt[col].astype(str)
        key = ["ID", "pronoun", "position"]

    gpt_cols = [c for c in key + ["is_dropped", "context", "person", "number"] if c in gpt.columns]
    gpt_sub = gpt[gpt_cols].drop_duplicates(subset=key)
    merged = manual.merge(
        gpt_sub,
        on=key,
        how="left",
        suffixes=("_manual", "_gpt"),
    )
    merged = merged.dropna(subset=["is_dropped_gpt"])

    for col in ["person", "number"]:
        mcol, gcol = f"{col}_manual", f"{col}_gpt"
        if mcol in merged.columns:
            merged[mcol] = merged[mcol].fillna("").astype(str).apply(_norm_cat)
        if gcol in merged.columns:
            merged[gcol] = merged[gcol].fillna("").astype(str).apply(_norm_cat)

    manual_drop = merged["is_dropped_manual"].map(lambda x: bool(x) if pd.notna(x) else False)
    gpt_drop = merged["is_dropped_gpt"].map(lambda x: bool(x) if pd.notna(x) else False)

    total = len(merged)

    all_agree = (manual_drop == gpt_drop)
    if "person_manual" in merged.columns and "person_gpt" in merged.columns:
        all_agree = all_agree & (merged["person_manual"] == merged["person_gpt"])
    if "number_manual" in merged.columns and "number_gpt" in merged.columns:
        all_agree = all_agree & (merged["number_manual"] == merged["number_gpt"])

    metrics_rows = []
    y_true_drop = manual_drop.astype(int)
    y_pred_drop = gpt_drop.astype(int)
    p, r, f1, s = precision_recall_fscore_support(y_true_drop, y_pred_drop, labels=[0, 1], zero_division=0)
    for i, label in enumerate(["False", "True"]):
        metrics_rows.append({
            "metric": "is_dropped", "class": label,
            "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4), "support": int(s[i]),
        })
    if "person_manual" in merged.columns and "person_gpt" in merged.columns:
        y_true_p = merged["person_manual"]
        y_pred_p = merged["person_gpt"]
        labels_p = [c for c in ["1st", "2nd", "3rd", "Impersonal"] if (y_true_p == c).any() or (y_pred_p == c).any()]
        if labels_p:
            p, r, f1, s = precision_recall_fscore_support(y_true_p, y_pred_p, labels=labels_p, zero_division=0)
            for i, label in enumerate(labels_p):
                metrics_rows.append({
                    "metric": "person", "class": label,
                    "precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f1[i], 4), "support": int(s[i]),
                })
    if "number_manual" in merged.columns and "number_gpt" in merged.columns:
        y_true_n = merged["number_manual"]
        y_pred_n = merged["number_gpt"]
        labels_n = [c for c in ["Singular", "Plural", "None"] if (y_true_n == c).any() or (y_pred_n == c).any()]
        if labels_n:
            p, r, f1, s = precision_recall_fscore_support(y_true_n, y_pred_n, labels=labels_n, zero_division=0)
            for i, label in enumerate(labels_n):
                metrics_rows.append({
                    "metric": "number", "class": label,
                    "precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f1[i], 4), "support": int(s[i]),
                })

    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print("=" * 50)
    print("validation")
    print("=" * 50)
    print(f"n={total}")
    print()
    print("metrics ->", METRICS_OUTPUT)

    disagree = ~all_agree
    errors = merged[disagree].copy()
    errors["manual_is_dropped"] = errors["is_dropped_manual"].map(lambda x: bool(x) if pd.notna(x) else False)
    errors["gpt_is_dropped"] = errors["is_dropped_gpt"].map(lambda x: bool(x) if pd.notna(x) else False)
    errors["disagree_drop"] = errors["manual_is_dropped"] != errors["gpt_is_dropped"]
    errors["disagree_person"] = False
    errors["disagree_number"] = False
    if "person_manual" in errors.columns and "person_gpt" in errors.columns:
        errors["disagree_person"] = errors["person_manual"] != errors["person_gpt"]
    if "number_manual" in errors.columns and "number_gpt" in errors.columns:
        errors["disagree_number"] = errors["number_manual"] != errors["number_gpt"]

    if len(errors) > 0:
        os.makedirs(os.path.dirname(ERROR_OUTPUT), exist_ok=True)
        ctx_col = "context_manual" if "context_manual" in errors.columns else "context"
        if ctx_col not in errors.columns and "context" in errors.columns:
            ctx_col = "context"
        out_cols = [
            "ID", "author", "pronoun", "position", ctx_col,
            "manual_is_dropped", "gpt_is_dropped", "disagree_drop",
            "person_manual", "person_gpt", "disagree_person",
            "number_manual", "number_gpt", "disagree_number",
        ]
        out_cols = [c for c in out_cols if c in errors.columns]
        errors[out_cols].to_csv(ERROR_OUTPUT, index=False, encoding="utf-8-sig")
        print()
        print(f"errors -> {ERROR_OUTPUT} n={len(errors)}")
        print()
        print("first 5 errors:")
        for _, r in errors.head(5).iterrows():
            pn_safe = str(r.get("pronoun", "")).encode("ascii", "replace").decode()
            parts = [f"ID={r['ID']} pronoun={pn_safe}"]
            parts.append(f"drop: M={r['manual_is_dropped']} G={r['gpt_is_dropped']}")
            if "person_manual" in r and "person_gpt" in r:
                parts.append(f"person: M={r['person_manual']} G={r['person_gpt']}")
            if "number_manual" in r and "number_gpt" in r:
                parts.append(f"number: M={r['number_manual']} G={r['number_gpt']}")
            print("  " + " | ".join(parts))
    else:
        print()
        print("no disagreements.")


if __name__ == "__main__":
    main()
