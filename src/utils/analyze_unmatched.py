"""Debug unmatched rows between manual merge and GPT detailed."""
import os

import repo_bootstrap

repo_bootstrap.prepare_repo(__file__)

import pandas as pd


def norm(s):
    return " ".join(str(s).strip().split()).strip()


merged = pd.read_csv("data/annotations/manual_annotation_merged.csv")
gpt_raw = pd.read_csv("outputs/01_pronoun_detection/gpt_annotation_detailed.csv")

merged["context_norm"] = merged["context"].fillna("").astype(str).apply(norm)
merged["pronoun_norm"] = merged["pronoun"].fillna("").astype(str).str.strip().str.lower()
merged["ID"] = merged["ID"].astype(str)

gpt_explicit = gpt_raw[gpt_raw["source_mapping"].notna() & (gpt_raw["source_mapping"] != "IMPLIED")].copy()
gpt_explicit["context_norm"] = gpt_explicit["context_sentence_ukr"].fillna("").astype(str).apply(norm)
gpt_explicit["pronoun_norm"] = (
    gpt_explicit["source_mapping"].fillna("").astype(str).str.strip().str.lower()
)
gpt_explicit["ID"] = gpt_explicit["original_id"].astype(str)

m = merged.merge(
    gpt_explicit[["ID", "context_norm", "pronoun_norm"]],
    on=["ID", "context_norm", "pronoun_norm"],
    how="left",
    indicator=True,
)
unmatched = m[m["_merge"] == "left_only"]

print("unmatched analysis")
print("is_dropped=True:", unmatched["is_dropped"].fillna(False).astype(bool).sum())
print("is_dropped=False:", (~unmatched["is_dropped"].fillna(False).astype(bool)).sum())
print("GPT IMPLIED count:", (gpt_raw["source_mapping"] == "IMPLIED").sum())

vc = unmatched["pronoun"].value_counts().head(15)
vc.to_csv("outputs/01_pronoun_detection/unmatched_pronoun_dist.csv", encoding="utf-8-sig")
unmatched.head(20).to_csv(
    "outputs/01_pronoun_detection/unmatched_sample.csv", index=False, encoding="utf-8-sig"
)
print("wrote outputs/01_pronoun_detection/unmatched_sample.csv")
