"""Sample poems by year (<2022 / >=2022), split into sentences for manual CSV."""
import os
import re

import repo_bootstrap

repo_bootstrap.prepare_repo(__file__)

import pandas as pd

INPUT_FILE = "outputs/01_annotation_pronoun_detection/ukrainian_pronouns_projection_final.csv"
OUTPUT_FILE = "outputs/01_annotation_pronoun_detection/poems_for_manual_annotation.csv"

YEAR_BOUNDARY = 2022
POEMS_PER_GROUP = 10


def remove_blank_lines(text: str) -> str:
    if pd.isna(text):
        return ""
    lines = [line for line in str(text).split("\n") if line.strip()]
    return "\n".join(lines)


def is_skip_sentence(s: str) -> bool:
    s = str(s).strip()
    if not s:
        return True
    if not any(c.isalnum() for c in s):
        return True
    if all(c.isdigit() or c in " .," for c in s) and any(c.isdigit() for c in s):
        return True
    return False


def split_into_sentences(text: str) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    text = remove_blank_lines(text).strip()
    raw_sentences = re.split(r"[\n.!?;]+", text)
    return [s.strip() for s in raw_sentences if s.strip() and not is_skip_sentence(s.strip())]


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"missing: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    cols_to_keep = ["ID", "author", "date", "Language", "text", "Theme"]
    df_unique = df[cols_to_keep].drop_duplicates(subset=["ID"])

    df_unique["year"] = pd.to_datetime(df_unique["date"], errors="coerce").dt.year
    df_unique = df_unique.dropna(subset=["year"])

    before = df_unique[df_unique["year"] < YEAR_BOUNDARY]
    after = df_unique[df_unique["year"] >= YEAR_BOUNDARY]

    n_before = min(POEMS_PER_GROUP, len(before))
    n_after = min(POEMS_PER_GROUP, len(after))
    selected_before = before.sample(n=n_before, random_state=42) if n_before > 0 else pd.DataFrame()
    selected_after = after.sample(n=n_after, random_state=42) if n_after > 0 else pd.DataFrame()

    df_selected = pd.concat([selected_before, selected_after], ignore_index=True)

    rows = []
    for _, row in df_selected.iterrows():
        poem_id = row["ID"]
        author = row["author"]
        date = row["date"]
        lang = row["Language"]
        full_text = row["text"]
        theme = row["Theme"]
        full_text_clean = remove_blank_lines(full_text) if pd.notna(full_text) else ""

        sentences = split_into_sentences(full_text)

        for sent_id, sentence in enumerate(sentences, start=1):
            rows.append(
                {
                    "ID": poem_id,
                    "author": author,
                    "date": date,
                    "Language": lang,
                    "sentence_id": sent_id,
                    "sentence": sentence.strip(),
                    "context": full_text_clean,
                    "Theme": theme,
                    "manual_annotation": "",
                }
            )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["date", "ID", "sentence_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"poems={len(df_selected)} sentences={len(result_df)} -> {OUTPUT_FILE}")
    before_ids = set(selected_before["ID"]) if len(selected_before) > 0 else set()
    after_ids = set(selected_after["ID"]) if len(selected_after) > 0 else set()
    before_sents = result_df[result_df["ID"].isin(before_ids)]
    after_sents = result_df[result_df["ID"].isin(after_ids)]
    print(f"before {YEAR_BOUNDARY}: sents={len(before_sents)} poems={n_before}")
    print(f"from {YEAR_BOUNDARY}: sents={len(after_sents)} poems={n_after}")


if __name__ == "__main__":
    main()
