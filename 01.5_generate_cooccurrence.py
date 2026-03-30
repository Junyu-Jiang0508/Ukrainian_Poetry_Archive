import pandas as pd
import spacy
import os
from collections import defaultdict

INPUT_PATH = "outputs/01_pronouns_detection/ukrainian_pronouns_detailed.csv"
OUTPUT_DIR = "outputs/02_pronoun_cooccurrence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} pronoun instances")

try:
    nlp = spacy.load("uk_core_news_sm")
except:
    print("Please install: python -m spacy download uk_core_news_sm")
    raise

def extract_cooccurrence_from_text(text, pronoun_list, window=5):
    if pd.isna(text):
        return []
    doc = nlp(str(text))
    results = []
    for i, token in enumerate(doc):
        if token.pos_ == "PRON" and token.text.lower() in pronoun_list:
            for j in range(max(0, i-window), min(len(doc), i+window+1)):
                neighbor = doc[j]
                if j != i and neighbor.pos_ in ["VERB", "NOUN", "ADJ"]:
                    results.append({
                        "pronoun": token.text.lower(),
                        "word": neighbor.lemma_.lower()
                    })
    return results

pronoun_set = set(df["pronoun"].dropna().unique())
print(f"Found {len(pronoun_set)} unique pronouns")

print("Extracting co-occurrences...")
grouped = df.groupby(["ID", "date", "text"])

cooccurrence_data = []
for (poem_id, date, text), group in grouped:
    coocs = extract_cooccurrence_from_text(text, pronoun_set, window=5)
    for cooc in coocs:
        cooccurrence_data.append({
            "ID": poem_id,
            "date": date,
            "pronoun": cooc["pronoun"],
            "word": cooc["word"]
        })

df_cooc = pd.DataFrame(cooccurrence_data)
print(f"Extracted {len(df_cooc)} co-occurrence instances")

cooc_counts = df_cooc.groupby(["pronoun", "word", "date"]).size().reset_index(name="count")
cooc_counts = cooc_counts.sort_values("count", ascending=False)

output_path = os.path.join(OUTPUT_DIR, "pronoun_cooccurrence_with_date.csv")
cooc_counts.to_csv(output_path, index=False, encoding="utf-8")
print(f"Saved to {output_path}")

cooc_simple = df_cooc.groupby(["pronoun", "word"]).size().reset_index(name="count")
cooc_simple = cooc_simple.sort_values("count", ascending=False)
simple_path = os.path.join(OUTPUT_DIR, "pronoun_cooccurrence.csv")
cooc_simple.to_csv(simple_path, index=False, encoding="utf-8")
print(f"Saved simple version to {simple_path}")

print("Co-occurrence extraction complete.")

