import pandas as pd
import os

df = pd.read_csv("data/raw/ukrpoetry_database.csv")
df.columns = df.columns.str.strip()

df_uk = df[df["Language"].str.strip().str.lower() == "ukrainian"].copy()

df_uk = df_uk[["ID", "Author of poem", "Date posted", "Language", "Poem full text (copy and paste)"]]
df_uk.rename(columns={
    "Poem full text (copy and paste)": "text",
    "Author of poem": "author",
    "Date posted": "date"
}, inplace=True)

os.makedirs("outputs/00_pronoun_detection", exist_ok=True)
output_path = "outputs/00_pronoun_detection/ukrainian_filtered.csv"
df_uk.to_csv(output_path, index=False, encoding="utf-8")
print(f"Saved {len(df_uk)} Ukrainian records to {output_path}")
