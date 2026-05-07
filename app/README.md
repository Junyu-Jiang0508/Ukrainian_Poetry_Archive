# Poem Perspective Annotation App - Cloud Version

Multi-user Streamlit app for **whole-poem perspective** annotation, deployable on
Streamlit Community Cloud with Supabase backend.

This version only collects poem-level perspective judgments (primary +
optional secondary). The earlier sentence-level pronoun pro-drop UI has been
removed from the page; the corresponding helpers are kept commented out at
the bottom of `app.py` for future restoration.

## What you annotate

For each poem (Ukrainian original + Shakespeare-style English translation if
available):

- **Primary perspective** (required): `1st person`, `2nd person`,
  `3rd person`, `Mixed`, `Other`
- **Secondary perspective** (optional): same options or `None`

## Setup

### 1. Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Run `supabase_schema.sql` in the SQL Editor to create tables
   - Only `poem_perspectives` is read/written by this version.
   - The `annotations` table can be left in place (legacy).

### 2. Streamlit Secrets

For Streamlit Community Cloud, add secrets in the app settings:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
key = "your-anon-or-service-role-key"
```

For local development, create `.streamlit/secrets.toml` with the same content.

### 3. Data File

`poems_for_manual_annotation.csv` ships in this folder with **50 randomly
sampled Ukrainian original poems** (seed = 42) drawn from
`data/analysis_subsets/pronoun_annotation_roster_included.csv`.

Columns:

- `ID` — poem id (e.g. `UP1184_1`)
- `author`
- `year`, `date`, `temporal_period`
- `language` (always `Ukrainian` for this sample)
- `n_stanzas`
- `text` — full Ukrainian poem (stanzas joined by blank lines)
- `text_en` — full English translation (Shakespeare-style)

To regenerate with a different seed or sample size, see the inline script
used to build it (`README` section below) or rerun the sampling step.

### 4. Run Locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

### 5. Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io)
3. Set main file to `app.py`
4. Add secrets (Supabase url and key)
5. Deploy

## Multi-User Isolation

Each annotator enters their name in the sidebar. All poem perspectives are
stored per annotator in Supabase, so multiple annotators can work
simultaneously without overwriting each other's data.

## Regenerating the 50-poem sample

Run from repo root:

```bash
python3 - <<'PY'
import pandas as pd, numpy as np
df = pd.read_csv("data/analysis_subsets/pronoun_annotation_roster_included.csv")
ukr = df[(df["language"] == "Ukrainian") & (df["is_translation"] == False)]
ukr = ukr.drop_duplicates(subset=["poem_id", "stanza_index"])

def assemble(g):
    g = g.sort_values("stanza_index")
    return "\n\n".join(str(s).strip() for s in g["stanza_ukr"].dropna())

poems = (
    ukr.groupby("poem_id", sort=False)
    .apply(lambda g: pd.Series({
        "author": g["author"].iloc[0],
        "year": g["year_int"].iloc[0] if pd.notna(g["year_int"].iloc[0]) else g["year"].iloc[0],
        "language": g["language"].iloc[0],
        "temporal_period": g["temporal_period"].iloc[0],
        "n_stanzas": g["stanza_index"].nunique(),
        "text": assemble(g),
        "text_en": g["full_shakespeare_text"].dropna().iloc[0]
                   if g["full_shakespeare_text"].notna().any() else "",
    }), include_groups=False)
    .reset_index()
    .rename(columns={"poem_id": "ID"})
)
rng = np.random.default_rng(42)
idx = rng.choice(len(poems), size=50, replace=False); idx.sort()
poems.iloc[idx].to_csv("app/poems_for_manual_annotation.csv",
                      index=False, encoding="utf-8-sig")
PY
```
