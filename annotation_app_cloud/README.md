# Pronoun Annotation App - Cloud Version

Multi-user Streamlit app for pronoun pro-drop annotation, deployable on Streamlit Community Cloud with Supabase backend.

## Setup

### 1. Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Run `supabase_schema.sql` in the SQL Editor to create tables

### 2. Streamlit Secrets

For Streamlit Community Cloud, add secrets in the app settings:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
key = "your-anon-or-service-role-key"
```

For local development, create `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
key = "your-anon-or-service-role-key"
```

### 3. Data File

Add `poems_for_manual_annotation.csv` to this folder. You can copy from:

```
../outputs/01_pronouns_detection/poems_for_manual_annotation.csv
```

### 4. Run Locally

```bash
cd annotation_app_cloud
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

Each annotator enters their name in the sidebar. All annotations and poem perspectives are stored per annotator in Supabase, so multiple annotators can work simultaneously without overwriting each other's data.
