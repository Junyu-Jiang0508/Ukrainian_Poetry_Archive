# Archived utilities

These scripts are no longer part of the live pipeline (`src/utils/pipeline_catalog.py`)
but are retained for IAA / methodology reproducibility.

| Script | Purpose | When it was active |
|---|---|---|
| `merge_and_gpt_annotate.py` | Merge two human annotators' CSVs into the GPT-annotated table for IAA. | Inter-annotator agreement phase. |
| `validate_annotation.py` | Compare a single human-annotated batch against GPT output. | IAA phase. |
| `validate_merged_annotation.py` | Compare merged human consensus against GPT output. | IAA phase. |
| `manual_annotation_app.py` | Streamlit app used during manual labeling sessions. | Pre-GPT annotation pilots. |

These scripts use `repo_bootstrap.prepare_repo` which now walks up looking for
`requirements.txt` / `.git`, so they continue to work from this nested location.
