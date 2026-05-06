# The Grammar of Belonging

**Computational Analysis of Pronominal Deixis in Wartime Ukrainian Poetry (2014--2025)**

Junyu Jiang | Capstone Advisor: Prof. Amelia Glaser  
Contemporary Ukrainian Poetry Archive

---

## Overview

This project applies computational methods from corpus linguistics and NLP to analyze how Ukrainian poets deploy pronominal strategies---the deliberate use of *we*, *you*, *they*---to construct collective identity during wartime. By tracking shifts in pronoun usage across the Contemporary Ukrainian Poetry Archive (2014--2025), the analysis reveals how the boundaries of *Self* and *Other* are continuously renegotiated in response to the ongoing conflict.

## Research Questions

| # | Question | Stage |
|---|----------|-------|
| **RQ1** | How has the referential scope of 1PL *ми* (we) shifted across war phases? | `02b` per-cell GLM (`02_modeling_q1_per_cell_glm.py`) |
| **RQ2** | How does author-level heterogeneity modulate the period shift? | `02c` hierarchical random-slope (`02_modeling_q2_hierarchical.py`) |
| **RQ3** | Are typology- and period-based contrasts robust at corpus and cohort level? | `02a/02d/02f` (significance + typology models) |

All RQ scripts read the canonical stanza-level GPT annotation table:
`data/Annotated_GPT_rerun/pronoun_annotation.csv`.

## Pipeline

The pipeline is driven by `src/00_pipeline_orchestrator.py` and the
ordered stage catalog in `src/utils/pipeline_catalog.py`. Every stage has a
short numeric ID and a single canonical script. To inspect the live order:

```bash
PYTHONPATH=src python src/00_pipeline_orchestrator.py --list
```

| ID  | Script | Purpose |
|-----|--------|---------|
| 00a | `00_filtering.py`                            | Layer 0/1 split: posts → poems → stanzas |
| 00b | `00_gpt_human_review_batch.py`               | GPT adjudication for uncertain split rows |
| 00c | `00_public_list_filter.py`                   | Build public-list corpus + derivative CSVs |
| 00d | `00_layer0_layer1_to_run_filter.py`          | In-place trim of `data/To_run/00_filtering/` to public-list rules |
| 01a | `01_annotation_pronoun_detection.py`         | Morphological pronoun detection (spaCy) |
| 01b | `01_annotation_toolkit.py`                   | Sampling / QA helpers |
| 01c | `01_annotation_rule_annotate_pronouns.py`    | Heuristic pilot annotation |
| 01d | `01_annotation_gpt_annotation.py`            | Stanza-level GPT annotation engine (async) |
| 01e | `01_annotation_gpt_annotate_full.py`         | Wrapper that runs 01d with `--source public` |
| 02a | `02_modeling_significance_core_contrasts.py` | Two-period confirmatory contrasts + sensitivity |
| 02b | `02_modeling_q1_per_cell_glm.py`             | **RQ1**: per-cell one-vs-rest GLM (poem & stanza level) |
| 02c | `02_modeling_q2_hierarchical.py`             | **RQ2**: per-cell hierarchical random-slope models |
| 02d | `02_modeling_significance_models.py`         | Model-based inference for pronoun shifts |
| 02e | `02_modeling_significance_publication_figures.py` | Publication figures for inferential outputs |
| 02f | `02_modeling_typology_and_period_models.py`  | Typology + period cohort models |
| 03a | `03_reporting_descriptive_statistics.py`     | Methodology + corpus overview tables |
| 03b | `03_reporting_roster_freeze.py`              | Author roster freeze + diagnostics |

## Repository Structure

```
Ukrainian-Poetry/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── 00_pipeline_orchestrator.py    # Unified runner (--list / --from-stage / --to-stage / --only)
│   ├── 00_*.py                        # Filtering & corpus build (00a–00d)
│   ├── 01_*.py                        # Annotation (01a–01e)
│   ├── 02_*.py                        # Modeling (02a–02f)
│   ├── 03_*.py                        # Reporting (03a, 03b)
│   │
│   └── utils/                         # Shared library code (importable as `utils.<module>`)
│       ├── pipeline_catalog.py        # Source of truth for stage order
│       ├── workspace.py               # Repo-root + matplotlib environment helpers
│       ├── stage_io.py                # Stage-aware CSV reading/writing
│       ├── adaptive_temporal_binning.py  # `adaptive_binning` + `balanced_temporal_binning`
│       ├── annotation_cohort.py
│       ├── annotation_derived_columns.py
│       ├── csv_io.py
│       ├── label_normalization.py
│       ├── language_strata.py
│       ├── pronoun_encoding.py
│       ├── public_list_filters.py
│       ├── repo_bootstrap.py          # `prepare_repo` for legacy stand-alone scripts
│       ├── reporting_common.py
│       ├── stats_common.py
│       └── _archive/                  # IAA / manual-annotation scripts (see archive README)
│
├── app/                               # Cloud annotation app (Streamlit + Supabase)
├── docs/                              # Workflow doc + reports
├── data/                              # Gitignored — raw, processed, GPT runs, annotations
└── outputs/                           # Gitignored — per-stage analysis artifacts
```

## Data Flow (Live Path)

```
data/raw/ukrpoetry_database.csv ─┬─► 00_filtering.py     ──► data/To_run/00_filtering/layer0,layer1
                                 │
                                 └─► 00_public_list_filter.py ──► data/processed/ukrpoetry_database_public_list.csv

data/To_run/00_filtering/layer1_stanzas_one_per_row.csv
        │
        ▼
01_annotation_gpt_annotation.py (async, stanza-level)
        │
        ▼
data/Annotated_GPT/pronoun_annotation.csv
        │
        ▼ (after manual rerun curation)
data/Annotated_GPT_rerun/pronoun_annotation.csv  ◄── canonical input for all 02/03 stages
        │
        ├─► 02_modeling_significance_core_contrasts.py
        ├─► 02_modeling_q1_per_cell_glm.py            (RQ1)
        ├─► 02_modeling_q2_hierarchical.py            (RQ2)
        ├─► 02_modeling_significance_models.py
        ├─► 02_modeling_significance_publication_figures.py
        ├─► 02_modeling_typology_and_period_models.py (RQ3 / typology)
        └─► 03_reporting_*.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download uk_core_news_sm
```

GPT annotation requires an OpenAI API key in `.env`:

```
OPENAI_API_KEY=sk-...
```

## Running the Pipeline

```bash
# List every stage in execution order
PYTHONPATH=src python src/00_pipeline_orchestrator.py --list

# Run a contiguous range
PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 02a --to-stage 03b

# Run specific stages only
PYTHONPATH=src python src/00_pipeline_orchestrator.py --only 02b 02c 03a

# Dry-run (print commands only)
PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 00a --to-stage 03b --dry-run
```

Each stage script can also be invoked directly, e.g.
`PYTHONPATH=src python src/02_modeling_q1_per_cell_glm.py`.

## Key Dependencies

- **NLP**: spaCy, Stanza, Transformers
- **Statistics**: scipy, statsmodels, ruptures, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Annotation**: OpenAI API (GPT-4o-mini), Streamlit (legacy IAA app in `src/utils/_archive/`)
