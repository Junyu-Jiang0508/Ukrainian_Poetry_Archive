# The Grammar of Belonging

**Computational Analysis of Pronominal Deixis in Wartime Ukrainian Poetry (2014--2025)**

Junyu Jiang | Capstone Advisor: Prof. Amelia Glaser  
Contemporary Ukrainian Poetry Archive

---

## Overview

This project applies computational methods from corpus linguistics and NLP to analyze how Ukrainian poets deploy pronominal strategies---the deliberate use of *we*, *you*, *they*---to construct collective identity during wartime. By tracking shifts in pronoun usage across the Contemporary Ukrainian Poetry Archive (2014--2025), the analysis reveals how the boundaries of *Self* and *Other* are continuously renegotiated in response to the ongoing conflict.

## Research Questions

| # | Question | Script |
|---|----------|--------|
| **RQ1** | How has the referential scope of 1PL *ми* (we) shifted from inclusive to collective/defensive? | `src/13_rq1_we_type_analysis.py` |
| **RQ2** | What apostrophic functions does 2nd-person *ти/ви* (you) serve---conversational vs. adversarial? | `src/14_rq2_addressee_analysis.py` |
| **RQ3** | How does pronoun--concept co-occurrence (land, home, mother) change across war phases? | `src/16_temporal_cooccurrence.py` |

## Repository Structure

```
Ukrainian-Poetry/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── src/                       # Analysis pipeline (numbered by execution order)
│   ├── 00_filtering.py              # Corpus language filtering
│   ├── 01_annotation_pronoun_detection.py      # Morphological pronoun extraction
│   ├── 01b_generate_cooccurrence.py # Pronoun--word co-occurrence matrix
│   ├── 02_modeling_pronoun_cooccurrence.py    # Co-occurrence network visualization
│   ├── 02_modeling_pronoun_semantic_space.py  # UMAP embedding & clustering
│   ├── 01_annotation_toolkit.py     # Stratified sampling & IAA helpers
│   ├── 01_annotation_rule_annotate_pronouns.py      # Rule-based pronoun classification
│   ├── 02_modeling_pronoun_projection.py      # Cross-lingual pronoun alignment
│   ├── 01_annotation_gpt_annotation.py        # Batch GPT annotation engine
│   ├── 02_modeling_adaptive_binning.py        # Adaptive temporal binning
│   ├── 02_modeling_breakpoint_regression.py   # WLS regression & PELT changepoints
│   ├── 00_public_list_filter.py     # Public-list corpus subset
│   ├── 01_annotation_gpt_annotate_full.py      # Full-corpus GPT annotation runner
│   ├── 01_annotation_gpt_exploration.py  # Annotation quality exploration
│   ├── 13_rq1_we_type_analysis.py   # RQ1: we-type temporal analysis
│   ├── 14_rq2_addressee_analysis.py # RQ2: addressee-type analysis
│   ├── 15_poem_perspective_analysis.py   # Poem-level perspective shifts
│   ├── 16_temporal_cooccurrence.py  # RQ3: PMI/NPMI by period
│   ├── 17_temporal_network.py       # Pronoun--concept network analysis
│   ├── 18_author_trajectories.py    # Author-level longitudinal clustering
│   ├── 19_publication_figures.py    # Publication-ready figures & LaTeX tables
│   └── 20_descriptive_statistics.py # Corpus descriptive statistics
│
├── src/utils/                 # Utility & validation scripts
│   ├── extract_poems_for_annotation.py
│   ├── merge_and_gpt_annotate.py
│   ├── validate_annotation.py
│   ├── validate_merged_annotation.py
│   ├── analyze_unmatched.py
│   ├── check_results.py
│   └── manual_annotation_app.py     # Streamlit manual annotation tool
│
├── app/                       # Cloud annotation application (Streamlit + Supabase)
│   ├── app.py
│   ├── requirements.txt
│   └── supabase_schema.sql
│
├── docs/                      # Reports & documentation
│   ├── final_report.md
│   ├── findings_report.md
│   ├── annotation_manual.md
│   └── unicode_template.tex
│
├── data/                      # Data directory (gitignored)
│   ├── raw/                         # Original corpus files
│   ├── processed/                   # Pipeline outputs
│   │   ├── my_gpt_run/             # Private GPT annotation run
│   │   └── gpt_annotation_public_run/  # Public-list GPT run
│   └── annotations/                 # Manual annotation results
│
└── outputs/                   # Analysis outputs (gitignored)
    ├── 00_filtering/
    ├── 01_pronoun_detection/
    ├── ...
    └── 20_descriptive_statistics/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download uk_core_news_sm
```

## Running the Pipeline

Scripts are numbered by execution order. Each script can be run independently from the project root:

```bash
python src/00_filtering.py
python src/01_annotation_pronoun_detection.py
# ...
python src/20_descriptive_statistics.py
```

GPT annotation (steps 07/11) requires an OpenAI API key in `.env`:

```
OPENAI_API_KEY=sk-...
```

## Key Dependencies

- **NLP**: spaCy, Stanza, Transformers, SentencePiece, SimAlign
- **Statistics**: scipy, statsmodels, ruptures, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Annotation**: OpenAI API (GPT-4o), Streamlit
