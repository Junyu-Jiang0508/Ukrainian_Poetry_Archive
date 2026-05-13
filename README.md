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
| **RQ1** | How has the referential scope of 1PL *ми* (we) shifted across war phases? | `02b` per-cell GLM with **co-primary inference**: Poisson+cluster(+wild bootstrap p) and NB+cluster; `02b2`/`02b3`/`02b4` (token / FV / FV-excl-imperative sensitivities); `00e` precomputes FV counts; `02bcmp` + `02bsc` summarize robustness |
| **RQ1b** | Which authors drive any observed shift, and is it robust? | `02bq1b` author×period FE (HC3 + strict per-cell filter) and per-author δ bootstrap forests |
| **RQ1c** | Did the same peace-time poets change, or did war recruit new entrants? | `02bq1c` GLM restricted to authors with first observed year ≤ 2014 (exploratory; not in main BH family) |
| **RQ2** | How does author-level heterogeneity modulate the period shift? | `02c` hierarchical random-slope (`02_modeling_q2_hierarchical.py`) |
| **RQ3** | Are typology- and period-based contrasts robust at corpus and cohort level? | `02a/02d/02f` (significance + typology models) |

**Estimand decoupling between `02b` (Absolute Salience) and `02a` (Attention Allocation).**
The two confirmatory stages target *different* estimands and must not be conflated in
the manuscript narrative:

- **`02b` — Absolute Salience.** Per-cell Poisson / NB GLM with offset
  `log(exposure)` (stanzas, tokens, or finite verbs). Estimand: *"Do poets write
  absolutely more 1pl (or 2sg, 2pl, 1sg) tokens, controlling for exposure?"*
  Cells are inferentially independent and the denominator is exposure, not other
  pronouns.
- **`02a` — Attention Allocation.** Closed-denominator binomial logit on the
  four-cell first/second-person quartet
  `{1sg, 1pl, 2sg, 2pl_vy_true_plural}` with `n_12 = sum(four-cell)` as the
  trial total. Estimand: *"Within the closed first/second-person sub-space, how
  is attention reallocated between self (`1sg`) and group (`1pl`), and between
  intimate (`2sg`) and collective (`2pl`)?"* The closed denominator is a
  feature, not a bug: it is the natural sample space for a pragmatic
  attention-allocation question.

Manuscript ordering: report `02b` absolute strength first; then turn to `02a`
to ask how, *given* changes in absolute strength, the first/second-person
weights have been internally redistributed. `02a` is fit as a *co-primary*
combination of (i) a binomial GLMM with random author intercept
(`lme4::glmer`, via `src/utils/r_glmm_runner.py`) and (ii) a Cox conditional
logistic regression (`src/utils/conditional_logit_fit.py`), both of which
avoid the incidental-parameter bias that the unconditional `+ C(author)` MLE
incurs at our sample size (N ≈ 33 authors).

**Cell-set split between frequentist and Bayesian paths.** The 5-cell split
(`{1sg, 1pl, 2sg, 2pl_vy_polite_singular, 2pl_vy_true_plural}`) is collapsed to a
4-cell `PRIMARY_GLM_CELLS_FREQUENTIST` set `{1sg, 1pl, 2sg, 2pl_vy_true_plural}`
for all frequentist stages (Q1, Q1b, Q1c, robustness). Polite-singular ви has 23
events / 18 poems / 13 authors with only 3 authors contributing events in both
periods — too sparse for MLE-based poem-level inference (separation kills the
fit). Q2 hierarchical retains the full 5-cell `PRIMARY_GLM_CELLS_BAYESIAN` set
because negative-binomial random-slope shrinkage produces meaningful (if wide)
HDIs even at this sparsity. The polite-singular column always remains in
`q1_poem_unit_cell_counts_12.csv` for future stanza-level work.

**Estimand note (`year` vs `Date posted`).** Primary period coding is composition-year
based (`year` → `period_three_way`). The `invasion_20220224` robustness spec uses
posting date (`Date posted`) and should be interpreted as an **alternate estimand**
rather than a same-estimand robustness check.

**Offset-selection note.** Stanza offset is retained for continuity, but because many
poems have `exposure_n_stanzas == 1`, token/FV offsets are co-reported and folded into
the specification-curve outputs (`02bsc`) to avoid denominator cherry-picking.

All RQ scripts read the canonical stanza-level GPT annotation table:
`data/Annotated_GPT_rerun/pronoun_annotation.csv`.

## Pipeline

The pipeline is driven by `src/00_pipeline_orchestrator.py` and the
ordered stage catalog in `src/utils/pipeline_catalog.py`. Every stage has a
short numeric ID and a single canonical script, plus optional explicit
`depends_on` dependencies resolved by the orchestrator before execution. To
inspect the live order:

```bash
PYTHONPATH=src python src/00_pipeline_orchestrator.py --list
```

| ID  | Script | Purpose |
|-----|--------|---------|
| 00a | `00_filtering.py`                            | Layer 0/1 split: posts → poems → stanzas |
| 00b | `00_gpt_human_review_batch.py`               | GPT adjudication for uncertain split rows |
| 00c | `00_public_list_filter.py`                   | Build public-list corpus + derivative CSVs |
| 00d | `00_layer0_layer1_to_run_filter.py`          | In-place trim of `data/To_run/00_filtering/` to public-list rules |
| 00e | `00e_compute_finite_verb_exposure.py`       | Precompute Stanza finite-verb counts per stanza → `stanza_finite_verb_counts.csv` |
| 01a | `01_annotation_pronoun_detection.py`         | Morphological pronoun detection (spaCy) |
| 01b | `01_annotation_toolkit.py`                   | Sampling / QA helpers |
| 01c | `01_annotation_rule_annotate_pronouns.py`    | Heuristic pilot annotation |
| 01d | `01_annotation_gpt_annotation.py`            | Stanza-level GPT annotation engine (async) |
| 01e | `01_annotation_gpt_annotate_full.py`         | Wrapper that runs 01d with `--source public` |
| 01f | `01_annotation_vy_register_audit.py`         | Manual QA package for `vy_register` (full polite-singular + stratified true-plural sample) |
| 02a | `02_modeling_significance_core_contrasts.py` | **Attention Allocation**: closed-denominator binomial on the 1st/2nd-person 4-cell quartet; co-primary lme4 GLMM + Cox conditional logit (legacy `+C(author)` MLE retained as sensitivity only) |
| 02b | `02_modeling_q1_per_cell_glm.py`             | **Absolute Salience (RQ1)**: per-cell Poisson / NB GLM with `log(exposure)` offset; co-primary inference file includes Poisson+wild-bootstrap and NB |
| 02b2 | `02_modeling_q1_per_cell_glm.py --exposure-type=n_tokens` | **RQ1 sensitivity**: same script, token offset |
| 02b3 | `02_modeling_q1_per_cell_glm.py --exposure-type=n_finite_verbs` | **RQ1 sensitivity**: finite-verb offset (requires `00e`) |
| 02b4 | `02_modeling_q1_per_cell_glm.py --exposure-type=n_finite_verbs_excl_imperative` | **RQ1 sensitivity**: FV offset excluding imperatives |
| 02bvl | `02_modeling_finite_verb_validation_sample.py` | Stratified Stanza validation tokens + morph vs depparse agreement |
| 02bq1c | `02_modeling_q1c_pre_invasion_cohort.py`   | **RQ1c**: exploratory pre-invasion cohort GLM (not in main BH family) |
| 02bq1b | `02_modeling_q1b_within_author_fe.py`      | **RQ1b**: parametric author×period FE (HC3) + per-author δ bootstrap |
| 02bq3 | `02_modeling_q3_sparse_2pl_aggregated.py`   | Supplementary author×period legacy-2pl aggregation |
| 02brat | `02_modeling_ratio_indices.py`              | Build shared poem-level ratio index table (FV-gated), exclusions, and sensitivity denominator variant |
| 02bratpop | `02_modeling_ratio_q1_binomial.py`      | Ratio-population binomial GLM with clustered SE + co-primary bootstrap p outputs |
| 02bratq1b | `02_modeling_ratio_q1b_within_author_fe.py` | Ratio within-author FE binomial + sparsity audits + author-level bootstrap deltas |
| 02bratq2 | `02_modeling_ratio_q2_hierarchical.py`   | Ratio hierarchical logistic (Bambi), random-slope fallback, author caterpillars |
| 02brobp | `02_modeling_robustness_period_definitions.py` | Q1 replicated under alternate period encodings |
| 02broba | `02_modeling_robustness_author_filter.py` | Q1 replicated under min-poems-per-period thresholds |
| 02bcmp | `02_modeling_robustness_offset_comparison.py` | Join Q1 offset GLM CSVs (long/wide + forest plots) |
| 02bsc | `02_modeling_specification_curve.py`        | Build specification-curve tables/figure across Q1-family reasonable specs |
| 02c | `02_modeling_q2_hierarchical.py`             | **RQ2**: hierarchical NB random-slope with posterior direction probability and q-direction summaries |
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
│   ├── 00_*.py                        # Filtering & corpus build (00a–00e)
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
        ├─► 00e_compute_finite_verb_exposure.py ──► data/To_run/00_filtering/stanza_finite_verb_counts.csv
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
# Stanza Ukrainian models (for `00e` finite-verb precompute and `02bvl` validation)
python -m stanza.download uk
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

## Method Notes

- `exposure_n_stanzas == 0` rows are excluded from offset models by `include_in_offset_models`; retain audit outputs when reporting selection effects.
- Q2 NB dispersion prior currently follows Bambi default family priorization (including HalfNormal-scale components for hierarchical terms); document this explicitly in methods write-up.
- Frequentist BH in Q1 remains within-stratum; Q2 uses posterior-direction false-sign risk columns plus BH-style `q_direction` summaries for author-level ranking.

## Key Dependencies

- **NLP**: spaCy, Stanza (`stanza` package; install Ukrainian models for `00e` / FV offset), Transformers
- **Statistics**: scipy, statsmodels, ruptures, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Annotation**: OpenAI API (GPT-4o-mini), Streamlit (legacy IAA app in `src/utils/_archive/`)
