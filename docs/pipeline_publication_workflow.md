# Ukrainian Poetry Pipeline (Publication Workflow)

This document mirrors the canonical stage catalog in
`src/utils/pipeline_catalog.py`. If you add or rename a stage, update the
catalog *and* this table.

## Quick Start

```bash
PYTHONPATH=src python src/00_pipeline_orchestrator.py --list
PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 02a --to-stage 03b
```

## Stage Map

| Stage | Script | Purpose | Primary Outputs |
|-------|--------|---------|-----------------|
| 00a | `src/00_filtering.py`                                | Split raw posts into poem/stanza layers           | `data/To_run/00_filtering/layer0_*.csv`, `layer1_*.csv` |
| 00b | `src/00_gpt_human_review_batch.py`                   | GPT adjudication for uncertain layer0 split rows  | adjudication JSON + merged overrides |
| 00c | `src/00_public_list_filter.py`                       | Build public-list corpus + derivative CSVs        | `data/processed/ukrpoetry_database_public_list.csv` + derivatives |
| 00d | `src/00_layer0_layer1_to_run_filter.py`              | In-place trim of `data/To_run/00_filtering/`      | filtered layer0/layer1 + review-queue sync |
| 01a | `src/01_annotation_pronoun_detection.py`             | Morphological pronoun detection                   | `outputs/01_annotation_pronoun_detection/ukrainian_pronouns_detailed.csv` |
| 01b | `src/01_annotation_toolkit.py`                       | Sampling / annotation prep                        | sampled / QA outputs |
| 01c | `src/01_annotation_rule_annotate_pronouns.py`        | Rule-based pilot annotation                       | annotated pilot CSV |
| 01d | `src/01_annotation_gpt_annotation.py`                | Stanza-level GPT annotation engine                | `data/Annotated_GPT/pronoun_annotation.csv` (+ `*_raw.jsonl`) |
| 01e | `src/01_annotation_gpt_annotate_full.py`             | Wrapper running 01d with `--source public`        | same artifacts under public-source dir |
| 01g | `src/01_annotation_source_pronoun_detection.py`     | **Canonical** full v2 depparse detection (no GPT) | `data/Annotated_Source/tokens_v2_full.csv` |
| 01h | `src/01_annotation_export_gpt_compatible.py`        | Map v2 tokens → downstream schema                 | `data/Annotated_Source/pronoun_annotation_v2.csv` |
| 02a | `src/02_modeling_significance_core_contrasts.py`     | Two-period confirmatory contrasts + sensitivity   | core contrast tables |
| 02b | `src/02_modeling_q1_per_cell_glm.py`                 | **RQ1**: per-cell one-vs-rest GLM (poem & stanza) | `outputs/02_modeling_q1_per_cell_glm/q1_*.csv` |
| 02c | `src/02_modeling_q2_hierarchical.py`                 | **RQ2**: hierarchical random-slope by author      | `outputs/02_modeling_q2_hierarchical/q2_*.csv` |
| 02d | `src/02_modeling_significance_models.py`             | Model-based inference for pronoun shifts          | inference tables |
| 02e | `src/02_modeling_significance_publication_figures.py`| Publication figures from 02d                      | publication PNG/PDF |
| 02f | `src/02_modeling_typology_and_period_models.py`      | Typology + period cohort models                   | typology tables |
| 03a | `src/03_reporting_descriptive_statistics.py`         | Methodology + corpus overview tables              | A/B/C appendix tables, adaptive_intervals.csv |
| 03b | `src/03_reporting_roster_freeze.py`                  | Author roster freeze + diagnostics                | frozen roster CSV |

## Data Inputs (canonical)

All `02*` modeling stages and the `03*` reporting stages read the canonical
source-side annotation table (resolved via
`utils.workspace.canonical_pronoun_annotation_csv`):

```
data/Annotated_Source/pronoun_annotation_v2.csv
```

This is produced by stages **01g** (`01_annotation_source_pronoun_detection.py
--full --mode v2`, depparse detection from the Ukrainian/Russian source text — no
GPT translation) and **01h** (`01_annotation_export_gpt_compatible.py`, schema map).
It supersedes the GPT table `data/Annotated_GPT_rerun/pronoun_annotation.csv`
(stages 01d/01e, retained for provenance). Override per-run with `--input` on any
stage or with the `PRONOUN_ANNOTATION_CSV` environment variable.

## Recommended Reproducible Runs

- **Modeling + reporting only (paper-facing)**  
  `PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 02a --to-stage 03b`

- **Public-list build + everything downstream**  
  `PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 00c --to-stage 03b`

- **Dry-run command check**  
  `PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 00a --to-stage 03b --dry-run`

## Naming and Structure Conventions

- Stage scripts use `<group>_<slug>.py` filenames; the orchestrator orders them by the catalog.
- Shared logic lives under `src/utils/` and is imported as `utils.<module>` (run with `PYTHONPATH=src`).
- Archived methodology scripts (manual annotation, IAA validation, GPT/human merge) live in `src/utils/_archive/` and are not part of any live stage.
- New stages must update both `src/utils/pipeline_catalog.py` and this table.
