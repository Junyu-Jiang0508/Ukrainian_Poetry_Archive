# Ukrainian Poetry Pipeline (Publication Workflow)

This document defines the canonical stage order, artifacts, and run commands.
Use it as the single source of truth for reproducible analysis.

## Quick Start

```bash
PYTHONPATH=src python src/00_pipeline_orchestrator.py --list
PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 13 --to-stage 18
```

## Stage Map

| Stage | Script | Purpose | Primary Outputs |
|---|---|---|---|
| 00a | `src/00_filtering.py` | Split raw posts into poem/stanza layers | `data/To_run/00_filtering/layer0_*.csv`, `layer1_*.csv` |
| 00b | `src/00_gpt_human_review_batch.py` | GPT adjudication for uncertain layer0 split rows | adjudication JSON + merged overrides |
| 01 | `src/01_pronoun_detection.py` | Detect explicit + dropped pronouns | `outputs/01_pronoun_detection/ukrainian_pronouns_detailed.csv` |
| 02 | `src/02_pronoun_cooccurrence.py` | High-frequency pronoun-word network plots | static PNG + interactive HTML |
| 03 | `src/03_pronoun_semantic_space.py` | FastText/UMAP semantic clustering + drift | UMAP CSV/HTML + cluster labels |
| 04 | `src/04_annotation_toolkit.py` | Annotation support utilities | sampled/QA outputs |
| 05 | `src/05_annotate_pronouns.py` | Heuristic pilot annotation on sample | annotated 50-row CSV |
| 06 | `src/06_pronoun_projection.py` | Translation/alignment projection baseline | projection final CSV |
| 07 | `src/07_gpt_annotation.py` | Main stanza-level GPT annotation | detailed annotation CSV + jsonl |
| 08 | `src/08_adaptive_binning.py` | Adaptive temporal intervals | binned time-series CSV |
| 09 | `src/09_breakpoint_regression.py` | Breakpoint/changepoint modeling | regression/changepoint tables |
| 10 | `src/10_public_list_filter.py` | Public list corpus filtering | public corpus + derivative CSVs |
| 11a | `src/11_layer0_layer1_to_run_filter.py` | Public-list sync for To_run layer0/layer1 | trimmed layer0/layer1 |
| 11b | `src/11_gpt_annotate_full.py` | Wrapper for full GPT run modes | delegated run artifacts |
| 12 | `src/12_gpt_annotation_exploration.py` | Exploratory diagnostics | field/crosstab summaries |
| 13 | `src/13_descriptive_statistics.py` | Descriptive and methodology tables | A/B/C appendix tables |
| 14 | `src/14_rq1_marginal_distributions.py` | RQ1 marginals by core periods | section 1 tables/plots |
| 15 | `src/15_key_cross_distributions.py` | RQ2 key cross-distributions | section 2 tables/plots |
| 16 | `src/16_author_heterogeneity.py` | RQ3 heterogeneity robustness | section 3 outputs |
| 17 | `src/17_poem_level_metrics.py` | Poem-level features | section 4 outputs |
| 18 | `src/18_publication_priority_figures.py` | Final publication priority figures | P1-P4 figures/tables |

## Recommended Reproducible Runs

- **Tables/Figures only (paper-facing)**  
  `PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 13 --to-stage 18`

- **Public-list GPT path (full analysis)**  
  `PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 10 --to-stage 18`

- **Dry run command check**  
  `PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 00a --to-stage 03 --dry-run`

## Naming and Structure Conventions

- Stage scripts use numbered filenames (`00` to `18`) to preserve execution order.
- Shared reusable logic lives under `src/utils/` (no duplicated logic across stage scripts).
- Artifacts default to deterministic stage-specific directories.
- New stage additions should update:
  - `src/utils/pipeline_catalog.py`
  - this workflow document
