# "Someday we'll become a people:" The Poetic Plural in Wartime Ukraine

Computational analysis of pronoun use in Ukrainian poetry posted to Facebook, 2014–2025.

**Junyu Jiang** · Advisor: **Prof. Amelia Glaser** (UC San Diego)

---

## Overview

We test whether Ukrainian poets shifted from the first-person singular ("I") to
the plural ("we") after Russia's full-scale invasion in February 2022. Rather than
comparing different poets, we compare each poet to their own earlier work
(within-author design). The headline finding: there is **no shift at the corpus
level**, but a clear one **among poets who remained in Ukraine** — and the wartime
"we" splits into an *exhortational* "we" (poets further from the front) and a
*testimonial* "we" (poets inside the fighting).

## Data

- Source: Contemporary Ukrainian Poetry Archive (Glaser, 2024)
- After filtering: **1,601 poems by 105 authors** (2014–2025)
- Modeled roster: **33 authors** with ≥5 poems on each side of the 2022 cutpoint
- Canonical input: `data/Annotated_GPT_rerun/pronoun_annotation.csv`

## Method

1. **Pronoun recovery.** Ukrainian and Russian are null-subject languages, so most
   subjects are implied. We use the [Stanza](https://stanfordnlp.github.io/stanza/)
   pipeline to recover dropped subjects and tag each as 1sg / 1pl / 2sg / 2pl.
2. **Modeling.** Per-cell within-author GLMs (Poisson + negative-binomial, token
   offset) for absolute rate, and a hierarchical NB model with author random slopes
   for individual trajectories.
3. **Collocations.** Dependency-parsed period-differential collocates reveal *what
   the "we" governs* — notably the deontic verb мусити ("must"), absent before 2022.
4. **Temporal check.** Adaptive binning + PELT change-point detection around 2022.

## Repository

```
src/
  00_*  Filtering & corpus build
  01_*  Pronoun annotation (spaCy + GPT)
  02_*  Modeling (GLM, hierarchical, collocations)
  03_*  Reporting tables & figures
  utils/  Shared library code
data/     (gitignored) corpus, annotations
outputs/  (gitignored) per-stage results
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download uk_core_news_sm
python -m stanza.download uk
```

GPT annotation needs an OpenAI key in `.env`:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
# List all stages in order
PYTHONPATH=src python src/00_pipeline_orchestrator.py --list

# Run a range
PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 02a --to-stage 03b
```
