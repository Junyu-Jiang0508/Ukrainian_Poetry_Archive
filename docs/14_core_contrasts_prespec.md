# 14 Core Contrasts Pre-specification

Date: 2026-04-30

This document locks the design before running the next round of author-trajectory main analyses.

## Scope and Research Question

Primary question (non-directional): among sustained poets who appear in both 2014 and 2022+ bins, do within-author pronoun shifts show:

1. a period effect,
2. asymmetry in person x number structure,
3. language heterogeneity.

No directional claim (e.g., "1st-person internal restructuring") is pre-assumed at this stage.

## Data and Units

- Source: `data/Annotated_GPT_rerun/pronoun_annotation.csv`
- Filtering: exclude repeat/translation poems using existing pipeline logic.
- Main panel: authors present in both `2014` and `2022+` bins.
- Main analysis unit: author-level equal-weight (author x period aggregates).
- Supported by diagnostics:
  - cross-period authors (2014 & 2022+): 38
  - among these, dominant-language composition: Ukrainian 31, Russian 7, Qirimli 0

## Primary Analysis Choice

- Main analysis: author-level equal-weight paired design.
- Poem-level pooled models are prespecified as robustness/appendix only.
- We do not rewrite main directional conclusions based on appendix-only checks.

## Confirmatory Family (BH corrected within family)

Family size = 4 tests.

1. `Delta P(1pl | 1st)` from 2014 to 2022+ (within-author change)
2. `Delta P(2pl | 2nd)` from 2014 to 2022+ (within-author change)
3. Difference-in-differences:
   - `[Delta P(1pl | 1st)] - [Delta P(2pl | 2nd)]`
4. Language heterogeneity omnibus for (1):
   - because Qirimli has no cross-period authors in this panel, this becomes Ukrainian vs Russian omnibus/single-df contrast.

Any additional term (including 3rd-person cells, exploratory language splits, or slope variants) is exploratory and not part of this confirmatory FDR family.

## Reporting Commitments

- Report effect sizes and 95% CIs for all confirmatory tests.
- Report raw p-values and BH-adjusted q-values (within the 4-test family).
- Clearly label exploratory analyses as exploratory.
- Keep null findings in main narrative where they are directly tied to confirmatory contrasts.

## Robustness Plan (Prespecified)

- Appendix robustness:
  - poem-level pooled/clustered models,
  - alternative weighting choices.
- Robustness outputs are transparency checks, not grounds for changing the confirmatory family or post-hoc redefining primary claims.

