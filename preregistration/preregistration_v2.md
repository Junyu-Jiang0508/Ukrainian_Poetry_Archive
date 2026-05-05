# Preregistration v2: Typology, Atypicality, and Qualitative Sampling

Signed date: 2026-05-05

## Scope

- Cohort is fixed to the 12 included authors from `outputs/15_roster_freeze/roster_v1_frozen.csv`.
- Time periods are fixed: `P1_2014_18`, `P2_2019_21`, `P3_2022plus`.
- Confirmatory family remains the 6 contrasts in `contrasts_v1.md` (BH only within that family).

## A Priori Typology (Theory-Driven)

Typology is pre-specified before rerunning three-period models and before any qualitative case selection:

- **Type A (inclusive-we intensification)**: directional shift toward collective first-person plurality.
- **Type B (exclusive-we boundarying)**: directional shift toward first-person plurality used for out-group demarcation.
- **Type C (accusatory second-person turn)**: directional shift toward second-person addressivity.
- **Type D (introspective first-person singular turn)**: directional shift toward singular first-person introspection.

Operational assignment rule:
- Primary quantitative signals are per-author `P3_vs_P2_1pl_cell_shift` and `P3_vs_P2_2sg_cell_shift`.
- Signs/magnitudes map to candidate types; final qualitative coding uses this predeclared typology framework.

## Typical vs Atypical Threshold

- For each confirmatory contrast, compute the cohort median and standard deviation of per-author estimates.
- A poet is marked **atypical** for a contrast when:
  - `abs(author_estimate - cohort_median) >= 1 * cohort_sd`.
- A poet is globally atypical if atypical on either:
  - `P3_vs_P2_1pl_cell_shift` OR
  - `P3_vs_P2_2sg_cell_shift`.

## Qualitative Sampling Rule (Fixed Ex Ante)

- For each globally atypical poet, sample 5 poems from period `P3_2022plus`.
- Sampling is random without replacement with fixed seed `20260505`.
- If fewer than 5 poems exist in P3, include all available P3 poems and document the shortfall.

## Visualization Rule

- Period-based visualization is primary (P1/P2/P3), not yearly trajectory lines.
- Yearly plots are supplementary only due to small-n leverage risk in sparse years.

## Reporting Rule

- Main text reports point estimates, 95% CI, p-values, and BH-q for confirmatory contrasts.
- Wild cluster bootstrap remains supplementary sensitivity for transparency (no population-inference claim).
