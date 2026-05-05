# Preregistration v2: Typology, Atypicality, and Qualitative Sampling

Signed date: 2026-05-05

Amended date: 2026-05-05 (post-audit)
- Amendment rationale: original 1 SD deviation rule showed low discriminating power at small N and conflated directional deviation with same-direction intensity.
- Amendment rule: replace 1 SD atypical labeling with sign-discordance against cohort medians, applied uniformly to the v2 reanalysis.

## Scope

- Cohort is fixed to the **analysis-feasible 9-author roster** in `outputs/15_roster_freeze/roster_v2_n12ge5_frozen.csv`.
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
- Signs map to candidate types; magnitude is reported separately as intensity.

## Typical vs Atypical Threshold

Direction and magnitude are explicitly separated:

- Compute cohort medians for `P3_vs_P2_1pl_cell_shift` and `P3_vs_P2_2sg_cell_shift`.
- A poet is **direction-discordant** for a contrast if:
  - `sign(author_estimate) != sign(cohort_median)` and both are non-zero.
- A poet is **atypical-core** if direction-discordant on both 1pl and 2sg contrasts.
- A poet is **atypical-broad** if direction-discordant on either contrast.
- A poet is **extreme-same-direction** if direction concordant but `|author_estimate - cohort_median| >= 1.5 * cohort_sd`.

## Qualitative Sampling Rule (Fixed Ex Ante)

- Primary close reading set: all **atypical-core** poets.
- If atypical-core size is below 2, extend with atypical-broad poets by largest absolute directional gap.
- For each selected poet, sample 5 poems from period `P3_2022plus`.
- Sampling is random without replacement with fixed seed `20260505`.
- If fewer than 5 poems exist in P3, include all available P3 poems and document the shortfall.

## Visualization Rule

- Period-based visualization is primary (P1/P2/P3), not yearly trajectory lines.
- Yearly plots are supplementary only due to small-n leverage risk in sparse years.

## Reporting Rule

- Main text reports point estimates, 95% CI, p-values, and BH-q for confirmatory contrasts.
- Wild cluster bootstrap remains supplementary sensitivity for transparency (no population-inference claim).
- Any blended B/C profile is reported as exploratory annotation only (not a new preregistered type).
