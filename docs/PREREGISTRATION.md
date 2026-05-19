# Pre-registration / Analysis-Freeze Document

**Project**: *The Grammar of Belonging — Computational Analysis of Pronominal Deixis in Wartime Ukrainian Poetry (2014--2025)*
**Authors**: Junyu Jiang · Amelia Glaser
**Repository tag at freeze**: *to be tagged at submission time; recommend* `analysis-frozen-YYYYMMDD`
**Document version**: drafted alongside the P2-1 / P2-5 refactor (2026-05-19)

This document records every analytic choice made *before* the manuscript was
shown to anyone. Subsequent changes to the analysis must either be made under
a new tag or labelled `exploratory` and reported as such. Reviewers reading
the manuscript should expect every claim made under §Results to map to a
contrast pre-specified here.

---

## 1. Estimands

We distinguish two confirmatory estimands. Both are computed; only one is
treated as primary inference.

### 1.1 Primary: Attention Allocation (02a)

*"Within the closed first/second-person quartet $\{$1sg, 1pl, 2sg, 2pl_vy_true_plural$\}$,
do within-author probabilities of selecting each cell shift across the 2022 cutpoint?"*

* Outcome variable: per-poem per-cell binomial trial count, with the per-poem
  sum across the four cells as the trial total ``n_12``.
* Model: closed-denominator binomial GLMM with random author intercept
  (``lme4::glmer`` via ``src/utils/r_glmm_runner.py``).
* Co-primary: Chamberlain conditional logit (``survival::clogit`` via
  ``src/utils/r_clogit_runner.py``).
* Sensitivity: unconditional MLE with $+C(\text{author})$ fixed effects,
  reported as a sensitivity only.

### 1.2 Secondary: Absolute Salience (02b)

*"Conditional on the attention reallocation, does any individual cell rise
or fall in absolute rate per unit of text after 2022?"*

* Outcome variable: per-poem per-cell count.
* Model: per-cell Poisson with cluster-robust SE on author + wild-cluster
  bootstrap p-value (Rademacher weights, 1,999 reps); co-primary Negative
  Binomial with cluster-robust SE.
* Primary offset: ``log(exposure_n_tokens)`` (post-P0-3 refactor).
* Sensitivity offsets: stanza count, finite-verb count, finite-verb count
  excluding imperatives.

### 1.3 Author Heterogeneity (02c)

* Hierarchical negative-binomial random-slope model in Bambi/PyMC.
* Reported via posterior probability of direction
  $\Pr(\delta_i > 0 \mid \text{data})$ with a BH-style
  $q_{\text{direction}}$ across authors within each cell.

---

## 2. Pre-registered cell set

* Frequentist cell set (02a, 02b, robustness): four cells
  $\{$1sg, 1pl, 2sg, 2pl_vy_true_plural$\}$.
* Bayesian cell set (02c): five cells; adds ``2pl_vy_polite_singular``
  because random-slope shrinkage stabilizes a sparse cell that MLE cannot.
* `2pl_vy_polite_singular` has 23 events / 18 poems / 13 authors with only 3
  authors contributing events in both periods; near-complete separation in
  unconditional MLE motivates its exclusion from the frequentist family.

---

## 3. Pre-registered period coding

* Three-way categorical: ``pre_2014`` (descriptive only; <10 poems),
  ``P1_2014_2021`` (reference), ``P2_2022_plus`` (treatment).
* Period assignment uses *composition year* (``year``) where available.
  The robustness specification ``invasion_20220224`` uses Facebook posting
  date and is interpreted as an *alternate estimand*, not a same-estimand
  sensitivity check (because composition and posting denote different events).

---

## 4. Pre-registered multiplicity control

* **BH-FDR within stratum × within cell-family**: for the four-cell
  frequentist family $\{1\text{sg},1\text{pl},2\text{sg},2\text{pl\_vy\_true\_plural}\}$
  in each of the two primary strata (Ukrainian, Russian). Pooled
  Ukrainian∪Russian is reported descriptively, not in the BH family.
* **Specification curve joint inference**: Romano-Wolf step-down across
  specifications (token / stanza / finite-verb offsets × period codings ×
  roster thresholds × ratio variants), reported alongside Holm-Bonferroni
  for a no-dependence baseline (P2-4).

---

## 5. Pre-registered roster

* `outputs/03_reporting_roster_freeze/roster_v1_frozen.csv` is the canonical
  author roster. Inclusion rule: author must have ≥ 8 poems in *each*
  period (P1 and P2). 33 authors qualify; 1 explicitly excluded under
  bilingual-switcher review.
* Sensitivity: ``min_per_period >= 6``, ``>= 10``, and leave-one-author-out
  variants are pre-registered as sensitivities (already implemented in
  ``02_modeling_significance_core_contrasts.py``).

---

## 6. Pre-registered language strata

* Primary strata for BH-FDR: ``Ukrainian``, ``Russian``.
* Descriptive stratum (not in BH): ``pooled_Ukrainian_Russian``.
* Excluded from inference (grammatical non-comparability): ``Qirimli`` /
  Crimean Tatar and any mixed-language code.

---

## 7. Pre-registered sensitivities (02bsc family)

* Offset: ``n_tokens`` (primary, post-P0-3), ``n_stanzas``,
  ``n_finite_verbs``, ``n_finite_verbs_excl_imperative``.
* Period coding: composition-year three-way (primary), drop-pre-2014,
  posting-date 24 Feb 2022 cutoff (alternate estimand),
  author-onset ≤ 2014 cohort.
* Author threshold: roster_ge8 (primary), drop_2014, drop_switchers,
  leave-one-author-out (each).

---

## 8. Pre-registered exploratory analyses

The following are exploratory and *not* in the BH family. They support
qualitative interpretation; their p-values are not claimed as confirmatory.

* P1-A/B: dependency-parsed pronoun collocations + period-differential
  log-likelihood per (cell, deprel, head_lemma).
  Multiplicity: per-cell BH; reported as descriptive evidence.
* P1-C: static-vector semantic drift via FastText + Procrustes.
* P1-D: stanza-level sentiment via XLM-R; sentiment × cell × period mixed
  model with author random intercept.
* P1-E: pronoun co-occurrence ego networks by cell × period.
* P1-F: BERTopic topic assignments, intended for use as a covariate in
  re-fits of 02a / 02b / 02c; reported as sensitivity.
* P2-2: smooth-year B-spline GLM and PELT change-point bootstrap CI; tests
  whether the data-driven breakpoint covers 2022.

---

## 9. Deferred / not in this study

* Inclusive vs exclusive 1PL classification of ``ми`` is *deferred*.
  Bracketed in the manuscript (Introduction); no claim about referent
  composition is made in confirmatory inference.
* GPT-based pronoun-referent clustering is *not* performed because no
  human-validated gold standard for referents is available.
* Genre / poetic form annotation (sonnet vs. free verse vs. prose poem)
  is *not* included; reported as a limitation.
* **Poem-time-of-writing location** is *unobserved*; we adjust only on
  time-invariant author features (see §10).

---

## 10. Author-covariate time-anchoring (post-refactor)

Author-level predictors are partitioned by their time anchor; we use only
the time-invariant subset as group-level predictors in within-author ×
period models. Snapshot covariates (single time-point observations) are
*not* used as static controls because emigration after 24 February 2022
is itself a consequence of the period effect, so adjusting on
``Current location`` would induce post-treatment / collider bias.

**Schema** (see ``src/utils/author_covariates.py``):

* **Time-invariant** (safe as group-level predictors on the period slope):
  ``gender``, ``birth_year``, ``generation_cohort``, ``region_of_birth``.
* **Single-time-point snapshot** (descriptive only; not a control):
  ``region_jan2022`` (~21/281 authors with data, pre-invasion),
  ``region_at_archive_freeze`` (post-2022 snapshot),
  ``language_xlsx_primary_at_freeze``
  (xlsx ``Primary language``; the source does not document which year
  this reflects).
* **Empirical, period-resolved** (preferred period-specific controls
  because the time anchor matches the inferential unit):
  ``language_corpus_p1`` and ``language_corpus_p2`` from the corpus's
  by-period dominant-language flag in
  ``outputs/03_reporting_roster_freeze/roster_v1_frozen.csv``;
  ``bilingual_switcher_corpus`` from the same source.

**Recommended group-level predictor set for 02c slope models**:

    period_slope_i ~ gender + generation_cohort + region_of_birth +
                     language_corpus_p1

**Strictly excluded from "control" status**: ``region_at_archive_freeze``,
``region_jan2022`` (small N), ``language_xlsx_primary_at_freeze``
(undocumented time anchor).

**Provenance**: every non-empty cell in ``data/author_covariates.csv``
traces to either ``data/raw/author.xlsx`` (with row index + column name +
raw value) or ``outputs/03_reporting_roster_freeze/roster_v1_frozen.csv``;
the full audit lives in
``data/author_covariates_provenance_audit.csv``. Editorial mappings
(city → region table, year → decade bucketing) are flagged with the
``EDIT_MAP_AUTHORED_BY_ASSISTANT`` provenance tag and are reviewable
against the source ``source_raw_value`` column.

**Vol'vach pending**: the source ``data/raw/author.xlsx`` lists
``павло вольвач`` as a Cyrillic alias of ``Pavlo Vol'vach``. The archive
team (Amelia, 2025-05) indicates that these are two distinct poets and
that source-side reconciliation is pending. The Cyrillic-lowercase row
in the covariates CSV is therefore deliberately blanked with a
``PENDING`` note in its ``notes`` field until the reconciliation lands.

---

## 11. Decision log

Any change to the contents of §§1–10 after the analysis-freeze tag must be
recorded here with date, reason, and the deviation's classification
(``confirmatory``, ``exploratory``, or ``deviation-reported-as-limitation``).

* *empty at freeze*

---

## 11. Reproduction

To reproduce every confirmatory output:

```bash
PYTHONPATH=src python src/00_pipeline_orchestrator.py --from-stage 02a --to-stage 03b
```

The orchestrator's stage catalog (`src/utils/pipeline_catalog.py`) is the
canonical execution order. Each stage's README in `outputs/<stage>/README.md`
explains the stage's outputs and any caveats. The current freeze state is
captured in the git tag.
