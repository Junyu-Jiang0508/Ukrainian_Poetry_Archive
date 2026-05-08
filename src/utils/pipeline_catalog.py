"""Canonical ordered stage catalog for end-to-end analysis runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.workspace import repository_root


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    script_path: Path
    title: str
    description: str
    extra_args: tuple[str, ...] = ()


def build_pipeline_catalog() -> list[PipelineStage]:
    """Return numbered stages in execution order."""
    root = repository_root()
    src = root / "src"
    rows: list[tuple[str, Path, str, str, tuple[str, ...]]] = [
        ("00a", src / "00_filtering.py", "Layer 0/1 Filtering", "Split posts into poems and stanza rows.", ()),
        ("00b", src / "00_gpt_human_review_batch.py", "Human Review Batch", "Batch adjudication for uncertain split rows.", ()),
        ("00c", src / "00_public_list_filter.py", "Public-list Filter", "Filter public corpus and derivative files.", ()),
        ("00d", src / "00_layer0_layer1_to_run_filter.py", "To-run Filter", "Trim To_run layer0/layer1 with public-list rules.", ()),
        (
            "00e",
            src / "00e_compute_finite_verb_exposure.py",
            "Finite-verb precompute",
            "Stanza finite-verb counts (morph pipeline) → data/To_run/00_filtering/stanza_finite_verb_counts.csv.",
            (),
        ),
        ("01a", src / "01_annotation_pronoun_detection.py", "Pronoun Detection", "Detect explicit and dropped pronouns.", ()),
        ("01b", src / "01_annotation_toolkit.py", "Annotation Toolkit", "Sampling, QA, and annotation prep utilities.", ()),
        ("01c", src / "01_annotation_rule_annotate_pronouns.py", "Rule Annotation", "Heuristic pilot labeling on sampled pronouns.", ()),
        ("01d", src / "01_annotation_gpt_annotation.py", "GPT Annotation", "Stanza-level GPT pronoun annotation.", ()),
        ("01e", src / "01_annotation_gpt_annotate_full.py", "Full GPT Runner", "Wrapper for full/public GPT annotation runs.", ()),
        (
            "01f",
            src / "01_annotation_vy_register_audit.py",
            "vy_register audit",
            "Manual QA package: full polite-singular census + stratified true-plural sample and optional agreement scoring.",
            (),
        ),
        # --- Core analyses (frequentist, 4-cell unless flagged Bayesian) ---
        ("02a", src / "02_modeling_significance_core_contrasts.py", "Core Contrasts", "Two-period confirmatory contrasts and sensitivity.", ()),
        ("02b", src / "02_modeling_q1_per_cell_glm.py", "Q1 Per-cell GLM (stanza offset)", "Per-cell Poisson with stanza offset (4-cell primary inference).", ()),
        ("02b2", src / "02_modeling_q1_per_cell_glm.py", "Q1 Per-cell GLM (token offset)", "Same script, --exposure-type=n_tokens; written side-by-side as token-offset sensitivity.", ("--exposure-type", "n_tokens")),
        ("02b3", src / "02_modeling_q1_per_cell_glm.py", "Q1 Per-cell GLM (finite-verb offset)", "Same script, --exposure-type=n_finite_verbs; syntactic-slot exposure sensitivity.", ("--exposure-type", "n_finite_verbs")),
        (
            "02b4",
            src / "02_modeling_q1_per_cell_glm.py",
            "Q1 Per-cell GLM (FV excl imperative)",
            "Same script, --exposure-type=n_finite_verbs_excl_imperative; imperative-excluded FV offset sensitivity.",
            ("--exposure-type", "n_finite_verbs_excl_imperative"),
        ),
        (
            "02bvl",
            src / "02_modeling_finite_verb_validation_sample.py",
            "FV validation sample",
            "Stratified Stanza token export + morph vs depparse pipeline agreement diagnostics.",
            (),
        ),
        ("02bq1c", src / "02_modeling_q1c_pre_invasion_cohort.py", "Q1c Pre-invasion Cohort", "Exploratory Q1 GLM restricted to authors with first observed year ≤ 2014 (not in main BH family).", ()),
        ("02bq1b", src / "02_modeling_q1b_within_author_fe.py", "Q1b Author×Period FE + Bootstrap", "Parametric Poisson FE (HC1 + strict per-cell filter) and per-author δ bootstrap.", ()),
        ("02bq3", src / "02_modeling_q3_sparse_2pl_aggregated.py", "Q3 Sparse legacy 2pl", "Supplementary author×period legacy-2pl aggregation models.", ()),
        # --- Sensitivity / robustness ---
        ("02brobp", src / "02_modeling_robustness_period_definitions.py", "Robustness Period Specs", "Q1 replicated under alternate period encodings.", ()),
        ("02broba", src / "02_modeling_robustness_author_filter.py", "Robustness Author Thresholds", "Q1 replicated under roster-style min poems per period.", ()),
        (
            "02bcmp",
            src / "02_modeling_robustness_offset_comparison.py",
            "Offset comparison",
            "Join Q1 offset GLM CSVs into long/wide tables and forest plots (3- or 4-way).",
            (),
        ),
        (
            "02bsc",
            src / "02_modeling_specification_curve.py",
            "Specification curve",
            "Aggregate Q1/Q1c robustness combinations into specification-curve table and figure.",
            (),
        ),
        # --- Bayesian path (5-cell, retains polite-singular under shrinkage) ---
        ("02c", src / "02_modeling_q2_hierarchical.py", "Q2 Hierarchical (5-cell Bayesian)", "Per-cell hierarchical NB with author random slopes; PRIMARY_GLM_CELLS_BAYESIAN.", ()),
        # --- Other modeling + figures ---
        ("02d", src / "02_modeling_significance_models.py", "Significance Models", "Model-based inference for pronoun shifts.", ()),
        ("02e", src / "02_modeling_significance_publication_figures.py", "Significance Figures", "Publication figures for inferential outputs.", ()),
        ("02f", src / "02_modeling_typology_and_period_models.py", "Typology + Period Models", "Typology and period-based cohort models.", ()),
        ("03a", src / "03_reporting_descriptive_statistics.py", "Descriptive Statistics", "Methodology and corpus overview tables.", ()),
        ("03b", src / "03_reporting_roster_freeze.py", "Roster Freeze", "Author roster freeze and diagnostics.", ()),
    ]
    return [
        PipelineStage(
            stage_id=stage_id,
            script_path=script_path,
            title=title,
            description=description,
            extra_args=tuple(extra_args),
        )
        for stage_id, script_path, title, description, extra_args in rows
    ]


def stage_by_id(stage_id: str) -> PipelineStage | None:
    """Lookup a single stage by exact ID."""
    wanted = stage_id.strip().lower()
    for stage in build_pipeline_catalog():
        if stage.stage_id.lower() == wanted:
            return stage
    return None
