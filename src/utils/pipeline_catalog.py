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


def build_pipeline_catalog() -> list[PipelineStage]:
    """Return numbered stages in execution order."""
    root = repository_root()
    src = root / "src"
    rows = [
        ("00a", src / "00_filtering.py", "Layer 0/1 Filtering", "Split posts into poems and stanza rows."),
        ("00b", src / "00_gpt_human_review_batch.py", "Human Review Batch", "Batch adjudication for uncertain split rows."),
        ("00c", src / "00_public_list_filter.py", "Public-list Filter", "Filter public corpus and derivative files."),
        ("00d", src / "00_layer0_layer1_to_run_filter.py", "To-run Filter", "Trim To_run layer0/layer1 with public-list rules."),
        ("01a", src / "01_annotation_pronoun_detection.py", "Pronoun Detection", "Detect explicit and dropped pronouns."),
        ("01b", src / "01_annotation_toolkit.py", "Annotation Toolkit", "Sampling, QA, and annotation prep utilities."),
        ("01c", src / "01_annotation_rule_annotate_pronouns.py", "Rule Annotation", "Heuristic pilot labeling on sampled pronouns."),
        ("01d", src / "01_annotation_gpt_annotation.py", "GPT Annotation", "Stanza-level GPT pronoun annotation."),
        ("01e", src / "01_annotation_gpt_annotate_full.py", "Full GPT Runner", "Wrapper for full/public GPT annotation runs."),
        ("01f", src / "01_annotation_gpt_exploration.py", "Exploration", "Initial QA and distribution diagnostics."),
        ("02a", src / "02_modeling_pronoun_cooccurrence.py", "Co-occurrence Network", "Build high-frequency pronoun-word network."),
        ("02b", src / "02_modeling_pronoun_semantic_space.py", "Semantic Space", "FastText + UMAP clustering and drift."),
        ("02c", src / "02_modeling_pronoun_projection.py", "Cross-lingual Projection", "Projection-based pronoun alignment."),
        ("02d", src / "02_modeling_adaptive_binning.py", "Adaptive Binning", "Temporal adaptive interval construction."),
        ("02e", src / "02_modeling_breakpoint_regression.py", "Breakpoint Regression", "Interrupted trend and changepoint analysis."),
        ("02f", src / "02_modeling_significance_core_contrasts.py", "Core Contrasts", "Three-period confirmatory contrasts and sensitivity."),
        ("02g", src / "02_modeling_significance_models.py", "Significance Models", "Model-based inference for pronoun shifts."),
        ("02h", src / "02_modeling_significance_publication_figures.py", "Significance Figures", "Publication figures for inferential outputs."),
        ("02i", src / "02_modeling_typology_and_period_models.py", "Typology + Period Models", "Typology and period-based cohort models."),
        ("03a", src / "03_reporting_descriptive_statistics.py", "Descriptive Statistics", "Methodology and corpus overview tables."),
        ("03b", src / "03_reporting_roster_freeze.py", "Roster Freeze", "Author roster freeze and diagnostics."),
    ]
    return [
        PipelineStage(
            stage_id=stage_id,
            script_path=script_path,
            title=title,
            description=description,
        )
        for stage_id, script_path, title, description in rows
    ]


def stage_by_id(stage_id: str) -> PipelineStage | None:
    """Lookup a single stage by exact ID."""
    wanted = stage_id.strip().lower()
    for stage in build_pipeline_catalog():
        if stage.stage_id.lower() == wanted:
            return stage
    return None
